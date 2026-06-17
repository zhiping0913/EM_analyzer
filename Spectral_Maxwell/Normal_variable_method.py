from functools import partial
import sys
import subprocess
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336')
from typing import Optional
from line_profiler import profile
import numpy as np


def _detect_gpu_count():
    """Count NVIDIA GPUs via `nvidia-smi` without initializing JAX."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return sum(1 for line in result.stdout.strip().split('\n') if line.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return 0


_n_gpus = _detect_gpu_count()
USE_GPU = _n_gpus >= 2

import jax
if USE_GPU:
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'gpu')
else:
    jax.config.update('jax_num_cpu_devices', 6)
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'cpu')
print(f"Backend: {'GPU' if USE_GPU else 'CPU'}, local device count: {jax.local_device_count()}", flush=True)
print(jax.devices(), flush=True)
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax import jit
import jax.numpy as jnp
import scipy.constants as C
from jax_array_info import sharding_info
from EM_analyzer.pretreat_fields import check_divergence,stack_Fields
from EM_analyzer.Spectral_Maxwell.kgrid import grid_k
from EM_analyzer.spectrum import get_spectrum_from_field_with_coordinate,get_field_from_spectrum_with_coordinate


if USE_GPU:
    # 2 GPUs: 1-D mesh on 'EM' axis (size 2, E vs B*c). Channel axis (size 3) is replicated.
    devices = jax.local_devices()[:2]
    mesh = Mesh(np.array(devices).reshape(2,), ('EM',))
    sharding_EM = NamedSharding(mesh, P('EM', None))   # (2, 3, Nx_pad, Ny_pad, Nz_pad): shard axis 0
    sharding_k  = NamedSharding(mesh, P())             # (3, Nx_pad, Ny_pad, Nz_pad): fully replicated
else:
    # 6 CPUs: 2-D mesh, 'EM' axis (size 2, E vs B*c) × 'channel' axis (size 3, x/y/z component).
    devices = jax.local_devices()[:6]
    mesh = Mesh(np.array(devices).reshape(2, 3), ('EM', 'channel'))
    sharding_EM = NamedSharding(mesh, P('EM', 'channel'))   # (2, 3, Nx_pad, Ny_pad, Nz_pad)
    sharding_k  = NamedSharding(mesh, P('channel'))         # (3, Nx_pad, Ny_pad, Nz_pad)

@profile
@partial(jit)
def evolution_t(omega_dot_t, EMk0, signed_k_cross_partner, k_dot_v_over_c):
    """Evolve stacked EM fields in k-space by time omega_dot_t/omega.

    Args:
        omega_dot_t: ω·t, shape: (Nx_pad, Ny_pad, Nz_pad). Unit: rad
        EMk0: Stacked initial EM fields, shape: (2, 3, Nx_pad, Ny_pad, Nz_pad).
              EMk0[0] = Ek0 (V·m³), EMk0[1] = Bk0*c (same units).
        signed_k_cross_partner: Pre-swapped and signed cross products, shape: (2, 3, Nx_pad, Ny_pad, Nz_pad).
              [0] = +i * (k_hat × (Bk0*c))  ← partner for E evolution
              [1] = -i * (k_hat × Ek0)       ← partner for B evolution
              Stored at init time to avoid jnp.roll and sign-multiply in the hot path.
        k_dot_v_over_c: k_hat·v/c precomputed outside JIT, shape: (Nx_pad, Ny_pad, Nz_pad).
              Avoids an all-reduce over the channel axis inside the JIT on every call.
    Returns:
        EMk_evolution_in_window: shape (2, 3, Nx_pad, Ny_pad, Nz_pad).
                                 [0] = Ek after evolution, [1] = Bk*c after evolution.
    """
    coswt_ = jnp.cos(omega_dot_t)[None, None, :, :, :]   #shape=(1, 1, Nx_pad, Ny_pad, Nz_pad)
    sinwt_ = jnp.sin(omega_dot_t)[None, None, :, :, :]

    # No roll, no signs multiply: both are absorbed into signed_k_cross_partner at init.
    # Ek_evo_c  = Ek0  * cos + (+i * k×Bk0_c) * sin
    # Bk_evo_c  = Bk0_c * cos + (-i * k×Ek0)  * sin
    EMk_evolution = EMk0 * coswt_ + signed_k_cross_partner * sinwt_   #shape=(2, 3, Nx_pad, Ny_pad, Nz_pad)

    # k_dot_v_over_c is precomputed in Python (cached if v unchanged) — no device communication here.
    window_phase_shift = k_dot_v_over_c * omega_dot_t   #shape=(Nx_pad, Ny_pad, Nz_pad)
    phase_factor = jnp.exp(1j * window_phase_shift)[None, None, :, :, :]   #shape=(1, 1, Nx_pad, Ny_pad, Nz_pad)
    EMk_evolution_in_window = EMk_evolution * phase_factor   #shape=(2, 3, Nx_pad, Ny_pad, Nz_pad)
    return EMk_evolution_in_window

@partial(jit)
def transverse_projection(Fk, k_hat, k_mask):
    """
    Project Fk onto divergence-free (transverse) subspace.
    Fk shape: (3, Nx, Ny, Nz)
    """
    Fk_proj = jnp.array(Fk, copy=True)   #shape=(3, Nx, Ny, Nz)
    k_dot_Fk = jnp.einsum("lijk,lijk->ijk", k_hat, Fk)   #shape=(Nx, Ny, Nz)   k_hat · Fk
    Fk_proj = jnp.where(k_mask[jnp.newaxis, :, :, :], 
                        Fk_proj- k_hat* k_dot_Fk[jnp.newaxis, :, :, :],   #shape=(3, Nx, Ny, Nz)
                        Fk_proj)   #k=0 mode
    return Fk_proj
@partial(jit)
def initialize_EMk0(EMk0, k_hat, k_mask):
    """Enforce transversality and precompute signed cross products for the stacked EM array.

    Args:
        EMk0: shape (2, 3, Nx_pad, Ny_pad, Nz_pad). [0]=Ek0, [1]=Bk0*c.
        k_hat: shape (3, Nx_pad, Ny_pad, Nz_pad).
        k_mask: shape (Nx_pad, Ny_pad, Nz_pad). True where k≠0.
    Returns:
        EMk0_proj: transversally-projected EMk0, shape (2, 3, ...), sharded with sharding_EM.
        signed_k_cross_partner: Pre-swapped, sign-absorbed cross products, shape (2, 3, ...),
            sharded with sharding_EM.
            [0] = +i * (k_hat × (Bk0*c))  ← partner for E evolution
            [1] = -i * (k_hat × Ek0)       ← partner for B evolution
            Pre-swapping avoids jnp.roll on the EM (device) axis at every evolution() call.
            Absorbing signs avoids a constant complex multiply in the hot path.
    """
    # Enforce transversality on each EM slice independently
    Ek0_proj = transverse_projection(EMk0[0], k_hat, k_mask)   #shape=(3, Nx_pad, Ny_pad, Nz_pad)
    Bk0_proj = transverse_projection(EMk0[1], k_hat, k_mask)   #shape=(3, Nx_pad, Ny_pad, Nz_pad)
    EMk0_proj = jnp.stack([Ek0_proj, Bk0_proj])                #shape=(2, 3, Nx_pad, Ny_pad, Nz_pad)
    # Cross products, stored in partner order with signs absorbed.
    # Partner for E (slot 0) is +i * k×(Bk0*c); partner for B (slot 1) is -i * k×Ek0.
    signed_k_cross_partner = jnp.stack([
         1j * jnp.cross(k_hat, EMk0_proj[1], axis=0),   #shape=(3, ...)  +i*(k×Bk0_c)
        -1j * jnp.cross(k_hat, EMk0_proj[0], axis=0),   #shape=(3, ...)  -i*(k×Ek0)
    ])   #shape=(2, 3, Nx_pad, Ny_pad, Nz_pad)
    return jax.device_put(EMk0_proj, sharding_EM), jax.device_put(signed_k_cross_partner, sharding_EM)



class Spectral_Maxwell_Solver:
    """
    Exact spectral Maxwell solver (vacuum).
    Fields are assumed transverse.
    This is explicitly checked at initialization.
    """
    def __init__(
        self, 
        E0: jnp.ndarray, B0: jnp.ndarray, 
        x_coordinate=[0],y_coordinate=[0],z_coordinate=[0],
        ):
        """
        Parameters
        ----------
        E0: jnp.ndarray
            Initial E fields, unit: V/m, shape (3, Nx, Ny, Nz)
        B0 : jnp.ndarray
            Initial B fields, unit: T, shape (3, Nx, Ny, Nz)
        x_coordinate : list
            x-axis grid points, unit: m, shape (Nx,)
        y_coordinate : list
            y-axis grid points, unit: m, shape (Ny,)
        z_coordinate : list
            z-axis grid points, unit: m, shape (Nz,)
        pad_width: tuple or None
            If not None, pad the input fields to the given width before initializing the solver. The pad_width should be a tuple of three integers (pad_x, pad_y, pad_z) specifying the number of points to pad on each axis. The padding will be applied symmetrically on both sides of each axis.
            If None, pad_width=(Nx,Ny,Nz)
        """
        self.x_coordinate = jnp.array(x_coordinate).flatten()
        self.y_coordinate = jnp.array(y_coordinate).flatten()
        self.z_coordinate = jnp.array(z_coordinate).flatten()
        self.Nx = self.x_coordinate.size
        self.Ny = self.y_coordinate.size
        self.Nz = self.z_coordinate.size
        assert self.Nx>0 and self.Ny>0 and self.Nz>0, "Grid axes must have at least one point."
        self.shape = (self.Nx, self.Ny, self.Nz)
        self.E0 = jnp.asarray(E0,dtype=jnp.float64)   #shape=(3, Nx, Ny, Nz)
        self.B0 = jnp.asarray(B0,dtype=jnp.float64)   #shape=(3, Nx, Ny, Nz)
        assert self.E0.shape == (3, self.Nx, self.Ny, self.Nz), f"E0 shape {self.E0.shape} does not match grid shape {(3, self.Nx, self.Ny, self.Nz)}"
        assert self.B0.shape == (3, self.Nx, self.Ny, self.Nz), f"B0 shape {self.B0.shape} does not match grid shape {(3, self.Nx, self.Ny, self.Nz)}"
        print(f"Initial field shapes {self.E0.shape} verified.", flush=True)
        # Check transversality in real space
        check_divergence(Field=self.E0, x_coordinate=self.x_coordinate, y_coordinate=self.y_coordinate, z_coordinate=self.z_coordinate)
        check_divergence(Field=self.B0, x_coordinate=self.x_coordinate, y_coordinate=self.y_coordinate, z_coordinate=self.z_coordinate)
        EM0 = jnp.stack([self.E0, self.B0 * C.speed_of_light])   #shape=(2, 3, Nx, Ny, Nz): [0]=E, [1]=B*c
        spectrum, k_coordinate_each_axis, pad_slices=get_spectrum_from_field_with_coordinate(
            field=EM0,   #shape=(2, 3, Nx, Ny, Nz)
            axis=[2,3,4],
            r_coordinate_each_axis=[self.x_coordinate, self.y_coordinate, self.z_coordinate],
            out_sharding=sharding_EM,
        )
        sharding_info(spectrum, "Initial spectrum from field")
        self.EMk0 = spectrum   #shape=(2, 3, Nx_pad, Ny_pad, Nz_pad): EMk0[0]=Ek0, EMk0[1]=Bk0*c
        self.kx_coordinate, self.ky_coordinate, self.kz_coordinate = k_coordinate_each_axis
        self.Nx_pad=self.kx_coordinate.size
        self.Ny_pad=self.ky_coordinate.size
        self.Nz_pad=self.kz_coordinate.size
        self.pad_slices = pad_slices
        print(f"Padded field shapes {self.EMk0.shape} verified.", flush=True)

        # k-space grid
        self.grid_k = grid_k(
            kx_coordinate=self.kx_coordinate,
            ky_coordinate=self.ky_coordinate,
            kz_coordinate=self.kz_coordinate,
        )
        self.k = self.grid_k.k   #shape=(3, Nx_pad, Ny_pad, Nz_pad)
        self.k_norm = self.grid_k.k_norm   #shape=(Nx_pad, Ny_pad, Nz_pad)
        self.k_hat = jax.device_put(self.grid_k.k_hat, sharding_k)   #shape=(3, Nx_pad, Ny_pad, Nz_pad)
        self.omega = C.speed_of_light * self.k_norm   #shape=(Nx_pad, Ny_pad, Nz_pad)
        print(f"k-space grid shapes {self.k.shape} verified.", flush=True)

        self.EMk0, self.signed_k_cross_partner = initialize_EMk0(self.EMk0, self.k_hat, self.grid_k.k_mask)
        print(f"Transversality enforced in k-space.", flush=True)
        # Cache for k·v/c — populated on first evolution() call, reused while v is unchanged.
        self._k_dot_v_over_c: Optional[jnp.ndarray] = None
        self._last_window_shift_velocity: Optional[jnp.ndarray] = None

    @profile
    def evolution(
        self, evolution_time=0.0,window_shift_velocity=jnp.array((0.0,0.0,0.0)),
        return_spectrum=False,
        ):
        """_summary_

        Args:
            evolution_time: evolution time. Unit: s
            window_shift_velocity: The (3d) velocity of the window following the evolution of the field. (vx,vy,vz). Unit: m/s
        Returns:
            EB_evolution_dict={
                "E": E_evolution_in_window,   #shape=(3, Nx, Ny, Nz), unit: V/m
                "B": B_evolution_in_window,   #shape=(3, Nx, Ny, Nz), unit: T
                "x_coordinate": window_x_coordinate,   #shape=(Nx,), unit: m
                "y_coordinate": window_y_coordinate,   #shape=(Ny,), unit: m
                "z_coordinate": window_z_coordinate,   #shape=(Nz,), unit: m
            }
        """
        window_shift_velocity=jnp.array(window_shift_velocity,dtype=jnp.float64).flatten()
        assert window_shift_velocity.shape==(3,)
        print(f"Evolution time: {evolution_time} s, window shift velocity: {window_shift_velocity} m/s", flush=True)
        # k·v/c: sum over the channel axis of k_hat (sharding_k), which requires an
        # all-reduce over 3 devices.  Cache the result — recompute only when v changes.
        if (self._last_window_shift_velocity is None or
                not jnp.array_equal(window_shift_velocity, self._last_window_shift_velocity)):
            self._k_dot_v_over_c = jnp.einsum(
                'lijk,l->ijk', self.k_hat, window_shift_velocity / C.speed_of_light
            )   #shape=(Nx_pad, Ny_pad, Nz_pad)
            self._last_window_shift_velocity = window_shift_velocity
        EMk_evolution_in_window = evolution_t(
            omega_dot_t=self.omega*evolution_time,
            EMk0=self.EMk0,
            signed_k_cross_partner=self.signed_k_cross_partner,
            k_dot_v_over_c=self._k_dot_v_over_c,
        )   #shape=(2, 3, Nx_pad, Ny_pad, Nz_pad)
        if return_spectrum:
            return {
                "Ek": EMk_evolution_in_window[0],   #shape=(3, Nx_pad, Ny_pad, Nz_pad), unit: V·m³
                "Bk": EMk_evolution_in_window[1] / C.speed_of_light,   #shape=(3, Nx_pad, Ny_pad, Nz_pad), unit: T·m
                "kx_coordinate": self.kx_coordinate,   #shape=(Nx_pad,), unit: rad/m
                "ky_coordinate": self.ky_coordinate,   #shape=(Ny_pad,), unit: rad/m
                "kz_coordinate": self.kz_coordinate,   #shape=(Nz_pad,), unit: rad/m
                "pad_slices": self.pad_slices,   #tuple of slice objects to unpad the spectrum if needed
            }
        else:
            window_x_coordinate=self.x_coordinate+window_shift_velocity[0]*evolution_time
            window_y_coordinate=self.y_coordinate+window_shift_velocity[1]*evolution_time
            window_z_coordinate=self.z_coordinate+window_shift_velocity[2]*evolution_time

            EM_field_evolution=get_field_from_spectrum_with_coordinate(
                spectrum=jax.device_put(EMk_evolution_in_window, sharding_EM),
                axis=[2,3,4],
                k_coordinate_each_axis=[self.kx_coordinate, self.ky_coordinate, self.kz_coordinate],
                pad_slices=self.pad_slices,
                real=True,
            )   #shape=(2, 3, Nx, Ny, Nz)

            E_evolution_in_window = EM_field_evolution[0]                      #shape=(3, Nx, Ny, Nz), unit: V/m
            B_evolution_in_window = EM_field_evolution[1] / C.speed_of_light   #shape=(3, Nx, Ny, Nz), unit: T

            return {
                "E": E_evolution_in_window,   #shape=(3, Nx, Ny, Nz), unit: V/m
                "B": B_evolution_in_window,   #shape=(3, Nx, Ny, Nz), unit: T
                "x_coordinate": window_x_coordinate,   #shape=(Nx,), unit: m
                "y_coordinate": window_y_coordinate,   #shape=(Ny,), unit: m
                "z_coordinate": window_z_coordinate,   #shape=(Nz,), unit: m
            }

class Spectral_Maxwell_Solver_1D():
    """
    1D version of Spectral_Maxwell_Solver, with fields and grid only along x axis.
    """
    def __init__(
        self, 
        E0x:Optional[jnp.ndarray]=None, E0y:Optional[jnp.ndarray]=None, E0z:Optional[jnp.ndarray]=None,
        B0x:Optional[jnp.ndarray]=None, B0y:Optional[jnp.ndarray]=None, B0z:Optional[jnp.ndarray]=None,
        x_coordinate=[0],
        ):
        self.x_coordinate = jnp.asarray(x_coordinate,dtype=jnp.float64).flatten()
        self.Nx=self.x_coordinate.size
        assert self.Nx>0, "Grid axis must have at least one point."
        
        E0=stack_Fields(Field_x=E0x, Field_y=E0y, Field_z=E0z)  #shape=(3, Nx,1,1)
        B0=stack_Fields(Field_x=B0x, Field_y=B0y, Field_z=B0z)  #shape=(3, Nx,1,1)
        assert E0.shape == (3, self.Nx, 1, 1), f"E0 shape {E0.shape} does not match grid shape {(3, self.Nx, 1, 1)}"
        assert B0.shape == (3, self.Nx, 1, 1), f"B0 shape {B0.shape} does not match grid shape {(3, self.Nx, 1, 1)}"
        print(f"Initial field shapes {E0.shape} verified.", flush=True)
        self.Solver=Spectral_Maxwell_Solver(
            E0=E0, B0=B0, 
            x_coordinate=x_coordinate, y_coordinate=[0], z_coordinate=[0],
            )
    def evolution(self, evolution_time=0.0,window_shift_velocity=0.0):
        """_summary_

        Args:
            evolution_time (float, optional): _description_. Defaults to 0.0.
            window_shift_velocity: vx. Defaults to 0.0.

        Returns:
            _type_: _description_
        """
        window_shift_velocity=jnp.pad(jnp.array(window_shift_velocity,dtype=jnp.float64).flatten(), pad_width=((0, 2),))   #shape=(3,)
        EB_evolution_dict=self.Solver.evolution(
            evolution_time=evolution_time,window_shift_velocity=window_shift_velocity
            )
        E_evolution_in_window=EB_evolution_dict["E"][:, :, 0, 0]   #shape=(3, Nx)
        B_evolution_in_window=EB_evolution_dict["B"][:, :, 0, 0]   #shape=(3, Nx)
        window_x_coordinate=EB_evolution_dict["x_coordinate"]   #shape=(Nx,)
        return {
            'Ex': E_evolution_in_window[0,:],
            'Ey': E_evolution_in_window[1,:],
            'Ez': E_evolution_in_window[2,:],
            'Bx': B_evolution_in_window[0,:],
            'By': B_evolution_in_window[1,:],
            'Bz': B_evolution_in_window[2,:],
            'x_coordinate': window_x_coordinate,
        }

class Spectral_Maxwell_Solver_2D():
    """
    2D version of Spectral_Maxwell_Solver, with fields and grid only along x and y axes.
    """
    def __init__(
        self, 
        E0x:Optional[jnp.ndarray]=None, E0y:Optional[jnp.ndarray]=None, E0z:Optional[jnp.ndarray]=None,
        B0x:Optional[jnp.ndarray]=None, B0y:Optional[jnp.ndarray]=None, B0z:Optional[jnp.ndarray]=None,    
        x_coordinate=[0], y_coordinate=[0],
        ):
        self.x_coordinate = jnp.asarray(x_coordinate,dtype=jnp.float64).flatten()
        self.y_coordinate = jnp.asarray(y_coordinate,dtype=jnp.float64).flatten()
        self.Nx=self.x_coordinate.size
        self.Ny=self.y_coordinate.size
        assert self.Nx>0 and self.Ny>0, "Grid axes must have at least one point."
        E0=stack_Fields(Field_x=E0x, Field_y=E0y, Field_z=E0z)  #shape=(3, Nx, Ny,1)
        B0=stack_Fields(Field_x=B0x, Field_y=B0y, Field_z=B0z)  #shape=(3, Nx, Ny,1)
        assert E0.shape == (3, self.Nx, self.Ny, 1), f"E0 shape {E0.shape} does not match grid shape {(3, self.Nx, self.Ny, 1)}"
        assert B0.shape == (3, self.Nx, self.Ny, 1), f"B0 shape {B0.shape} does not match grid shape {(3, self.Nx, self.Ny, 1)}"
        print(f"Initial field shapes {E0.shape} verified.", flush=True)
        self.Solver=Spectral_Maxwell_Solver(
            E0=E0, B0=B0, 
            x_coordinate=x_coordinate, y_coordinate=y_coordinate, z_coordinate=[0],
            )
    def evolution(self, evolution_time=0.0,window_shift_velocity=(0.0,0.0)):
        """_summary_

        Args:
            evolution_time (float, optional): _description_. Defaults to 0.0.
            window_shift_velocity: (vx,vy). Defaults to (0.0,0.0).

        Returns:
            _type_: _description_
        """
        window_shift_velocity=jnp.pad(jnp.array(window_shift_velocity,dtype=jnp.float64).flatten(), pad_width=((0, 1),))   #shape=(3,)
        EB_evolution_dict=self.Solver.evolution(
            evolution_time=evolution_time,window_shift_velocity=window_shift_velocity
            )
        E_evolution_in_window=EB_evolution_dict["E"][:, :, :, 0]   #shape=(3, Nx, Ny)
        B_evolution_in_window=EB_evolution_dict["B"][:, :, :, 0]   #shape=(3, Nx, Ny)
        window_x_coordinate=EB_evolution_dict["x_coordinate"]   #shape=(Nx,)
        window_y_coordinate=EB_evolution_dict["y_coordinate"]   #shape=(Ny,)
        return {
            'Ex': E_evolution_in_window[0,:,:],
            'Ey': E_evolution_in_window[1,:,:],
            'Ez': E_evolution_in_window[2,:,:],
            'Bx': B_evolution_in_window[0,:,:],
            'By': B_evolution_in_window[1,:,:],
            'Bz': B_evolution_in_window[2,:,:],
            'x_coordinate': window_x_coordinate,
            'y_coordinate': window_y_coordinate,
        }