from functools import partial
import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336')
from typing import Optional
from line_profiler import profile
import numpy as np

from EM_analyzer.device_config import configure_jax_backend
_backend_info = configure_jax_backend()
USE_GPU = _backend_info['USE_GPU']

import jax
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax import jit
import jax.numpy as jnp
import scipy.constants as C
from EM_analyzer.pretreat_fields import check_divergence,stack_Fields,print_shard_layout
from EM_analyzer.Spectral_Maxwell.kgrid import grid_k
from EM_analyzer.spectrum import get_spectrum_from_field_with_coordinate,get_field_from_spectrum_with_coordinate


# Use jax.devices() (global view) so multi-process / multi-node distributed
# runs see all devices across nodes, not just the local 3.
if USE_GPU:
    # 2 GPUs: 1-D mesh on 'EM' axis (size 2, E vs B*c). Channel axis (size 3) is replicated.
    devices = jax.devices()[:2]
    mesh = Mesh(np.array(devices).reshape(2,), ('EM',))
    sharding_EM = NamedSharding(mesh, P('EM', None))   # (2, 3, Nx_pad, Ny_pad, Nz_pad): shard axis 0
    sharding_k  = NamedSharding(mesh, P())             # (3, Nx_pad, Ny_pad, Nz_pad): fully replicated
else:
    # 6 CPUs (1 node × 6 devices, or 2 nodes × 3 devices via jax.distributed):
    # 2-D mesh, 'EM' axis (size 2, E vs B*c) × 'channel' axis (size 3, x/y/z component).
    # When distributed across 2 nodes, jax.devices() orders by (process_id, local_id),
    # so reshape(2, 3) puts node-0's 3 devices on EM=0 and node-1's 3 devices on EM=1.
    devices = jax.devices()[:6]
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


@partial(jit, static_argnames=('t_chunk_size',))
def _evolution_at_points_all_t(
    rts,
    t_arr,
    EMk0, signed_k_cross_partner, omega,
    kx, ky, kz,
    normalization,
    t_chunk_size,
):
    """Reconstruct E, B*c at every point in `rts` for every t in `t_arr`.

    Physical (Riemann-sum) inverse of the 0-centered spectrum, matching the
    same normalization as `get_field_from_spectrum_with_coordinate`:

        f(r0, t) = (Δkx·Δky·Δkz / (2π)^3) · Σ_k FE(k, t) · exp(i k·r0)

    with FE(k, t) evolved via the normal-variable identity used by
    `evolution_t`:

        EMk(k, t) = EMk0(k)·cos(ω·t) + signed_k_cross_partner(k)·sin(ω·t)

    Strategy: precompute `point_phase_all = exp(i·k·r)` for every point in
    one vmapped shot, then `lax.map` over t (chunked by `t_chunk_size` for
    vectorized throughput). Per t we build (Nkx·Nky·Nkz) cos/sin once and
    fuse them with `point_phase_all` and `EMk0` in a single einsum. This
    kills both nested-map redundancies (cos/sin no longer per-point,
    point_phase no longer per-t) at the cost of persisting one big
    tensor:  point_phase_all,  shape (N_point, Nkx, Nky, Nkz).

    Args:
        rts:        (N_point, 3), lab-frame coordinates, m. Column order (x,y,z).
        t_arr:      (N_t,) times, s.
        EMk0:       (2, 3, Nkx, Nky, Nkz) transversally-projected k-space
                    fields at t=0 (units V·m³, matching evolution_t's EMk0).
        signed_k_cross_partner: (2, 3, Nkx, Nky, Nkz), same convention as
                    evolution_t's arg — signs and swap already absorbed.
        omega:      (Nkx, Nky, Nkz), rad/s.
        kx, ky, kz: replicated 1-D k-axes.
        normalization: scalar (Δkx·Δky·Δkz / (2π)^3), 1/m^3.
        t_chunk_size: passed as `batch_size` to the outer t map. Larger =
                    more vectorization but more transient memory for the
                    per-chunk (chunk_t, Nkx, Nky, Nkz) cos/sin buffers.
                    None runs t sequentially.

    Returns:
        EMk_at_points_all_t: (2, 3, N_point, N_t), real (E, B*c).
    """
    # 1. Precompute point_phase for every point, once. Persistent tensor.
    def _one_point_phase(rt):
        k_dot_r = (
            kx[:, None, None] * rt[0]
            + ky[None, :, None] * rt[1]
            + kz[None, None, :] * rt[2]
        )   # (Nkx, Nky, Nkz)
        return jnp.exp(1j * k_dot_r)   # (Nkx, Nky, Nkz), complex
    point_phase_all = jax.vmap(_one_point_phase)(rts)   # (N_point, Nkx, Nky, Nkz), complex

    # 2. Per t: fused einsum against point_phase_all. cos/sin computed once
    #    per t (and, when t_chunk_size vmaps a chunk, once per chunk element).
    def _at_one_t(t):
        omega_t = omega * t                                              # (Nkx, Nky, Nkz)
        coswt = jnp.cos(omega_t)                                         # (Nkx, Nky, Nkz)
        sinwt = jnp.sin(omega_t)                                         # (Nkx, Nky, Nkz)
        term_cos = jnp.einsum(
            'cmxyz,pxyz,xyz->cmp', EMk0, point_phase_all, coswt,
        )                                                                # (2, 3, N_point), complex
        term_sin = jnp.einsum(
            'cmxyz,pxyz,xyz->cmp', signed_k_cross_partner, point_phase_all, sinwt,
        )                                                                # (2, 3, N_point), complex
        return normalization * (term_cos + term_sin)                     # (2, 3, N_point), complex

    t_map_kwargs = (
        {} if t_chunk_size is None else {'batch_size': int(t_chunk_size)}
    )
    EMk_over_t = jax.lax.map(_at_one_t, t_arr, **t_map_kwargs)          # (N_t, 2, 3, N_point), complex
    # Take real part: E and B*c are real; imag is numerical noise.
    return jnp.real(jnp.transpose(EMk_over_t, (1, 2, 3, 0)))            # (2, 3, N_point, N_t)


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
        pad=True,
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
        # check_divergence(Field=self.E0, x_coordinate=self.x_coordinate, y_coordinate=self.y_coordinate, z_coordinate=self.z_coordinate)
        # check_divergence(Field=self.B0, x_coordinate=self.x_coordinate, y_coordinate=self.y_coordinate, z_coordinate=self.z_coordinate)
        EM0 = jnp.stack([self.E0, self.B0 * C.speed_of_light])   #shape=(2, 3, Nx, Ny, Nz): [0]=E, [1]=B*c
        spectrum, k_coordinate_each_axis, pad_slices=get_spectrum_from_field_with_coordinate(
            field=EM0,   #shape=(2, 3, Nx, Ny, Nz)
            axis=[2,3,4],
            r_coordinate_each_axis=[self.x_coordinate, self.y_coordinate, self.z_coordinate],
            out_sharding=sharding_EM,
            pad=pad,
        )
        print_shard_layout(spectrum, "Initial spectrum from field")
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

        # Δk per axis (dk=2π for degenerate 1-point axes, so the factor
        # dk/(2π) is a no-op there — same convention as the FFT path).
        self.dkx = self.grid_k.dkx
        self.dky = self.grid_k.dky
        self.dkz = self.grid_k.dkz
        # Riemann-sum normalization for point-wise inverse:
        # f(r0, t) = (Δkx·Δky·Δkz / (2π)^3) · Σ_k FE(k, t) · exp(i k·(r0 - Δ))
        self._point_inverse_normalization = (
            self.dkx * self.dky * self.dkz / (2 * jnp.pi) ** 3
        )
        # r-offset per axis (physical position the FFT treats as its "r=0").
        # fftshift puts the DC bin at index N//2, and the FFT convention is
        # that the shifted-index N//2 corresponds to r_FFT=0. The physical
        # position at that index is `coord_pad[N//2]`, which is 0 only when
        # the padded coord happens to be exactly centered on 0. For every
        # other case the point-wise inverse must use r0_physical - Δ so the
        # returned value lines up with what `evolution()` (fftn+ifftn round
        # trip) returns on the grid.
        #
        # We can reconstruct Δ from the original coord + pad_slices without
        # storing the padded coord:  Δ = x[0] + (N_pad//2 - pad_before)·dr,
        # and for a degenerate axis (N=1) it is simply x[0].
        def _r_offset(coord, N_pad, pad_slice, d):
            if coord.size <= 1:
                return jnp.asarray(coord[0], dtype=jnp.float64)
            pad_before = pad_slice.start if pad_slice.start is not None else 0
            return coord[0] + (N_pad // 2 - pad_before) * d
        self._r_offset_x = _r_offset(self.x_coordinate, self.Nx_pad, self.pad_slices[0], self.grid_k.dx)
        self._r_offset_y = _r_offset(self.y_coordinate, self.Ny_pad, self.pad_slices[1], self.grid_k.dy)
        self._r_offset_z = _r_offset(self.z_coordinate, self.Nz_pad, self.pad_slices[2], self.grid_k.dz)

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
            Ek = EMk_evolution_in_window[0]                            #shape=(3, Nx_pad, Ny_pad, Nz_pad), V·m³
            Bk = EMk_evolution_in_window[1] / C.speed_of_light         #shape=(3, Nx_pad, Ny_pad, Nz_pad), T·m
            return {
                # stacked
                "Ek": Ek,
                "Bk": Bk,
                # component slices — 0-copy views on the stacked array
                "Ex_spectrum": Ek[0], "Ey_spectrum": Ek[1], "Ez_spectrum": Ek[2],
                "Bx_spectrum": Bk[0], "By_spectrum": Bk[1], "Bz_spectrum": Bk[2],
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
                # stacked
                "E": E_evolution_in_window,   #shape=(3, Nx, Ny, Nz), V/m
                "B": B_evolution_in_window,   #shape=(3, Nx, Ny, Nz), T
                # component slices — 0-copy views on the stacked array
                "Ex": E_evolution_in_window[0], "Ey": E_evolution_in_window[1], "Ez": E_evolution_in_window[2],
                "Bx": B_evolution_in_window[0], "By": B_evolution_in_window[1], "Bz": B_evolution_in_window[2],
                "x_coordinate": window_x_coordinate,   #shape=(Nx,), unit: m
                "y_coordinate": window_y_coordinate,   #shape=(Ny,), unit: m
                "z_coordinate": window_z_coordinate,   #shape=(Nz,), unit: m
            }

    def evolution_batch(
        self,
        t_coordinate,
        window_shift_velocity=jnp.array((0.0, 0.0, 0.0)),
        t_chunk_size=None,
    ):
        """Batched full-grid evolution: same math as `evolution()`, but for a
        whole array of times in one call.

        Internally `lax.map`s over `t_coordinate` with `batch_size = t_chunk_size`
        so multiple full-field IFFTs get vmapped (and XLA-fused) together —
        exactly what you'd otherwise write as a `for t in t_arr` Python loop
        of `evolution()` calls, but batched. `t_chunk_size` trades vectorized
        throughput for transient memory: each chunk allocates
        `t_chunk_size · (2·3·Nx_pad·Ny_pad·Nz_pad) · 16 B` for the complex
        spectrum plus the same for the real field. Set as large as fits.

        Args:
            t_coordinate: (N_t,) times, s.
            window_shift_velocity: (3,) m/s. Same meaning as in `evolution()`.
                A single velocity applies to every t — the returned coordinate
                arrays are the un-shifted `self.x/y/z_coordinate`; the physical
                position of grid index i at time t is `x[i] + vx·t` (and same
                for y, z).
            t_chunk_size: `batch_size` for the t-map. None runs t sequentially.

        Returns:
            EB_evolution_batch_dict = {
                "E":            (3, Nx, Ny, Nz, N_t), V/m,
                "B":            (3, Nx, Ny, Nz, N_t), T,
                "t_coordinate": (N_t,),
                "x_coordinate": (Nx,), m,
                "y_coordinate": (Ny,), m,
                "z_coordinate": (Nz,), m,
            }
        """
        t_arr = jnp.asarray(t_coordinate, dtype=jnp.float64).flatten()   # (N_t,)
        v = jnp.array(window_shift_velocity, dtype=jnp.float64).flatten()
        assert v.shape == (3,), f"window_shift_velocity must be length-3, got {v.shape}"

        # Refresh the k·v/c cache (same logic as evolution()).
        if (self._last_window_shift_velocity is None
                or not jnp.array_equal(v, self._last_window_shift_velocity)):
            self._k_dot_v_over_c = jnp.einsum(
                'lijk,l->ijk', self.k_hat, v / C.speed_of_light,
            )
            self._last_window_shift_velocity = v

        # Per-t worker: evolve in k-space, IFFT to real, unpad. Uses negative
        # axes so it's safe when lax.map's internal vmap adds a leading axis.
        def _at_one_t(t):
            EMk_t = evolution_t(
                omega_dot_t=self.omega * t,
                EMk0=self.EMk0,
                signed_k_cross_partner=self.signed_k_cross_partner,
                k_dot_v_over_c=self._k_dot_v_over_c,
            )   # (2, 3, Nx_pad, Ny_pad, Nz_pad), sharded on ('EM','channel')
            EM_field = get_field_from_spectrum_with_coordinate(
                spectrum=EMk_t,
                axis=(-3, -2, -1),
                k_coordinate_each_axis=[
                    self.kx_coordinate, self.ky_coordinate, self.kz_coordinate,
                ],
                pad_slices=self.pad_slices,
                real=True,
            )   # (2, 3, Nx, Ny, Nz), real
            return EM_field

        map_kwargs = {} if t_chunk_size is None else {'batch_size': int(t_chunk_size)}
        EM_batch = jax.lax.map(_at_one_t, t_arr, **map_kwargs)   # (N_t, 2, 3, Nx, Ny, Nz)
        # Move the t axis to the tail so the layout is consistent with
        # evolution_at_points (which returns (..., N_point, N_t)).
        EM_batch = jnp.moveaxis(EM_batch, 0, -1)                  # (2, 3, Nx, Ny, Nz, N_t)

        E_batch = EM_batch[0]                                     # (3, Nx, Ny, Nz, N_t), V/m
        B_batch = EM_batch[1] / C.speed_of_light                  # (3, Nx, Ny, Nz, N_t), T

        return {
            # stacked
            "E": E_batch,
            "B": B_batch,
            # component slices — 0-copy views
            "Ex": E_batch[0], "Ey": E_batch[1], "Ez": E_batch[2],
            "Bx": B_batch[0], "By": B_batch[1], "Bz": B_batch[2],
            "t_coordinate": t_arr,
            "x_coordinate": self.x_coordinate,
            "y_coordinate": self.y_coordinate,
            "z_coordinate": self.z_coordinate,
        }

    def evolution_at_points(
        self,
        t_coordinate,
        x_point=None, y_point=None, z_point=None,
        t_chunk_size=None,
        point_chunk_size=None,
    ):
        """Evaluate E(t) and B(t) at fixed lab-frame points over an array of times.

        Compared to `evolution()`: the integrand is the same continuous
        inverse Fourier kernel

            f(r0, t) = (1/(2π)^n) · ∫ FE(k, t) · exp(i k·r0) dk

        but instead of asking for the field on the full padded grid — where
        the FFT butterfly wins at O(N log N) total — we ask for it at a
        user-chosen set of points and evaluate the Riemann sum directly, at
        O(N) per point. That trade gives up FFT's log-N speedup in exchange
        for two things: (1) **point-selection freedom** — we compute nothing
        for the grid points we don't care about; (2) **sub-grid resolution**
        — r0 is a real number, not a grid index, so the returned value is
        the exact bandlimited-sinc-interpolated field. No window shift: r0
        is fixed in the lab frame and the pulse drifts past it naturally.

        Implementation: precompute `point_phase = exp(i·k·r0)` for every
        point once, then `lax.map` over t with a fused einsum. Neither
        cos(ω·t) (which only depends on t) nor point_phase (which only
        depends on r0) is recomputed. Memory cost is a persistent tensor
        of shape (N_point, Nkx, Nky, Nkz) — roughly N_point·Nk·16 B — so
        for large N_point × Nk (e.g. sampling a fine grid off a padded
        3-D k-grid) pass `point_chunk_size` to split the point axis into
        chunks that each fit; results are concatenated on return.

        Args:
            t_coordinate: (N_t,) times, s.
            x_point, y_point, z_point: lab-frame point coordinates, m.
                Each may be None (treated as 0), a scalar, or a 1-D array;
                non-scalar entries must share the same length N_point.
            t_chunk_size: optional int; if set, the t-map runs with this
                `batch_size` — larger = more vectorized throughput but
                more transient memory for the per-chunk cos/sin buffers.
                None (default) processes t sequentially.
            point_chunk_size: optional int; if set, the point axis is
                processed in Python-level chunks of this size and the
                per-chunk results are concatenated. Use when
                `N_point · Nkx · Nky · Nkz · 16 B` doesn't fit in memory.
                None (default) computes all points at once.

        Returns:
            EB_evolution_dict = {
                "E":            (3, N_point, N_t), V/m,
                "B":            (3, N_point, N_t), T,
                "t_coordinate": (N_t,),
                "x_coordinate": (N_point,), m,
                "y_coordinate": (N_point,), m,
                "z_coordinate": (N_point,), m,
            }
        """
        # 1. Normalize point coordinates to (N_point,) arrays.
        def _to_1d(v):
            if v is None:
                return None
            return jnp.asarray(v, dtype=jnp.float64).flatten()
        xs = _to_1d(x_point)
        ys = _to_1d(y_point)
        zs = _to_1d(z_point)
        sizes = [a.size for a in (xs, ys, zs) if a is not None and a.size > 1]
        N_point = max(sizes) if sizes else max(
            (a.size for a in (xs, ys, zs) if a is not None), default=1,
        )
        def _broadcast(arr):
            if arr is None:
                return jnp.zeros((N_point,), dtype=jnp.float64)
            if arr.size == 1:
                return jnp.broadcast_to(arr, (N_point,))
            assert arr.size == N_point, (
                f"point-coordinate arrays must share a length; got {arr.size}, expected {N_point}"
            )
            return arr
        xs = _broadcast(xs)
        ys = _broadcast(ys)
        zs = _broadcast(zs)

        # 2. Time axis.
        t_arr = jnp.asarray(t_coordinate, dtype=jnp.float64).flatten()   # (N_t,)

        # 3. Shift the physical points by the FFT r-offset so exp(i·k·r_shifted)
        # in the Riemann sum yields the field at the physical r0. Then call
        # the core kernel — it does the (t outer, points inner) double lax.map
        # so ω·t / cos·t / sin·t are computed once per t, not once per (point,t).
        rts = jnp.stack([
            xs - self._r_offset_x,
            ys - self._r_offset_y,
            zs - self._r_offset_z,
        ], axis=-1)   # (N_point, 3)

        _t_chunk = None if t_chunk_size is None else int(t_chunk_size)
        if point_chunk_size is None or point_chunk_size >= N_point:
            EMk_at_points = _evolution_at_points_all_t(
                rts,
                t_arr,
                self.EMk0, self.signed_k_cross_partner,
                self.omega,
                self.kx_coordinate, self.ky_coordinate, self.kz_coordinate,
                self._point_inverse_normalization,
                _t_chunk,
            )   # (2, 3, N_point, N_t)
        else:
            # Split points in Python and concatenate; keeps the persistent
            # point_phase_all tensor to size (point_chunk_size, Nkx, Nky, Nkz).
            _p_chunk = int(point_chunk_size)
            chunks = []
            for start in range(0, N_point, _p_chunk):
                end = min(start + _p_chunk, N_point)
                chunks.append(
                    _evolution_at_points_all_t(
                        rts[start:end],
                        t_arr,
                        self.EMk0, self.signed_k_cross_partner,
                        self.omega,
                        self.kx_coordinate, self.ky_coordinate, self.kz_coordinate,
                        self._point_inverse_normalization,
                        _t_chunk,
                    )   # (2, 3, chunk, N_t)
                )
            EMk_at_points = jnp.concatenate(chunks, axis=2)   # (2, 3, N_point, N_t)

        E_at_points = EMk_at_points[0]                          # (3, N_point, N_t), V/m
        B_at_points = EMk_at_points[1] / C.speed_of_light       # (3, N_point, N_t), T

        return {
            # stacked
            "E": E_at_points,
            "B": B_at_points,
            # component slices — 0-copy views
            "Ex": E_at_points[0], "Ey": E_at_points[1], "Ez": E_at_points[2],
            "Bx": B_at_points[0], "By": B_at_points[1], "Bz": B_at_points[2],
            "t_coordinate": t_arr,
            "x_coordinate": xs,
            "y_coordinate": ys,
            "z_coordinate": zs,
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
        pad=True,
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
            pad=pad,
            )
    def evolution(self, evolution_time=0.0, window_shift_velocity=0.0, return_spectrum=False):
        """
        Args
        ----
        evolution_time : float, s
        window_shift_velocity : float, m/s (vx only for the 1-D solver)
        return_spectrum : bool
            False → return real fields (default, existing behavior).
            True  → skip the IFFT and return the k-space spectrum together
                    with `kx_coordinate` and `pad_slices` so the caller can
                    plug the result straight into e.g.
                    `spectrum.get_envelope_from_spectrum_with_coordinate` or
                    `spectrum.get_field_from_spectrum_with_coordinate`.
        """
        window_shift_velocity_3d = jnp.pad(
            jnp.array(window_shift_velocity, dtype=jnp.float64).flatten(),
            pad_width=((0, 2),),
        )   # shape=(3,)
        result = self.Solver.evolution(
            evolution_time=evolution_time,
            window_shift_velocity=window_shift_velocity_3d,
            return_spectrum=return_spectrum,
        )
        if return_spectrum:
            Ek = result['Ek'][:, :, 0, 0]   # (3, Nx_pad) — strip y, z degenerate axes
            Bk = result['Bk'][:, :, 0, 0]
            window_x_coordinate = self.x_coordinate + window_shift_velocity_3d[0] * evolution_time
            return {
                # stacked
                'Ek': Ek,
                'Bk': Bk,
                # component slices
                'Ex_spectrum': Ek[0], 'Ey_spectrum': Ek[1], 'Ez_spectrum': Ek[2],
                'Bx_spectrum': Bk[0], 'By_spectrum': Bk[1], 'Bz_spectrum': Bk[2],
                'kx_coordinate': result['kx_coordinate'],
                'pad_slices':    result['pad_slices'][:1],   # drop y, z pad-slices
                'x_coordinate':  window_x_coordinate,
            }
        E_evolution_in_window = result["E"][:, :, 0, 0]   # (3, Nx)
        B_evolution_in_window = result["B"][:, :, 0, 0]
        window_x_coordinate   = result["x_coordinate"]    # (Nx,)
        return {
            # stacked
            'E': E_evolution_in_window,
            'B': B_evolution_in_window,
            # component slices
            'Ex': E_evolution_in_window[0], 'Ey': E_evolution_in_window[1], 'Ez': E_evolution_in_window[2],
            'Bx': B_evolution_in_window[0], 'By': B_evolution_in_window[1], 'Bz': B_evolution_in_window[2],
            'x_coordinate': window_x_coordinate,
        }

    def evolution_batch(
        self,
        t_coordinate,
        window_shift_velocity=0.0,
        t_chunk_size=None,
    ):
        """See `Spectral_Maxwell_Solver.evolution_batch`.

        1-D thin wrapper: y and z axes are degenerate.
        """
        v3 = jnp.pad(
            jnp.array(window_shift_velocity, dtype=jnp.float64).flatten(),
            pad_width=((0, 2),),
        )   # (3,)
        result = self.Solver.evolution_batch(
            t_coordinate=t_coordinate,
            window_shift_velocity=v3,
            t_chunk_size=t_chunk_size,
        )
        E = result["E"][:, :, 0, 0, :]     # (3, Nx, N_t)
        B = result["B"][:, :, 0, 0, :]
        return {
            # stacked
            "E": E, "B": B,
            # component slices
            "Ex": E[0], "Ey": E[1], "Ez": E[2],
            "Bx": B[0], "By": B[1], "Bz": B[2],
            "t_coordinate": result["t_coordinate"],
            "x_coordinate": result["x_coordinate"],
        }

    def evolution_at_points(
        self,
        t_coordinate,
        x_point=None,
        t_chunk_size=None,
        point_chunk_size=None,
    ):
        """See `Spectral_Maxwell_Solver.evolution_at_points`.

        1-D thin wrapper: y and z axes are degenerate.
        """
        result = self.Solver.evolution_at_points(
            t_coordinate=t_coordinate,
            x_point=x_point, y_point=None, z_point=None,
            t_chunk_size=t_chunk_size,
            point_chunk_size=point_chunk_size,
        )
        E = result["E"]   # (3, N_point, N_t)
        B = result["B"]
        return {
            # stacked
            "E": E, "B": B,
            # component slices
            "Ex": E[0], "Ey": E[1], "Ez": E[2],
            "Bx": B[0], "By": B[1], "Bz": B[2],
            "t_coordinate": result["t_coordinate"],  # (N_t,)
            "x_coordinate": result["x_coordinate"],  # (N_point,)
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
        pad=True,
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
            pad=pad,
            )
    def evolution(self, evolution_time=0.0, window_shift_velocity=(0.0, 0.0), return_spectrum=False):
        """
        Args
        ----
        evolution_time : float, s
        window_shift_velocity : tuple (vx, vy), m/s
        return_spectrum : bool
            False → return real fields (default).
            True  → skip the IFFT and return (Ex_spectrum, Ey_spectrum, …,
                    kx_coordinate, ky_coordinate, pad_slices) so the caller
                    can plug the result straight into
                    `spectrum.get_envelope_from_spectrum_with_coordinate` or
                    `spectrum.get_field_from_spectrum_with_coordinate`.
        """
        window_shift_velocity_3d = jnp.pad(
            jnp.array(window_shift_velocity, dtype=jnp.float64).flatten(),
            pad_width=((0, 1),),
        )   # shape=(3,)
        result = self.Solver.evolution(
            evolution_time=evolution_time,
            window_shift_velocity=window_shift_velocity_3d,
            return_spectrum=return_spectrum,
        )
        if return_spectrum:
            Ek = result['Ek'][:, :, :, 0]   # (3, Nx_pad, Ny_pad) — strip z
            Bk = result['Bk'][:, :, :, 0]
            window_x_coordinate = self.x_coordinate + window_shift_velocity_3d[0] * evolution_time
            window_y_coordinate = self.y_coordinate + window_shift_velocity_3d[1] * evolution_time
            return {
                # stacked
                'Ek': Ek,
                'Bk': Bk,
                # component slices
                'Ex_spectrum': Ek[0], 'Ey_spectrum': Ek[1], 'Ez_spectrum': Ek[2],
                'Bx_spectrum': Bk[0], 'By_spectrum': Bk[1], 'Bz_spectrum': Bk[2],
                'kx_coordinate': result['kx_coordinate'],
                'ky_coordinate': result['ky_coordinate'],
                'pad_slices':    result['pad_slices'][:2],   # drop z pad-slice
                'x_coordinate':  window_x_coordinate,
                'y_coordinate':  window_y_coordinate,
            }
        E_evolution_in_window = result["E"][:, :, :, 0]   # (3, Nx, Ny)
        B_evolution_in_window = result["B"][:, :, :, 0]
        window_x_coordinate   = result["x_coordinate"]    # (Nx,)
        window_y_coordinate   = result["y_coordinate"]    # (Ny,)
        return {
            # stacked
            'E': E_evolution_in_window,
            'B': B_evolution_in_window,
            # component slices
            'Ex': E_evolution_in_window[0], 'Ey': E_evolution_in_window[1], 'Ez': E_evolution_in_window[2],
            'Bx': B_evolution_in_window[0], 'By': B_evolution_in_window[1], 'Bz': B_evolution_in_window[2],
            'x_coordinate': window_x_coordinate,
            'y_coordinate': window_y_coordinate,
        }

    def evolution_batch(
        self,
        t_coordinate,
        window_shift_velocity=(0.0, 0.0),
        t_chunk_size=None,
    ):
        """See `Spectral_Maxwell_Solver.evolution_batch`.

        2-D thin wrapper: the z-axis is degenerate and dropped from the
        returned dict.
        """
        v3 = jnp.pad(
            jnp.array(window_shift_velocity, dtype=jnp.float64).flatten(),
            pad_width=((0, 1),),
        )   # (3,)
        result = self.Solver.evolution_batch(
            t_coordinate=t_coordinate,
            window_shift_velocity=v3,
            t_chunk_size=t_chunk_size,
        )
        E = result["E"][:, :, :, 0, :]     # (3, Nx, Ny, N_t)
        B = result["B"][:, :, :, 0, :]
        return {
            # stacked
            "E": E, "B": B,
            # component slices
            "Ex": E[0], "Ey": E[1], "Ez": E[2],
            "Bx": B[0], "By": B[1], "Bz": B[2],
            "t_coordinate": result["t_coordinate"],
            "x_coordinate": result["x_coordinate"],
            "y_coordinate": result["y_coordinate"],
        }

    def evolution_at_points(
        self,
        t_coordinate,
        x_point=None, y_point=None,
        t_chunk_size=None,
        point_chunk_size=None,
    ):
        """See `Spectral_Maxwell_Solver.evolution_at_points`.

        2-D thin wrapper: the z-axis is degenerate so `z_point` is fixed at
        0 and dropped from the returned dict.
        """
        result = self.Solver.evolution_at_points(
            t_coordinate=t_coordinate,
            x_point=x_point, y_point=y_point, z_point=None,
            t_chunk_size=t_chunk_size,
            point_chunk_size=point_chunk_size,
        )
        E = result["E"]   # (3, N_point, N_t)
        B = result["B"]
        return {
            # stacked
            "E": E, "B": B,
            # component slices
            "Ex": E[0], "Ey": E[1], "Ez": E[2],
            "Bx": B[0], "By": B[1], "Bz": B[2],
            "t_coordinate": result["t_coordinate"],  # (N_t,)
            "x_coordinate": result["x_coordinate"],  # (N_point,)
            "y_coordinate": result["y_coordinate"],  # (N_point,)
        }