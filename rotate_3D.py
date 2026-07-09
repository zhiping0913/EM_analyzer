"""
Rigid rotation of a 3-D scalar / vector field between two Cartesian frames.

Rotation setup
--------------
Consider a source frame `x0-y0-z0` with orthonormal unit vectors
`e_{x0}, e_{y0}, e_{z0}`. A vector field in this frame reads

    A = A_{x0} e_{x0} + A_{y0} e_{y0} + A_{z0} e_{z0}.

We want the same field expressed in a rotated frame `x1-y1-z1`. The rotated
unit vectors are defined by three Euler angles `(phi, psi, theta)`:

    e_{x1} = ( cos φ cos ψ cos θ − sin φ sin ψ) e_{x0}
           + ( cos φ sin ψ cos θ + sin φ cos ψ) e_{y0}
           −   cos φ sin θ                     e_{z0}

    e_{y1} = (−sin φ cos ψ cos θ − cos φ sin ψ) e_{x0}
           + (−sin φ sin ψ cos θ + cos φ cos ψ) e_{y0}
           +   sin φ sin θ                     e_{z0}

    e_{z1} =   cos ψ sin θ e_{x0}
           +   sin ψ sin θ e_{y0}
           +   cos θ       e_{z0}

The 3 × 3 rotation matrix built by `generate_rotation_matrix(phi, psi, theta)`
has these unit vectors as its rows, so component transformation is

    [A_{x1}]       [A_{x0}]
    [A_{y1}] = R · [A_{y0}]
    [A_{z1}]       [A_{z0}]

and A = A_{x0} e_{x0} + A_{y0} e_{y0} + A_{z0} e_{z0}
     = A_{x1} e_{x1} + A_{y1} e_{y1} + A_{z1} e_{z1}.

Physical meaning of (phi, psi, theta)
-------------------------------------
The three angles come from the standard optics geometry of a plane wave
striking a flat surface. Place the surface in the `z0 = 0` plane; its normal
is `e_n = e_{z0}`. Shine a monochromatic wave whose k-vector direction is
`e_k`. Choose the rotated frame so that `e_k = e_{z1}`. Then:

* **θ — incidence angle.** By construction ⟨e_n, e_k⟩ = ⟨e_{z0}, e_{z1}⟩ = θ.

* **ψ — orientation of the plane of incidence.** The s-polarization direction is
        e_s = (e_n × e_k) / |e_n × e_k| = −sin ψ · e_{x0} + cos ψ · e_{y0},
    so ⟨e_{y0}, e_s⟩ = ψ. Rotating ψ swings the plane of incidence about the
    surface normal.

* **φ — polarization angle in the plane of incidence.** The p-polarization
    direction is
        e_p = e_s × e_k = cos ψ cos θ · e_{x0} + sin ψ cos θ · e_{y0} − sin θ · e_{z0}.
    If the wave's electric field points along `e_{x1}`, then
        ⟨e_{x1}, e_p⟩ = arccos(e_{x1} · e_p) = arccos(cos φ) = φ.
    So φ = 0 is pure p-polarization, φ = π/2 is pure s-polarization.

Public API
----------
* `generate_rotation_matrix(phi, psi, theta)` → 3 × 3 rotation matrix `R`.
* `rotate(A0, R, x0_coordinate, y0_coordinate, z0_coordinate,
           x1_coordinate, y1_coordinate, z1_coordinate, type='vector'|'scalar')`
  — resamples `A0` from the source grid onto the rotated grid via
  `jax.scipy.ndimage.map_coordinates` (per-channel sharded when 3 devices are
  available). When `type='vector'`, additionally applies `R` to rotate the
  vector components; when `type='scalar'`, only the spatial remap is done.
* `Rotation(phi, psi, theta)` — convenience class wrapping the matrix and the
  rotate() call.
"""
import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336')
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/EM_analyzer')
from line_profiler import profile
from EM_analyzer.device_config import configure_jax_backend, get_channel_sharding
configure_jax_backend()                 # side-effect: pin backend + device count

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Union
from jax.scipy.ndimage import map_coordinates
from Spectral_Maxwell.kgrid import make_k_coordinate_from_r_coordinate
from EM_analyzer.pretreat_fields import square_integral_field
from EM_analyzer.fft_backend import fftn, ifftn

# Channel sharding for (3, Nx, Ny, Nz) fields — shared through
# device_config so rotate_3D and coordinate_transformation don't build
# competing meshes on the same physical devices. Falls back to a
# 1-device replicated mesh when the backend can't do a clean 3-way split
# (e.g. 2-GPU node).
sharding = get_channel_sharding(3)
mesh     = sharding.mesh

@jax.jit
def generate_rotation_matrix(phi, psi, theta):
    cph, sph = jnp.cos(phi), jnp.sin(phi)
    cps, sps = jnp.cos(psi), jnp.sin(psi)
    cth, sth = jnp.cos(theta), jnp.sin(theta)
    R=jnp.array([
        [cph*cps*cth - sph*sps,  cph*sps*cth + sph*cps, -cph*sth],
        [-sph*cps*cth - cph*sps, -sph*sps*cth + cph*cps,  sph*sth],
        [cps*sth,               sps*sth,                cth]
    ])   #shape: (3, 3)
    return R

@jax.jit
def get_dr(x_coordinate:jnp.ndarray, y_coordinate:jnp.ndarray, z_coordinate:jnp.ndarray):
    dx = x_coordinate[1] - x_coordinate[0] if x_coordinate.size > 1 else 1.0
    dy = y_coordinate[1] - y_coordinate[0] if y_coordinate.size > 1 else 1.0
    dz = z_coordinate[1] - z_coordinate[0] if z_coordinate.size > 1 else 1.0
    return jnp.array([dx, dy, dz])
@jax.jit
def get_grid_id(R:jnp.ndarray,x0_coordinate:jnp.ndarray, y0_coordinate:jnp.ndarray, z0_coordinate:jnp.ndarray, x1_coordinate:jnp.ndarray, y1_coordinate:jnp.ndarray, z1_coordinate:jnp.ndarray):
    Nx0 = x0_coordinate.size
    Ny0 = y0_coordinate.size
    Nz0 = z0_coordinate.size
    starts0 = jnp.array([x0_coordinate[0], y0_coordinate[0], z0_coordinate[0]])
    dr0 = get_dr(x0_coordinate, y0_coordinate, z0_coordinate)
    max_idx = jnp.array([Nx0 - 1, Ny0 - 1, Nz0 - 1])
   
    with mesh:
        # 2. 分片生成 grid1 (3, Nx1, Ny1, Nz1)
        grid1 = jax.lax.with_sharding_constraint(
            jnp.stack(jnp.meshgrid(x1_coordinate, y1_coordinate, z1_coordinate, indexing="ij"), axis=0), 
            sharding,
            )

        # 3. 坐标逆变换 r0 = R^T @ grid1
        # 由于 R 是 (3,3)，grid1 是分片的，einsum 会自动处理跨设备通信
        r0 = jnp.einsum('ml,mijk->lijk', R, grid1)
        r0 = jax.lax.with_sharding_constraint(r0, sharding)

        # 4. 计算索引并 Clip
        # 这里的计算是逐元素并行化的
        idx = (r0 - starts0[:, None, None, None]) / dr0[:, None, None, None]
        coords_idx = jnp.clip(idx, 0.0, max_idx[:, None, None, None])
    return coords_idx

@jax.jit
def _rotate_vector(A0:jnp.ndarray, R:jnp.ndarray,x0_coordinate:jnp.ndarray, y0_coordinate:jnp.ndarray, z0_coordinate:jnp.ndarray, x1_coordinate:jnp.ndarray, y1_coordinate:jnp.ndarray, z1_coordinate:jnp.ndarray):
    dr0 = get_dr(x0_coordinate, y0_coordinate, z0_coordinate)
    dr1 = get_dr(x1_coordinate, y1_coordinate, z1_coordinate)
    with mesh:
        coords_idx = get_grid_id(R, x0_coordinate, y0_coordinate, z0_coordinate, x1_coordinate, y1_coordinate, z1_coordinate)
        # 5. 分片插值
        A0 = jax.lax.with_sharding_constraint(A0, sharding)
        # vmap 处理 3 个通道
        A1_raw = jax.jit(jax.vmap(lambda channel: map_coordinates(channel, coords_idx, order=1, mode='nearest', cval=0.0),in_axes=0,out_axes=0))(A0)
        A1_raw = jax.lax.with_sharding_constraint(A1_raw, sharding)
        # 6. 矢量旋转变换
        A1 = jnp.einsum('ij,jklm->iklm', R, A1_raw)
        A1 = jax.lax.with_sharding_constraint(A1, sharding)
    jax.debug.print(f'Input field shape: {A0.shape}, Output field shape: {A1.shape}')
    I0=square_integral_field(Field=A0, dr=dr0)
    jax.debug.print('Input field integral: {I0}', I0=I0)
    I1=square_integral_field(Field=A1, dr=dr1)
    jax.debug.print('Output field integral: {I1}', I1=I1)
    jax.debug.print('Integral ratio I1/I0: {ratio}', ratio=I1/I0)
    return A1

def _rotate_scalar(A0:jnp.ndarray, R:jnp.ndarray,x0_coordinate:jnp.ndarray, y0_coordinate:jnp.ndarray, z0_coordinate:jnp.ndarray, x1_coordinate:jnp.ndarray, y1_coordinate:jnp.ndarray, z1_coordinate:jnp.ndarray):
    dr0 = get_dr(x0_coordinate, y0_coordinate, z0_coordinate)
    dr1 = get_dr(x1_coordinate, y1_coordinate, z1_coordinate)
    coords_idx = get_grid_id(R, x0_coordinate, y0_coordinate, z0_coordinate, x1_coordinate, y1_coordinate, z1_coordinate)
    A1 = map_coordinates(A0, coords_idx, order=1, mode='nearest', cval=0.0)
    jax.debug.print(f'Input field shape: {A0.shape}, Output field shape: {A1.shape}')
    I0=square_integral_field(Field=A0, dr=dr0)
    jax.debug.print('Input field integral: {I0}', I0=I0)
    I1=square_integral_field(Field=A1, dr=dr1)
    jax.debug.print('Output field integral: {I1}', I1=I1)
    jax.debug.print('Integral ratio I1/I0: {ratio}', ratio=I1/I0)
    return A1



def rotate(
    A0:jnp.ndarray, 
    R,
    x0_coordinate=[0], y0_coordinate=[0], z0_coordinate=[0], 
    x1_coordinate:Optional[jnp.ndarray]=None, y1_coordinate:Optional[jnp.ndarray]=None, z1_coordinate:Optional[jnp.ndarray]=None,
    type='vector'
    ):
    A0=jnp.asarray(A0)
    R=jnp.asarray(R)
    x0_coordinate=jnp.asarray(x0_coordinate).flatten()
    y0_coordinate=jnp.asarray(y0_coordinate).flatten()
    z0_coordinate=jnp.asarray(z0_coordinate).flatten()
    if x1_coordinate is None:
        x1_coordinate = x0_coordinate
    else:
        x1_coordinate = jnp.asarray(x1_coordinate).flatten()
    if y1_coordinate is None:
        y1_coordinate = y0_coordinate
    else:
        y1_coordinate = jnp.asarray(y1_coordinate).flatten()
    if z1_coordinate is None:
        z1_coordinate = z0_coordinate
    else: 
        z1_coordinate = jnp.asarray(z1_coordinate).flatten()
    assert R.shape==(3, 3), "R shape must be (3, 3)"
    assert type in ['vector','scalar'], "type must be 'vector' or 'scalar'"
    if type=='scalar':
        assert A0.shape==(x0_coordinate.size, y0_coordinate.size, z0_coordinate.size), "A0 shape must be (Nx0, Ny0, Nz0) for scalar field"
        A1 = _rotate_scalar(A0, R, x0_coordinate, y0_coordinate, z0_coordinate, x1_coordinate, y1_coordinate, z1_coordinate)
    else:
        assert A0.shape==(3, x0_coordinate.size, y0_coordinate.size, z0_coordinate.size), "A0 shape must be (3, Nx0, Ny0, Nz0) for vector field"
        A1 =_rotate_vector(A0, R, x0_coordinate, y0_coordinate, z0_coordinate, x1_coordinate, y1_coordinate, z1_coordinate)
    return A1





class Rotation:
    def __init__(self, phi=0.0, psi=0.0, theta=0.0):
        self.phi = phi
        self.psi = psi
        self.theta = theta
        self.R = generate_rotation_matrix(phi, psi, theta)
        self.RT = self.R.T
        print(f'Rotation with angles (phi, psi, theta)=({phi}, {psi}, {theta}) radians initialized.', flush=True)

    #@profile
    def rotate_r_space(self, A, x0_coordinate=[0], y0_coordinate=[0], z0_coordinate=[0], x1_coordinate=None, y1_coordinate=None, z1_coordinate=None,direction='0->1',type='vector'):
        assert direction in ['0->1','1->0'], "direction must be '0->1' or '1->0'"
        if direction=='1->0':
            A0=rotate(A0=A, R=self.RT, x0_coordinate=x1_coordinate, y0_coordinate=y1_coordinate, z0_coordinate=z1_coordinate, x1_coordinate=x0_coordinate, y1_coordinate=y0_coordinate, z1_coordinate=z0_coordinate,type=type)
            return A0
        else:
            A1=rotate(A0=A, R=self.R, x0_coordinate=x0_coordinate, y0_coordinate=y0_coordinate, z0_coordinate=z0_coordinate, x1_coordinate=x1_coordinate, y1_coordinate=y1_coordinate, z1_coordinate=z1_coordinate,type=type)
            return A1
    #@profile
    def rotate_k_space(self, A, x_coordinate=[0], y_coordinate=[0], z_coordinate=[0],direction='0->1',type='vector'):
        assert direction in ['0->1','1->0'], "direction must be '0->1' or '1->0'"
        kx_coordinate, dkx, dx = make_k_coordinate_from_r_coordinate(x_coordinate)
        ky_coordinate, dky, dy = make_k_coordinate_from_r_coordinate(y_coordinate)
        kz_coordinate, dkz, dz = make_k_coordinate_from_r_coordinate(z_coordinate)
        Ak = fftn(A, axes=(-3,-2,-1))   #shape: (3, Nx, Ny, Nz)
        if direction=='1->0':
            A0k=rotate(A0=Ak, R=self.RT, x0_coordinate=kx_coordinate, y0_coordinate=ky_coordinate, z0_coordinate=kz_coordinate, x1_coordinate=kx_coordinate, y1_coordinate=ky_coordinate, z1_coordinate=kz_coordinate,type=type)
            return ifftn(A0k, axes=(-3,-2,-1)).real
        else:
            A1k=rotate(A0=Ak, R=self.R, x0_coordinate=kx_coordinate, y0_coordinate=ky_coordinate, z0_coordinate=kz_coordinate, x1_coordinate=kx_coordinate, y1_coordinate=ky_coordinate, z1_coordinate=kz_coordinate,type=type)
            return ifftn(A1k, axes=(-3,-2,-1)).real
    #@profile
    def rotate(self, A, x0_coordinate, y0_coordinate, z0_coordinate, x1_coordinate=[0], y1_coordinate=[0], z1_coordinate=[0],direction='0->1',space='r',type='vector'):
        """
        Rotate field using selected space.
        space="r":
            Rotate in real space,
            requires (x0,y0,z0) and (x1,y1,z1)
        space="k":
            Rotate in k space,
            requires (x0,y0,z0) only
        """
        assert direction in ['0->1','1->0'], "direction must be '0->1' or '1->0'"
        assert space in ['r','k'], "space must be 'r' or 'k'"
        assert type in ['vector','scalar'], "type must be 'vector' or 'scalar'"
        if space == "r":
             return self.rotate_r_space(A, x0_coordinate, y0_coordinate, z0_coordinate, x1_coordinate, y1_coordinate, z1_coordinate,direction=direction,type=type)
        elif space == "k":
            return self.rotate_k_space(A, x0_coordinate, y0_coordinate, z0_coordinate,direction=direction,type=type)


