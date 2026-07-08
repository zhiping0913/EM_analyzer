import sys
from line_profiler import profile
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/EM_analyzer')

from EM_analyzer.device_config import configure_jax_backend
_backend_info = configure_jax_backend()

import jax
from jax.sharding import PartitionSpec, NamedSharding, Mesh
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple, Union
from jax.scipy.ndimage import map_coordinates
from Spectral_Maxwell.kgrid import make_k_coordinate_from_r_coordinate
from EM_analyzer.pretreat_fields import square_integral_field
from EM_analyzer.fft_backend import fftn, ifftn

# Fields are shape (3, Nx, Ny, Nz) — first axis has 3 components (x, y, z).
# Sharding strategy on the 'channel' axis:
#   - If we have exactly 3 devices → one component per device (ideal).
#   - Otherwise fall back to a 1-device replicated mesh: the code still runs
#     correctly, just without the 3-way parallel speedup. This keeps
#     rotate_3D compatible with the 2-GPU or 6-CPU backends chosen by
#     device_config (6 CPUs — first 3 used here; 2 GPUs — replicated, since
#     3 doesn't cleanly divide 2).
_channel_devices = jax.local_devices()[:3] if _backend_info['LOCAL_DEVICE_COUNT'] >= 3 else jax.local_devices()[:1]
mesh = Mesh(np.array(_channel_devices), ('channel',))
if len(_channel_devices) == 3:
    sharding = NamedSharding(mesh, PartitionSpec('channel', None, None, None))
else:
    # Not enough devices for a clean 3-way split → replicate.
    sharding = NamedSharding(mesh, PartitionSpec(None, None, None, None))
print(f"[rotate_3D] channel sharding uses {len(_channel_devices)} device(s)", flush=True)
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


