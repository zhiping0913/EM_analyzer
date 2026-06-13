from line_profiler import profile
import jax
from functools import partial
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import jit, vmap
import scipy.constants as C
from Spectral_Maxwell.kgrid import make_k_coordinate_from_r_coordinate,make_r_coordinate_from_k_coordinate
from EM_analyzer.fft_backend import fftn, ifftn
from EM_analyzer.pretreat_fields import check_divergence
@profile
@partial(jit, static_argnames=('k0'))
def build_dispersion(kx_coordinate, ky_coordinate, k0=1.0):
    """_summary_

    Args:
        kx_coordinate (_type_): shape: (Nx)
        ky_coordinate (_type_): shape: (Ny,)
        k0 (float, optional): _description_. Defaults to 1.0.

    Returns:
        kr: shape: (Nx, Ny)
        kz: shape: (Nx, Ny)
        k: shape: (3, Nx, Ny)
        k_hat: shape: (3, Nx, Ny)
        project_operator: shape: (3,3,Nx,Ny)
    """
    kx,ky=jnp.meshgrid(kx_coordinate, ky_coordinate, indexing="ij")   #shape: (Nx, Ny)
    kr = jnp.hypot(kx, ky)   #transverse wavevector, shape: (Nx, Ny)
    k_mask=kr <= k0   #shape: (Nx, Ny)
    kz = jnp.where(
        k_mask,
        jnp.sqrt(k0**2 - kr**2),
        1j * jnp.sqrt(kr**2 - k0**2)
    )   #longitudinal wavevector, shape: (Nx, Ny)

    # full wavevector (for B field)
    k = jnp.stack([kx, ky, kz], axis=0)   #shape: (3, Nx, Ny), (kx,ky,kz), |k|=k0, unit: m^-1
    k_hat=k/k0   #shape: (3, Nx, Ny), k_hat=k/k0, unit vector
    I = jnp.eye(3,dtype=jnp.float64)   #shape: (3,3)
    project_operator=I[:,:,jnp.newaxis,jnp.newaxis]-k_hat[jnp.newaxis,:,:,:]*k_hat[:,jnp.newaxis,:,:]   #shape: (3,3,Nx,Ny), I - k_i*k_j/k0^2
    return kr, kz,k_mask, k, k_hat, project_operator
@profile
@jit
def propagate_z(z: jnp.ndarray, Ek0_proj: jnp.ndarray, kz: jnp.ndarray,k_hat: jnp.ndarray, k_mask:jnp.ndarray):
    """
    Propagate angular spectrum by distance z.

    Args:
        Ek0_proj: shape (3, Nx, Ny), projected angular spectrum at z=0
        kz: shape (Nx, Ny), longitudinal wavevector
        k_hat: shape (3, Nx, Ny), unit wavevector
        k_mask: shape (Nx, Ny), mask for propagating waves
        z: propagation distance

    Returns:
        Ekz_proj: shape (3, Nx, Ny), projected angular spectrum at z
    """
    propagator = jnp.where(
        k_mask,
        jnp.exp(1j * kz * z),   #propagating wave
        jnp.exp(-jnp.abs(kz) * jnp.abs(z))  # evanescent wave
    )
    Ek_propagate_z = Ek0_proj * propagator[jnp.newaxis, :, :]   #shape: (3, Nx, Ny)
    Bk_propagate_z = jnp.cross(k_hat, Ek_propagate_z, axis=0) / C.speed_of_light   #shape: (3, Nx, Ny)
    return Ek_propagate_z, Bk_propagate_z


class Vector_angular_spectrum:
    """
    Vector angular spectrum propagation (nonparaxial, evanescent included)
    """

    def __init__(self, wavelength):
        # ---------------- constants ----------------
        self.wavelength = wavelength   #unit: m
        self.k0 = 2 * jnp.pi / wavelength   #unit: m^-1
        self.omega0 = C.speed_of_light * self.k0   #unit: rad/s
        # --------------- FFT functions -------------

        # flags
        self.initialized = False
    @profile
    def initial_E(self, Ex, Ey, Ez, x_coordinate=[0], y_coordinate=[0]):
        """
        Initialize from real-space vector field at z=0
        Ex, Ey, Ez: (Nx, Ny), unit: V/m
        x_coordinate, y_coordinate: unit: m
        """
        self.Ex0 = jnp.asarray(Ex,dtype=jnp.float64)
        self.Ey0 = jnp.asarray(Ey,dtype=jnp.float64)
        self.Ez0 = jnp.asarray(Ez,dtype=jnp.float64)

        self.x_coordinate = jnp.asarray(x_coordinate,dtype=jnp.float64).flatten()
        self.y_coordinate = jnp.asarray(y_coordinate,dtype=jnp.float64).flatten()

        self.Nx = self.x_coordinate.size
        self.Ny = self.y_coordinate.size
        assert self.Ex0.shape == (self.Nx, self.Ny)
        assert self.Ey0.shape == (self.Nx, self.Ny)
        assert self.Ez0.shape == (self.Nx, self.Ny)
        #dx*dk=2π/N
        self.kx_coordinate, self.dkx, self.dx = make_k_coordinate_from_r_coordinate(self.x_coordinate)
        self.ky_coordinate, self.dky, self.dy = make_k_coordinate_from_r_coordinate(self.y_coordinate)
        self.dk=min(filter(lambda v: v>0.0, [self.dkx,self.dky]))
        self.E0 = jnp.stack([self.Ex0, self.Ey0, self.Ez0], axis=0)   #shape: (3, Nx, Ny), unit: V/m
        self.Ek0=fftn(self.E0, axes=(1,2)) * self.dx * self.dy   #shape: (3, Nx, Ny), unit: V·m
        self.E0_max=jnp.max(jnp.linalg.norm(self.E0,axis=0))
        self.kr, self.kz, self.k_mask, self.k, self.k_hat, self.project_operator = build_dispersion(self.kx_coordinate[:,jnp.newaxis], self.ky_coordinate[jnp.newaxis,:], k0=self.k0)
        self.Ek0_project=jnp.einsum('lmij,mij->lij',self.project_operator,self.Ek0)   #shape: (3, Nx, Ny), projected Ek0
        self.initialized = True
    @profile
    def initial_Ek(self, EKx, EKy, EKz, kx_coordinate=[0], ky_coordinate=[0]):
        """
        Initialize directly from k-space angular spectrum
        EKx, EKy, EKz: (Nkx, Nky), unit V·m
        kx_coordinate, ky_coordinate: 0-centered spatial frequencies (1/m)
        """
        self.EKx0 = jnp.asarray(EKx,dtype=jnp.complex128)
        self.EKy0 = jnp.asarray(EKy,dtype=jnp.complex128)
        self.EKz0 = jnp.asarray(EKz,dtype=jnp.complex128)
        self.kx_coordinate = jnp.asarray(kx_coordinate,dtype=jnp.float64).flatten()
        self.ky_coordinate = jnp.asarray(ky_coordinate,dtype=jnp.float64).flatten()
        self.Nx = self.kx_coordinate.size
        self.Ny = self.ky_coordinate.size
        print(f"Initialized angular spectrum shape: EKx0: {self.EKx0.shape}, EKy0: {self.EKy0.shape}, EKz0: {self.EKz0.shape}", flush=True)
        print(f"kx_coordinate size: {self.Nx}, ky_coordinate size: {self.Ny}")
        assert self.EKx0.shape == self.EKy0.shape == self.EKz0.shape==(self.Nx, self.Ny), "Input field components must have the same shape."
        self.kx, self.ky = jnp.meshgrid(self.kx_coordinate, self.ky_coordinate, indexing="ij")   #shape: (Nx, Ny)
        self.x_coordinate, self.dx, self.dkx= make_r_coordinate_from_k_coordinate(self.kx_coordinate)
        self.y_coordinate, self.dy, self.dky = make_r_coordinate_from_k_coordinate(self.ky_coordinate)
        self.dk=min(filter(lambda v: v>0.0, [self.dkx,self.dky]))
        self.Ek0=jnp.stack([self.EKx0, self.EKy0, self.EKz0], axis=0)   #shape: (3, Nx, Ny), unit: V·m
        self.E0=ifftn(self.Ek0, axes=(1,2)) / (self.dx * self.dy)   #shape: (3, Nx, Ny), unit: V/m
        self.E0_max=jnp.max(jnp.linalg.norm(self.E0,axis=0))
        self.kr, self.kz, self.k_mask, self.k, self.k_hat, self.project_operator = build_dispersion(self.kx_coordinate, self.ky_coordinate, k0=self.k0)
        self.Ek0_project=jnp.einsum('lmij,mij->lij',self.project_operator,self.Ek0)   #shape: (3, Nx, Ny), projected Ek0
        self.initialized = True
    @profile
    def propagate(self, z_coordinate,space='r'):
        """
        Propagate the angular spectrum to given z planes.

        Args:
            z_coordinate (_type_): _description_
            space (str, optional): _description_. Defaults to 'r'.
        """
        if not self.initialized:
            raise RuntimeError("Field not initialized. Call initial_E or initial_Ek.")
        z_coordinate = jnp.asarray(z_coordinate).flatten()
        Nz = z_coordinate.size
        print(f"Propagating to {Nz} z planes.")
        batch_propagate = jit(vmap(
            fun=propagate_z,
            in_axes=(0,None, None, None, None),
            out_axes=(3,3)   #out_axes=(3,3) will give shape: (3, Nx, Ny, Nz), or out_axes=(0,0) will give shape: (Nz, 3, Nx, Ny)
        ))
        Ek_all_z, Bk_all_z = batch_propagate(z_coordinate,self.Ek0_project,self.kz,self.k_hat,self.k_mask)   #shape: (3, Nx, Ny, Nz)
        print(f"Ek_all_z shape: {Ek_all_z.shape}, Bk_all_z shape: {Bk_all_z.shape}")
        if space=='r':
            E_all_z = jnp.real(ifftn(Ek_all_z, axes=(1,2)) / (self.dx * self.dy))   #shape: (3, Nx, Ny, Nz)
            B_all_z = jnp.real(ifftn(Bk_all_z, axes=(1,2)) / (self.dx * self.dy))   #shape: (3, Nx, Ny, Nz)
            print(f"E_all_z shape: {E_all_z.shape}, B_all_z shape: {B_all_z.shape}")
            check_divergence(Field=E_all_z, x_coordinate=self.x_coordinate, y_coordinate=self.y_coordinate, z_coordinate=z_coordinate, threshold=5e-1, scale_length=self.wavelength)
            check_divergence(Field=B_all_z, x_coordinate=self.x_coordinate, y_coordinate=self.y_coordinate, z_coordinate=z_coordinate, threshold=5e-1, scale_length=self.wavelength)
            return E_all_z, B_all_z
        elif space=='k':
            return Ek_all_z, Bk_all_z
        
    @profile
    def propagate_angular_spectrum(self, z_coordinate,space='r'):
        """
        Propagate angular spectrum to given z planes in real space.
        Not jitted. Backup method.
        Returns:
            E     : (3, Nx, Ny, Nz)
            B     : (3, Nx, Ny, Nz)
            phase : (Nx, Ny, Nz)
        """

        if not self.initialized:
            raise RuntimeError("Field not initialized. Call initial_E or initial_Ek.")
        z_coordinate = jnp.asarray(z_coordinate).flatten()
        Nz = z_coordinate.size
        print(f"Propagating to {Nz} z planes.")
        propagator=jnp.where((self.kr <= self.k0)[:,:,jnp.newaxis],
                         jnp.exp(1j * self.kz[:,:,jnp.newaxis] * z_coordinate[jnp.newaxis, jnp.newaxis, :]),
                         jnp.exp(-jnp.abs(self.kz[:,:,jnp.newaxis]) * jnp.abs(z_coordinate[jnp.newaxis, jnp.newaxis, :])))   #shape: (Nx, Ny, Nz)
        Ek_propagate=self.Ek0_project[:,:,:,jnp.newaxis]*propagator[jnp.newaxis, :, :, :]   #shape: (3, Nx, Ny, Nz), unit: V·m
        Ek_propagate_fft=ifftn(Ek_propagate, axes=(1,2)) / (self.dx * self.dy)   #shape: (3, Nx, Ny, Nz)
        E_propagate=jnp.real(Ek_propagate_fft)   #shape: (3, Nx, Ny, Nz)
        print(f"E_propagate shape: {E_propagate.shape}")
        #E_propagate_phase=jnp.angle(Ek_propagate_fft[0,:,:,:])   #shape: (Nx, Ny, Nz)
        Bk_propagate = jnp.cross(self.k[:,:,:,jnp.newaxis], Ek_propagate, axis=0) / self.omega0   #shape: (3, Nx, Ny, Nz), unit: V·s
        Bk_propagate_fft=ifftn(Bk_propagate, axes=(1,2)) / (self.dx * self.dy)   #shape: (3, Nx, Ny, Nz)
        B_propagate=jnp.real(Bk_propagate_fft)   #shape: (3, Nx, Ny, Nz), unit: V·s/m^2=T
        print(f"B_propagate shape: {B_propagate.shape}")
        check_divergence(Field=E_propagate, x_coordinate=self.x_coordinate, y_coordinate=self.y_coordinate, z_coordinate=z_coordinate, threshold=5e-1, scale_length=self.wavelength)
        check_divergence(Field=B_propagate, x_coordinate=self.x_coordinate, y_coordinate=self.y_coordinate, z_coordinate=z_coordinate, threshold=5e-1, scale_length=self.wavelength)
        return E_propagate, B_propagate
