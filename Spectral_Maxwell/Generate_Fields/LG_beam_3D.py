import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336')
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/EM_analyzer')
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/EM_analyzer/Spectral_Maxwell')
import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import jit, vmap
from scipy.special import genlaguerre
from scipy.special import factorial
from typing import Tuple, Optional
import scipy.constants as C
from EM_analyzer.Spectral_Maxwell.kgrid import make_k_coordinate_from_r_coordinate
from EM_analyzer.Spectral_Maxwell.Angular_spectrum_method import Vector_angular_spectrum
from EM_analyzer.Spectral_Maxwell.Normal_variable_method import Spectral_Maxwell_Solver
from EM_analyzer.read_write import write_fields_to_nc
from EM_analyzer.rotate_3D import Rotation
class LG_beam:
    """
    (non-paraxial theory)
    Tight-focusing Laguerre-Gaussian beam based on the angular spectrum method.
    Using vector diffraction theory, strictly satisfying the transverse wave condition.
    LG beam focused at z=0. E Polarization along x direction. 
    Propagation along +z direction.
    """
    
    def __init__(
        self,
        wavelength: float,
        w0_lambda: float =5.0,
        phi_cep: float = 0.0,
        l: int=0,
        p: int=0,
        a0: float = 1.0,
        r_resolution: float = 50.0,
        k_resolution: float = 50.0,
    ):
        """
        Initialize Gaussian beam in 2D X-Z plane using angular spectrum method.
        Parameters:
            wavelength: wavelength (m)
            w0_lambda: beam waist radius (in units of wavelength) - 1/e intensity half-width in x direction
            phi_cep: carrier-envelope phase (radians)
            a0: normalized peak electric field amplitude (unit: 1)
            r_resolution: real-space resolution (λ0/dx)
            k_resolution: wavevector-space resolution (k0/dkx). k_resolution in calculation may be larger than given
        """
        self.wavelength = wavelength   #unit: m
        self.period = self.wavelength / C.speed_of_light   #unit: s
        self.k0 = 2 * jnp.pi / self.wavelength   #unit: m^-1
        self.omega0 = self.k0 * C.speed_of_light  # unit: rad/s
        self.Bc=(C.m_e*self.omega0)/(C.elementary_charge)   #unit: T. 1.338718e+04T for 800nm laser
        self.Ec=self.Bc*C.speed_of_light   #unit: V/m. 4.013376e+12V/m for 800nm laser
        self.w0 = w0_lambda * self.wavelength        #unit: m
        self.z_R = jnp.pi * self.w0**2 / self.wavelength   #unit: m, Rayleigh length
        self.phi_cep = phi_cep
        self.l = l
        self.p = p
        self.a0 = a0        #unit: 1
        self.amp=self.a0*self.Ec   #unit: V/m
        
        self.dr = wavelength / r_resolution  # real space
        self.Nr = round(max(r_resolution*k_resolution/2,2.5*self.w0/self.dr,1.5*self.p*wavelength/self.dr))  # number of points in one direction (from center to edge)
        self.Nx = 2 * self.Nr+1
        self.Ny = 2 * self.Nr+1
        # grid in real space
        self.x_coordinate = jnp.linspace(-self.Nr*self.dr, self.Nr*self.dr , self.Nx, endpoint=True,dtype=jnp.float64)  # shape: (Nx,), unit: m
        self.y_coordinate=jnp.linspace(-self.Nr*self.dr, self.Nr*self.dr , self.Ny, endpoint=True,dtype=jnp.float64)  # shape: (Nx,), unit: m
        self.xmax=self.x_coordinate[-1]
        # grid in frequency space
        self.kx_coordinate, self.dkx,self.dx= make_k_coordinate_from_r_coordinate(self.x_coordinate)
        self.ky_coordinate, self.dky,self.dy= make_k_coordinate_from_r_coordinate(self.y_coordinate)
        print(f'resolution: λ0/dx={self.wavelength/self.dx}, k0/dkx={self.k0/self.dkx}')
        
        self.x,self.y=jnp.meshgrid(self.x_coordinate, self.y_coordinate, indexing='ij')   # shape: (Nx, Ny), unit: m
        self.kx,self.ky=jnp.meshgrid(self.kx_coordinate, self.ky_coordinate, indexing='ij')   # shape: (Nx, Ny), unit: m^-1
        self.krho=jnp.hypot(self.kx,self.ky)   # shape: (Nx, Ny), unit: m^-1. kρ=sqrt(kx^2+ky^2)
        self.kphi=jnp.arctan2(self.ky,self.kx)   # shape: (Nx, Ny), unit: rad. kφ=arctan(ky/kx)
        #kz = sqrt(k0^2 - kx^2-ky^2), real or complex
        self.kz = jnp.where(self.krho <= self.k0,
                         jnp.sqrt(self.k0**2 - self.krho**2),
                         1j * jnp.sqrt(self.krho**2 - self.k0**2))   #evanescent. shape: (Nx, Ny), unit: m^-1
        E_tilde_0=self.compute_initial_angular_spectrum()   # shape: (3, Nx, Ny), unit: V·m
        # Initialize angular spectrum solver
        self.AS_solver=Vector_angular_spectrum(wavelength=self.wavelength)
        self.AS_solver.initial_Ek(EKx=E_tilde_0[0], EKy=E_tilde_0[1], EKz=E_tilde_0[2],
                                    kx_coordinate=self.kx_coordinate, ky_coordinate=self.ky_coordinate) 
        
    def compute_initial_angular_spectrum(self):
        """
        Compute initial angular spectrum at z=0 plane for LG beam.
        Returns:
            E_tilde_0: initial electric field angular spectrum, shape (3, Nx, Ny), unit: V/m/(m^-2)
        """
        # LG mode in r-space:
        # U_lp(ρ, φ) =√((2 p!)/(π (p+|l|)!)) (√2  r/w_0 )^|l| ×L_p^|l|  (2(r/w_0 )^2 )  exp⁡(-(r/w_0 )^2 )  exp⁡(ilϕ+iϕ_CEP )
        # LG mode in k-space:
        # FU_lp(kρ, kφ) = w_0^2×√((2 π p!)/(p+|l|)!)×(-i)^l×(-1)^p×(k_ρ·w_0/√2)^|l| ×L_p^|l|(2*(k_ρ·w_0/2)^2) ×exp⁡(-(k_ρ·w_0/2)^2)  exp⁡(i l kφ+i ϕ_CEP)
        
        
        # complete scalar angular spectrum
        L_pl=genlaguerre(self.p,abs(self.l))
        U = self.amp*self.w0**2*np.sqrt(2*np.pi*factorial(self.p)/(factorial(self.p+abs(self.l))))*(-1j)**self.l*(-1)**self.p*(self.krho*self.w0/np.sqrt(2))**abs(self.l)*L_pl(2*(self.krho*self.w0/2)**2)*np.exp(-(self.krho*self.w0/2)**2)*np.exp(1j*self.l*self.kphi+1j*self.phi_cep)   # shape: (Nx, Ny), unit: V·m
        
        # Electric field components in k-space
        Ex_tilde_0 = U    # shape: (Nx, Ny)
        Ey_tilde_0 = jnp.zeros(shape=(self.Nx, self.Ny))   # shape: (Nx, Ny)
        
        # Ez component from transverse wave condition: k·E = 0
        Ez_tilde_0 = jnp.where(jnp.abs(self.kz)>self.dkx/10, -(self.kx / self.kz) * Ex_tilde_0, 0.0)
        
        # Stack into 3D array
        E_tilde_0 = jnp.stack([
            Ex_tilde_0,
            Ey_tilde_0,
            Ez_tilde_0
        ], axis=0)  # shape (3, Nx, Ny)
        print(f"Initial angular spectrum shape: {E_tilde_0.shape}")
        return E_tilde_0   # unit: V·m
    def propagate(self, z_coordinate=[0]):
        E_propagate, B_propagate, E_propagate_phase=self.AS_solver.propagate_angular_spectrum(z_coordinate=z_coordinate)
        #check_divergence(Field=E_propagate, x_coordinate=self.x_coordinate, y_coordinate=self.y_coordinate, z_coordinate=z_coordinate, threshold=1e-2, scale_length=self.wavelength)
        #check_divergence(Field=B_propagate, x_coordinate=self.x_coordinate, y_coordinate=self.y_coordinate, z_coordinate=z_coordinate, threshold=1e-2, scale_length=self.wavelength)
        self.E_propagate=E_propagate   #shape: (3, Nx, Ny, Nz), unit: V/m
        self.B_propagate=B_propagate   #shape: (3, Nx, Ny, Nz), unit: T
        return E_propagate, B_propagate, E_propagate_phase
    def compute_optical_path(self, z_coordinate=[0]):
        """
        Paraxial optical path
        Parameters:
            z_coordinate: (Nz,), unit: m
        
        Returns:
            optical_path: optical path array (m), shape: (Nx, Ny, Nz)
        """
        z_coordinate = jnp.asarray(z_coordinate).flatten()
        x,y,z=jnp.meshgrid(self.x_coordinate, self.y_coordinate, z_coordinate, indexing='ij')  # shape: (Nx,Ny, Nz)
        kappa_z=z_coordinate / (z_coordinate**2 + self.z_R**2)  # shape: (Nz,)
        optical_path = z + ((x**2 + y**2)*kappa_z[jnp.newaxis,jnp.newaxis,:])/2  # shape: (Nx,Ny, Nz)
        return optical_path

    def get_pulse(self, FWHM_time: float,time_shift: float = 0.0,phi=0.0, psi=0.0,  theta=0.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        The wave is a pulsed Laguerre-Gaussian beam in 3D space.
        Propagation direction ek=(cos𝜓·sin𝜃,sin𝜓·sin𝜃,cos𝜃)
        P polarization direction ep=(cos𝜙·cos𝜓·cos𝜃 - sin𝜙·sin𝜓, cos𝜙·sin𝜓·cos𝜃 + sin𝜙·cos𝜓, -cos𝜙·sin𝜃)
        S polarization direction es=(-sin𝜙·cos𝜓·cos𝜃 - cos𝜙·sin𝜓, -sin𝜙·sin𝜓·cos𝜃 + cos𝜙·cos𝜓, sin𝜙·sin𝜃)
        The pulse is obtained by: 
        1.Compute the beam propagation in z direction without temporal envelope by using angular spectrum method. The z is in the range [-3*FWHM_time*c, 3*FWHM_time*c].
        2.Multiply a temporal Gaussian envelope to the propagated fields. The envelope is defined as:
            envelope = exp(-((optical_path/c)/tau_time)^2)
        where tau_time = FWHM_time/sqrt(2*ln(2))
        3.Rotate the fields to the desired incidence angle theta if needed.
        4.Propagate the fields to the desired center_time by using normal variables in k-space method.
        Parameters:
            FWHM_time: The full width at half maximum of the intensity. unit: s
            time_shift: pulse center time shift, unit: s
            phi: polarization angle, unit: radian
            psi: azimuthal angle, unit: radian
            theta: incidence angle, unit: radian
        Returns:
            EB_evolution_dict
        """
        tau_time=FWHM_time/jnp.sqrt(2*jnp.log(2))
        # calculate the temporal envelope
        #Nz=2*max(round(2*FWHM_time*C.speed_of_light/self.dx), self.Nr)+1
        Nz=2*round(2*FWHM_time*C.speed_of_light/self.dx)+1   #make Nz odd
        z_coordinate=jnp.linspace(-Nz//2*self.dx, Nz//2*self.dx, Nz,endpoint=True,dtype=jnp.float64)  # z位置 (米
        optical_path=self.compute_optical_path(z_coordinate=z_coordinate)   #shape: (Nx, Ny, Nz)
        envelope=jnp.exp(-jnp.square((optical_path/C.speed_of_light)/tau_time))   #shape: (Nx, Ny, Nz)
        # propagate the beam near the focus
        self.propagate(z_coordinate=z_coordinate)
        E_focus_pulse=self.E_propagate * envelope[jnp.newaxis,:,:,:]   #shape: (3, Nx, Ny, Nz)
        B_focus_pulse=self.B_propagate * envelope[jnp.newaxis,:,:,:]   #shape: (3, Nx, Ny, Nz)
        # rotate the fields
        if theta>0.0001 or phi>0.0001 or psi>0.0001:
            Field_rotation=Rotation(phi=phi, psi=psi, theta=theta)
            E_focus_rotate=Field_rotation.rotate(A=E_focus_pulse, x0_coordinate=self.x_coordinate, y0_coordinate=self.y_coordinate, z0_coordinate=z_coordinate,space='k', direction='1->0'  )
            B_focus_rotate=Field_rotation.rotate(A=B_focus_pulse, x0_coordinate=self.x_coordinate, y0_coordinate=self.y_coordinate, z0_coordinate=z_coordinate,space='k', direction='1->0'  )
        else:
            E_focus_rotate=E_focus_pulse
            B_focus_rotate=B_focus_pulse
        # propagate to the desired center_time
        if time_shift/self.period>0.001:
            EB_Spectral_Maxwell=Spectral_Maxwell_Solver(E0=E_focus_rotate, B0=B_focus_rotate, x_coordinate=self.x_coordinate, y_coordinate=self.y_coordinate, z_coordinate=z_coordinate)
            EB_evolution_dict=EB_Spectral_Maxwell.evolution(evolution_time=time_shift,window_shift_velocity=C.speed_of_light*jnp.array([jnp.sin(theta)*jnp.cos(psi),jnp.sin(theta)*jnp.sin(psi),jnp.cos(theta)]))
            x_coordinate_rotated=EB_evolution_dict['x_coordinate']
            y_coordinate_rotated=EB_evolution_dict['y_coordinate']
            z_coordinate_rotated=EB_evolution_dict['z_coordinate']
        else:
            EB_evolution_dict={
                "E": E_focus_rotate,   #shape=(3, Nx, 1, Nz), unit: V/m
                "B": B_focus_rotate,   #shape=(3, Nx, 1, Nz), unit: T
                "x_coordinate": self.x_coordinate,   #shape=(Nx,), unit: m
                "y_coordinate": self.y_coordinate,   #shape=(1,), unit: m
                "z_coordinate": z_coordinate,   #shape=(Nz,), unit: m
            }
            x_coordinate_rotated=self.x_coordinate
            y_coordinate_rotated=self.y_coordinate
            z_coordinate_rotated=z_coordinate
        # save the fields to NetCDF files
        self.write_fields_to_nc(E_field=EB_evolution_dict['E'], B_field=EB_evolution_dict['B'], x_coordinate=EB_evolution_dict['x_coordinate'],y_coordinate=EB_evolution_dict['y_coordinate'], z_coordinate=EB_evolution_dict['z_coordinate'], name="Field_t=%+05.01fT0"%(time_shift/self.period),working_dir=working_dir)
        # save the parameters to a text file
        with open(os.path.join(working_dir,'Initialize_Field.txt'),'a') as f:
            f.write(f'Laguerre-Gaussian beam 3D parameters:\n')
            f.write(f'wavelength= {self.wavelength} m\n')
            f.write(f'w0/λ0= {self.w0/self.wavelength} \n')
            f.write(f'FWHM_time= {FWHM_time} s\n')
            f.write(f'Polarization angle phi= {phi} rad\n')
            f.write(f'Azimuthal angle psi= {psi} rad\n')
            f.write(f'Incidence angle theta= {theta} rad\n')
            f.write(f'a0= {self.a0}\n')
            f.write(f'λ0/dx= {self.wavelength/self.dx}\n')
            f.write(f'x_min/λ0= {x_coordinate_rotated[0]/self.wavelength}\n')
            f.write(f'x_max/λ0= {x_coordinate_rotated[-1]/self.wavelength}\n')
            f.write(f'y_min/λ0= {y_coordinate_rotated[0]/self.wavelength}\n')
            f.write(f'y_max/λ0= {y_coordinate_rotated[-1]/self.wavelength}\n')
            f.write(f'z_min/λ0= {z_coordinate_rotated[0]/self.wavelength}\n')
            f.write(f'z_max/λ0= {z_coordinate_rotated[-1]/self.wavelength}\n')
            f.write(f'\n')
        return EB_evolution_dict

    def write_fields_to_nc(self, E_field, B_field, x_coordinate,y_coordinate,z_coordinate, name="LG_beam",working_dir="."):
        """
        Write electric and magnetic fields to NetCDF files in 3D format.
        Convert xyz coordinates (shape=(Nx, 1, Nz)) to xy coordinates (shape=(Nz, Nx)) for 2D representation.
        Parameters:
            E_field: Electric field array, shape (3, Nx, Ny, Nz)
            B_field: Magnetic field array, shape (3, Nx, Ny, Nz)
            z_coordinate: z position array (meters)
            name: Output file name prefix
        """
        write_fields_to_nc(
            field_dict_list=[
                {"field": E_field[0], "name": "Ex", "units": "V/m"},
                {"field": E_field[1], "name": "Ey", "units": "V/m"},
                {"field": E_field[2], "name": "Ez", "units": "V/m"},
                {"field": B_field[0], "name": "Bx", "units": "T"},
                {"field": B_field[1], "name": "By", "units": "T"},
                {"field": B_field[2], "name": "Bz", "units": "T"},
            ],
            coordinate_dict_list=[
                {"coordinate": x_coordinate, "name": "x", "units": "m"},
                {"coordinate": y_coordinate, "name": "y", "units": "m"},
                {"coordinate": z_coordinate, "name": "z", "units": "m"},
            ],
            nc_name=f"{name}.nc",working_dir=working_dir
        )
        return name

working_dir="/scratch/gpfs/MIKHAILOVA/zl8336/Gaussian_beam_pulse/LG20cpl/l=-3,p=3"

if __name__ == "__main__":
    laser_lambda = 0.8*C.micron		# Laser wavelength, unit:m
    laser_f0=1/laser_lambda   #unit: m^-1
    laser_k0=2*C.pi*laser_f0
    laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
    laser_period=laser_lambda/C.speed_of_light
    laser_Bc=(C.m_e*laser_omega0)/(C.elementary_charge)   #unit: T. 1.338718e+04T for 800nm laser
    laser_Ec=laser_Bc*C.speed_of_light   #unit: V/m. 4.013376e+12V/m for 800nm laser
    laser_a0 = 1		# Laser field strength
    laser_amp=laser_a0*laser_Ec   #unit: V/m
    laser_FWHM=5*C.femto   #The full width at half maximum of the intensity.
    laser_tau=laser_FWHM/jnp.sqrt(2*jnp.log(2)) 
    laser_w0_lambda= 2.0   # Beam waist radius (in units of wavelength) - 1/e intensity half-width in x direction
    laser_zR_lambda=C.pi*laser_w0_lambda**2
    laser_w0=laser_w0_lambda*laser_lambda
    laser_zR=laser_zR_lambda*laser_lambda
    l=-3
    p=3
    gaussian_beam = LG_beam(
        wavelength=laser_lambda,
        w0_lambda=laser_w0_lambda,
        phi_cep=0.0,
        l=l,
        p=p,
        a0=laser_a0,
        r_resolution=40,
        k_resolution=40,
    )
    gaussian_beam.get_pulse(FWHM_time=laser_FWHM,time_shift=0, theta=0.0, phi=0.0, psi=0.0)

exit(0)