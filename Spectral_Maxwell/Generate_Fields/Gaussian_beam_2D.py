import sys
from line_profiler import profile
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/EM_analyzer')
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/EM_analyzer/Spectral_Maxwell')
import os
import jax
jax.config.update("jax_enable_x64", True)
print(jax.local_device_count(),flush=True)
import jax.numpy as jnp
from typing import Tuple, Optional
import scipy.constants as C
from EM_analyzer.Spectral_Maxwell.kgrid import make_k_coordinate_from_r_coordinate
from EM_analyzer.Spectral_Maxwell.Angular_spectrum_method import Vector_angular_spectrum
from EM_analyzer.Spectral_Maxwell.Normal_variable_method import Spectral_Maxwell_Solver
from EM_analyzer.read_write import write_fields_to_nc
from rotate_3D import Rotation
class Gaussian_Beam_2D:
    """
    Gaussian beam class in 2D X-Z plane (non-paraxial theory)
    Electric field is polarized in the XY plane (Ex(x,z), Ey(x,z) in y=0 line), but field distribution is calculated in the X-Z plane
    """
    
    def __init__(
        self,
        wavelength: float,
        w0_lambda: float =5.0,
        phi_pol: float = 0.0,
        phi_cep: float = 0.0,
        a0: float = 1.0,
        r_resolution: float = 50.0,
        k_resolution: float = 50.0,
    ):
        """
        Initialize Gaussian beam in 2D X-Z plane using angular spectrum method.
        Parameters:
            wavelength: wavelength (m)
            w0_lambda: beam waist radius (in units of wavelength) - 1/e intensity half-width in x direction
            phi_pol: polarization direction angle (radians), defining the electric field polarization direction as (cos(phi_pol), sin(phi_pol),0)
            phi_cep: carrier-envelope phase (radians)
            a0: normalized peak electric field amplitude (unit: 1)
            r_resolution: real-space resolution (λ0/dx)
            k_resolution: wavevector-space resolution (k0/dkx). k_resolution in calculation may be larger than given
        """
        self.wavelength = wavelength   #unit: m
        self.period = self.wavelength / C.speed_of_light   #unit: s
        self.k0 = 2 * jnp.pi / self.wavelength   #unit: m^-1
        self.omega0 = self.k0 * C.speed_of_light  # unit: rad/s
        self.period=self.wavelength/C.speed_of_light
        self.Bc=(C.m_e*self.omega0)/(C.elementary_charge)   #unit: T. 1.338718e+04T for 800nm laser
        self.Ec=self.Bc*C.speed_of_light   #unit: V/m. 4.013376e+12V/m for 800nm laser
        self.w0 = w0_lambda * self.wavelength        #unit: m
        self.z_R = jnp.pi * self.w0**2 / self.wavelength   #unit: m, Rayleigh length
        self.phi_pol = phi_pol
        self.phi_cep = phi_cep
        self.a0 = a0        #unit: 1
        self.amp=self.a0*self.Ec   #unit: V/m
        
        self.dr = wavelength / r_resolution  # real space
        self.Nr = round(max(r_resolution*k_resolution/2,3.5*self.w0/self.dr))
        self.Nx = 2 * self.Nr+1
        
        # grid in real space
        self.x_coordinate = jnp.linspace(-self.Nr*self.dr, self.Nr*self.dr , self.Nx, endpoint=True,dtype=jnp.float64)  # shape: (Nx,), unit: m
        self.xmax=self.x_coordinate[-1]
        # grid in frequency space
        self.kx_coordinate, self.dkx,self.dx= make_k_coordinate_from_r_coordinate(self.x_coordinate)
        print(f'resolution: λ0/dx={self.wavelength/self.dx}, k0/dkx={self.k0/self.dkx}',flush=True)

        self.x = self.x_coordinate   # shape: (Nx,), unit: m
        self.kx = self.kx_coordinate   # shape: (Nx,), unit: m^-1
        #kz = sqrt(k0^2 - kx^2), real or complex
        self.kz = jnp.sqrt(self.k0**2 - self.kx**2 + 0j)   # shape: (Nx,), unit: m^-1

        # 计算初始角谱 (z=0)
        self.E_tilde0=self._compute_initial_angular_spectrum()
        self.Gaussian_Beam_2D_angular_spectrum=Vector_angular_spectrum(wavelength=self.wavelength)
        self.Gaussian_Beam_2D_angular_spectrum.initial_Ek(
            EKx=self.E_tilde0[0].reshape(self.Nx,1),
            EKy=self.E_tilde0[1].reshape(self.Nx,1),
            EKz=self.E_tilde0[2].reshape(self.Nx,1),
            kx_coordinate=self.kx_coordinate,
            ky_coordinate=jnp.array([0.0])
        )

    def _compute_initial_angular_spectrum(self):
        """
        Compute the initial angular spectrum of the Gaussian beam at z=0 plane in 2D X-Z geometry.
        
        In the 2D X-Z plane:
        - We have kx, but no ky
        - The electric field has 3 components: Ex, Ey, Ez
        - Transverse condition: k·E = kx*Ex + kz*Ez = 0 (since ky=0, Ey does not affect the transverse condition)
        - Polarization direction affects the relative amplitude of Ex and Ey
        """
        print("Initial angular spectrum at z=0 plane:", flush=True)
        
        # Compute the angular spectrum of Ex (Gaussian form)
        E_tilde0=self.amp*jnp.sqrt(jnp.pi) * self.w0 * jnp.exp(-(self.w0 * self.kx / 2)**2) * jnp.exp(1j * self.phi_cep)   #Spectrum of transverse E at z=0 plane, unit: V
        self.Ex_tilde0=E_tilde0 * jnp.cos(self.phi_pol)   #shape:(Nx,), unit: V
        self.Ey_tilde0=E_tilde0 * jnp.sin(self.phi_pol)   #shape:(Nx,), unit: V
        
        # Compute the angular spectrum of Ez using the transverse condition
        # Therefore: Ez = -(kx/kz) * Ex
        self.Ez_tilde0=jnp.where(jnp.abs(self.kz)>self.dkx/10, -(self.kx / self.kz) * self.Ex_tilde0, 0.0)
       
        self.E_tilde0 = jnp.stack([
            self.Ex_tilde0,
            self.Ey_tilde0,
            self.Ez_tilde0
        ])  # shape=(3, Nx), unit: V
        
        print(f"Initial angular spectrum shape: {self.E_tilde0.shape}", flush=True)
        return self.E_tilde0
    
    def propagate(self, z_coordinate=[0]):
        E_propagate, B_propagate=self.Gaussian_Beam_2D_angular_spectrum.propagate(z_coordinate=z_coordinate)
        #check_divergence(Field=E_propagate, x_coordinate=self.x_coordinate, y_coordinate=[0], z_coordinate=z_coordinate, threshold=1e-2, scale_length=self.wavelength)
        #check_divergence(Field=B_propagate, x_coordinate=self.x_coordinate, y_coordinate=[0], z_coordinate=z_coordinate, threshold=1e-2, scale_length=self.wavelength)
        self.E_propagate=E_propagate   #shape: (3, Nx, 1, Nz), unit: V/m
        self.B_propagate=B_propagate   #shape: (3, Nx, 1, Nz), unit: T
        return E_propagate, B_propagate

    def compute_optical_path(self, z_coordinate=[0]):
        """
        Paraxial optical path
        Parameters:
            z_coordinate: (Nz,), unit: m
        
        Returns:
            optical_path: optical path array (m), shape: (Nx, 1, Nz)
        """
        z_coordinate = jnp.asarray(z_coordinate).flatten()
        x,_,z=jnp.meshgrid(self.x_coordinate, jnp.array([0]),z_coordinate, indexing='ij')  # shape: (Nx,1, Nz)
        kappa_z=z_coordinate / (z_coordinate**2 + self.z_R**2)  # shape: (Nz,)
        optical_path = z + (x**2*kappa_z[jnp.newaxis,jnp.newaxis,:])/2  # shape: (Nx,1, Nz)
        return optical_path
    @profile
    def get_pulse(
        self, 
        FWHM_time: float,
        time_shift: float = 0.0, 
        theta: float = 0.0,
        working_dir: Optional[str] = None
        ):
        """
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
            theta: incidence angle, unit: radian
        Returns:
            EB_evolution_dict
        """
        tau_time=FWHM_time/jnp.sqrt(2*jnp.log(2))
        # calculate the temporal envelope
        Nz=2*max(round(5*FWHM_time*C.speed_of_light/self.dx), round(self.Nr*jnp.sin(theta)))+1
        z_coordinate=jnp.linspace(-Nz//2*self.dx, Nz//2*self.dx, Nz,endpoint=True,dtype=jnp.float64)  # z位置 (米
        optical_path=self.compute_optical_path(z_coordinate=z_coordinate)   #shape: (Nx, 1, Nz)
        envelope=jnp.exp(-jnp.square((optical_path/C.speed_of_light)/tau_time))   #shape: (Nx, 1, Nz)
        # propagate the beam near the focus
        self.E_propagate, self.B_propagate=self.Gaussian_Beam_2D_angular_spectrum.propagate(z_coordinate=z_coordinate,space='r')   #shape: (3, Nx, 1, Nz)
        E_focus_pulse=self.E_propagate * envelope[jnp.newaxis,:,:,:]   #shape: (3, Nx, 1, Nz)
        B_focus_pulse=self.B_propagate * envelope[jnp.newaxis,:,:,:]   #shape: (3, Nx, 1, Nz)
        # rotate the fields
        if theta>0.0001:
            Field_rotation=Rotation(theta=theta)
            E_focus_rotate=Field_rotation.rotate(A=E_focus_pulse, x0_coordinate=self.x_coordinate, y0_coordinate=jnp.array([0.0]), z0_coordinate=z_coordinate,x1_coordinate=self.x_coordinate, y1_coordinate=jnp.array([0.0]), z1_coordinate=z_coordinate,space='r', direction='1->0'  )
            B_focus_rotate=Field_rotation.rotate(A=B_focus_pulse, x0_coordinate=self.x_coordinate, y0_coordinate=jnp.array([0.0]), z0_coordinate=z_coordinate,x1_coordinate=self.x_coordinate, y1_coordinate=jnp.array([0.0]), z1_coordinate=z_coordinate,space='r', direction='1->0'  )
        else:
            E_focus_rotate=E_focus_pulse
            B_focus_rotate=B_focus_pulse
        # propagate to the desired center_time
        if jnp.abs(time_shift/self.period)>0.001:
            EB_Spectral_Maxwell=Spectral_Maxwell_Solver(E0=E_focus_rotate, B0=B_focus_rotate, x_coordinate=self.x_coordinate, y_coordinate=jnp.array([0.0]), z_coordinate=z_coordinate)
            EB_evolution_dict=EB_Spectral_Maxwell.evolution(evolution_time=time_shift,window_shift_velocity=C.speed_of_light*jnp.array([jnp.sin(theta),0.0,jnp.cos(theta)]))
            x_coordinate_rotated=EB_evolution_dict['x_coordinate']
            z_coordinate_rotated=EB_evolution_dict['z_coordinate']
        else:
            EB_evolution_dict={
                "E": E_focus_rotate,   #shape=(3, Nx, 1, Nz), unit: V/m
                "B": B_focus_rotate,   #shape=(3, Nx, 1, Nz), unit: T
                "x_coordinate": self.x_coordinate,   #shape=(Nx,), unit: m
                "y_coordinate": jnp.array([0.0]),   #shape=(1,), unit: m
                "z_coordinate": z_coordinate,   #shape=(Nz,), unit: m
            }
            x_coordinate_rotated=self.x_coordinate
            z_coordinate_rotated=z_coordinate
        if working_dir is not None:
            os.makedirs(working_dir, exist_ok=True)
            # save the fields to NetCDF files
            self.write_fields(E_field=EB_evolution_dict['E'], B_field=EB_evolution_dict['B'], x_coordinate=EB_evolution_dict['x_coordinate'],z_coordinate=EB_evolution_dict['z_coordinate'], name="Field_t=%+05.01fT0"%(time_shift/self.period),working_dir=working_dir)
            # save the parameters to a text file
            with open(os.path.join(working_dir,'Initialize_Field.txt'),'a') as f:
                f.write(f'Gaussian Beam 2D parameters:\n')
                f.write(f'wavelength= {self.wavelength} m\n')
                f.write(f'w0/λ0= {self.w0/self.wavelength} \n')
                f.write(f'FWHM_time= {FWHM_time} s\n')
                f.write(f'Incidence angle theta= {theta} rad\n')
                f.write(f'a0= {self.a0}\n')
                f.write(f'λ0/dx= {self.wavelength/self.dx}\n')
                f.write(f'x_min/λ0= {z_coordinate_rotated[0]/self.wavelength}\n')
                f.write(f'x_max/λ0= {z_coordinate_rotated[-1]/self.wavelength}\n')
                f.write(f'y_min/λ0= {x_coordinate_rotated[0]/self.wavelength}\n')
                f.write(f'y_max/λ0= {x_coordinate_rotated[-1]/self.wavelength}\n')
                f.write(f'\n')
        return EB_evolution_dict

    def write_fields(self, E_field, B_field, x_coordinate,z_coordinate, name="Gaussian_Beam_2D",working_dir="."):
        """
        Write electric and magnetic fields to NetCDF files in 2D format.
        Convert xyz coordinates (shape=(Nx, 1, Nz)) to xy coordinates (shape=(Nz, Nx)) for 2D representation.
        Parameters:
            E_field: Electric field array, shape (3, Nx, 1, Nz)
            B_field: Magnetic field array, shape (3, Nx, 1, Nz)
            z_coordinate: z position array (meters)
            name: Output file name prefix
        """
        write_fields_to_nc(
            field_dict_list=[
                {"data": E_field[0,:,0,:].T, "name": "Ey", "units": "V/m"},
                {"data": E_field[1,:,0,:].T, "name": "Ez", "units": "V/m"},
                {"data": E_field[2,:,0,:].T, "name": "Ex", "units": "V/m"},
                {"data": B_field[0,:,0,:].T, "name": "By", "units": "T"},
                {"data": B_field[1,:,0,:].T, "name": "Bz", "units": "T"},
                {"data": B_field[2,:,0,:].T, "name": "Bx", "units": "T"},
            ],
            coordinate_dict_list=[
                {"coordinate": z_coordinate, "name": "x", "units": "m"},
                {"coordinate": x_coordinate, "name": "y", "units": "m"},
            ],
            nc_name=f"{name}.nc",working_dir=working_dir,
        )
        return name
