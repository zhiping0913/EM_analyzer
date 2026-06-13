import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/start')
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
from typing import Tuple,Optional
import jax.numpy as jnp
import scipy.constants as C
import os
import xarray as xr
import pandas as pd
from Lorentz.Lorentz_transform import LorentzTransform
from read_write import read_nc,write_fields_to_nc,read_sdf

theta_degree=45
theta_rad=jnp.radians(theta_degree)

laser_lambda = 0.875*C.micron		# Laser wavelength
laser_f0=1/laser_lambda
laser_k0=2*C.pi*laser_f0
laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_period=laser_lambda/C.speed_of_light
laser_Bc=(C.m_e*laser_omega0)/(C.elementary_charge)
laser_Ec=laser_Bc*C.speed_of_light   #4.013376e+12V/m
laser_FWHM=8*C.femto   #The full width at half maximum of the intensity.
laser_tau=laser_FWHM/jnp.sqrt(2*jnp.log(2)) 
#laser_tau=laser_period/jnp.sqrt(jnp.pi)
laser_Nc=laser_omega0**2*C.m_e*C.epsilon_0/C.elementary_charge**2


laser_lambda_M=laser_lambda/jnp.cos(theta_rad)
laser_period_M=laser_period/jnp.cos(theta_rad)
laser_tau_M=laser_tau/jnp.cos(theta_rad)
laser_f0_M=laser_f0*jnp.cos(theta_rad)
laser_k0_M=laser_k0*jnp.cos(theta_rad)
laser_Bc_M=laser_Bc*jnp.cos(theta_rad)
laser_Ec_M=laser_Ec*jnp.cos(theta_rad)




lorentz_transform = LorentzTransform(
    beta_x=0,
    beta_y=jnp.sin(theta_rad),
    beta_z=0
    )
print(lorentz_transform)
print(lorentz_transform.transform_matrix)


def transform_current(
    x_coordinate:jnp.ndarray,
    t_coordinate:Optional[jnp.ndarray]=jnp.array([0.0]).flatten(),
    Number_Density_M:Optional[jnp.ndarray]=None,
    Jx_M:Optional[jnp.ndarray]=None,
    Jy_M:Optional[jnp.ndarray]=None,
    Jz_M:Optional[jnp.ndarray]=None,
    charge=-C.elementary_charge,
    ):
    """_summary_

    Args:
        x_coordinate (jnp.ndarray): _description_
        Number_Density_M (Optional[jnp.ndarray], optional): _description_. Defaults to None. Unit: 1/m^3. 
        Jx_M (Optional[jnp.ndarray], optional): _description_. Defaults to None. Unit: A/m^2.
        Jy_M (Optional[jnp.ndarray], optional): _description_. Defaults to None. Unit: A/m^2.
        Jz_M (Optional[jnp.ndarray], optional): _description_. Defaults to None. Unit: A/m^2.
        charge (_type_, optional): _description_. Defaults to -C.elementary_charge. Unit: C. The charge of the particles corresponding to the number density and current density.
    """
    x_coordinate=jnp.asarray(x_coordinate).flatten()
    t_coordinate=jnp.asarray(t_coordinate).flatten()
    Nx=x_coordinate.size
    Nt=t_coordinate.size
    if Number_Density_M is None:
        Number_Density_M=jnp.zeros(shape=(Nt,Nx))
    else:
        Number_Density_M=jnp.asarray(Number_Density_M)
    if Jx_M is None:
        Jx_M=jnp.zeros(shape=(Nt,Nx))
    else:
        Jx_M=jnp.asarray(Jx_M)
    if Jy_M is None:
        Jy_M=jnp.zeros(shape=(Nt,Nx))
    else:
        Jy_M=jnp.asarray(Jy_M)
    if Jz_M is None:
        Jz_M=jnp.zeros(shape=(Nt,Nx))
    else:
        Jz_M=jnp.asarray(Jz_M)
    assert Number_Density_M.shape==(Nt,Nx) and Jx_M.shape==(Nt,Nx) and Jy_M.shape==(Nt,Nx) and Jz_M.shape==(Nt,Nx)
    Four_current_L=lorentz_transform.transform_field(
        component0=Number_Density_M*charge,  # Convert number density to charge density
        component1=Jx_M,
        component2=Jy_M,
        component3=Jz_M,
        four_vector_type='four_current',
        direction='1->0'
    )
    Number_Density_L=Four_current_L[0]/charge  # Convert charge density back to number density
    Jx_L=Four_current_L[1]
    Jy_L=Four_current_L[2]
    Jz_L=Four_current_L[3]
    return Number_Density_L,Jx_L,Jy_L,Jz_L

def transform_EM(
    x_coordinate:jnp.ndarray,
    t_coordinate:Optional[jnp.ndarray]=jnp.array([0.0]).flatten(),
    Ex_M:Optional[jnp.ndarray]=None,
    Ey_M:Optional[jnp.ndarray]=None,
    Ez_M:Optional[jnp.ndarray]=None,
    Bx_M:Optional[jnp.ndarray]=None,
    By_M:Optional[jnp.ndarray]=None,
    Bz_M:Optional[jnp.ndarray]=None,
    ):
    """_summary_

    Args:
        x_coordinate (jnp.ndarray): _description_
        Ex_M (Optional[jnp.ndarray], optional): _description_. Defaults to None. Unit: V/m.
        Ey_M (Optional[jnp.ndarray], optional): _description_. Defaults to None. Unit: V/m.
        Ez_M (Optional[jnp.ndarray], optional): _description_. Defaults to None. Unit: V/m.
        Bx_M (Optional[jnp.ndarray], optional): _description_. Defaults to None. Unit: T.
        By_M (Optional[jnp.ndarray], optional): _description_. Defaults to None. Unit: T.
        Bz_M (Optional[jnp.ndarray], optional): _description_. Defaults to None. Unit: T.
    """
    x_coordinate=jnp.asarray(x_coordinate).flatten()
    t_coordinate=jnp.asarray(t_coordinate).flatten()
    Nx=x_coordinate.size
    Nt=t_coordinate.size
    if Ex_M is None:
        Ex_M=jnp.zeros(shape=(Nt,Nx))
    else:
        Ex_M=jnp.asarray(Ex_M)
    if Ey_M is None:
        Ey_M=jnp.zeros(shape=(Nt,Nx))
    else:
        Ey_M=jnp.asarray(Ey_M)
    if Ez_M is None:
        Ez_M=jnp.zeros(shape=(Nt,Nx))
    else:
        Ez_M=jnp.asarray(Ez_M)
    if Bx_M is None:
        Bx_M=jnp.zeros(shape=(Nt,Nx))
    else:
        Bx_M=jnp.asarray(Bx_M)
    if By_M is None:
        By_M=jnp.zeros(shape=(Nt,Nx))
    else:
        By_M=jnp.asarray(By_M)
    if Bz_M is None:
        Bz_M=jnp.zeros(shape=(Nt,Nx))
    else:
        Bz_M=jnp.asarray(Bz_M)
    assert Ex_M.shape==(Nt,Nx) and Ey_M.shape==(Nt,Nx) and Ez_M.shape==(Nt,Nx) and Bx_M.shape==(Nt,Nx) and By_M.shape==(Nt,Nx) and Bz_M.shape==(Nt,Nx)
    E_L,B_L=lorentz_transform.transform_electromagnetic_field(
        E_field=[Ex_M,Ey_M,Ez_M],
        B_field=[Bx_M,By_M,Bz_M],
        direction='1->0',
    )
    Ex_L,Ey_L,Ez_L=E_L
    Bx_L,By_L,Bz_L=B_L
    return Ex_L,Ey_L,Ez_L,Bx_L,By_L,Bz_L


working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Small_a0/test/L+D=0.2/L_front=0.05,L_rear=0.05'
data_dict=read_nc(os.path.join(working_dir,'Summarize_Field_M_Frame.nc'),key_name_list=['Ex','Ey','Ez','Bx','By','Bz','N_e','Jx_e','Jy_e','Jz_e'])
print(data_dict.keys())
x_coordinate=data_dict['x']
time=data_dict['time']
Ex_M=data_dict['Ex']
Ey_M=data_dict['Ey']
Ez_M=data_dict['Ez']
Bx_M=data_dict['Bx']
By_M=data_dict['By']
Bz_M=data_dict['Bz']
N_e_M=data_dict['N_e']
Jx_e_M=data_dict['Jx_e']
Jy_e_M=data_dict['Jy_e']
Jz_e_M=data_dict['Jz_e']
N_e_L,Jx_e_L,Jy_e_L,Jz_e_L=transform_current(
    x_coordinate=x_coordinate,t_coordinate=time,
    Number_Density_M=N_e_M,
    Jx_M=Jx_e_M,
    Jy_M=Jy_e_M,
    Jz_M=Jz_e_M
    )
Ex_L,Ey_L,Ez_L,Bx_L,By_L,Bz_L=transform_EM(
    x_coordinate=x_coordinate,
    t_coordinate=time,
    Ex_M=Ex_M,
    Ey_M=Ey_M,
    Ez_M=Ez_M,
    Bx_M=Bx_M,
    By_M=By_M,
    Bz_M=Bz_M
)

write_fields_to_nc(
    coordinate_dict_list=[
        {'name':'time','coordinate':time,'units':'s','long_name':'Time'},
        {'name':'x','coordinate':x_coordinate,'units':'m','long_name':'X Coordinate'},
    ],
    field_dict_list=[
        {'name':'Ex','data':Ex_L,'units':'V/m','long_name':'Transformed Electric Field Ex in Lab Frame'},
        {'name':'Ey','data':Ey_L,'units':'V/m','long_name':'Transformed Electric Field Ey in Lab Frame'},
        {'name':'Ez','data':Ez_L,'units':'V/m','long_name':'Transformed Electric Field Ez in Lab Frame'},
        {'name':'Bx','data':Bx_L,'units':'T','long_name':'Transformed Magnetic Field Bx in Lab Frame'},
        {'name':'By','data':By_L,'units':'T','long_name':'Transformed Magnetic Field By in Lab Frame'},
        {'name':'Bz','data':Bz_L,'units':'T','long_name':'Transformed Magnetic Field Bz in Lab Frame'},
        {'name':'N_e','data':N_e_L,'units':'1/m^3','long_name':'Transformed Electron Number Density in Lab Frame'},
        {'name':'Jx_e','data':Jx_e_L,'units':'A/m^2','long_name':'Transformed Electron Current Density Jx in Lab Frame'},
        {'name':'Jy_e','data':Jy_e_L,'units':'A/m^2','long_name':'Transformed Electron Current Density Jy in Lab Frame'},
        {'name':'Jz_e','data':Jz_e_L,'units':'A/m^2','long_name':'Transformed Electron Current Density Jz in Lab Frame'},
    ],
    nc_name='Summarize_Field_L_Frame',
    working_dir=working_dir
)
exit(0)

