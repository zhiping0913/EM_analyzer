import numpy as np
import scipy.constants as C
import os
import math
import matplotlib.pyplot as plt
import xarray as xr
working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface/a0=250/1D/45/Initialize_Field'
os.makedirs(name=working_dir,exist_ok=True)
print(working_dir)

laser_lambda = 0.8*C.micron		# Laser wavelength
laser_f0=1/laser_lambda
laser_k0=2*C.pi*laser_f0
laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_period=laser_lambda/C.speed_of_light
laser_Bc=(C.m_e*laser_omega0)/(C.elementary_charge)
laser_Ec=laser_Bc*C.speed_of_light
vacuum_length_x_lambda=20   #lambda
cells_per_lambda_x =4000

nx=round(2*vacuum_length_x_lambda*cells_per_lambda_x)

x_min=-vacuum_length_x_lambda*laser_lambda
x_max=vacuum_length_x_lambda*laser_lambda

dx=(x_max-x_min)/nx

#xb: Grid x at the boundary of the cell.
#x: Grid x at the center of the cell.
#Ex uses (x+dx/2,y,z)
#Ey uses (x,y+dy/2,z)
#Ez uses (x,y,z+dz/2)
#Bx uses (x,y+dy/2,z+dz/2)
#By uses (x+dx/2,y,z+dz/2)
#Bz uses (x+dx/2,y+dy/2,z)
#number density uses (x,y,z)
x_axis=np.linspace(start=x_min,stop=x_max,num=nx,endpoint=False)+dx/2
xb_axis=x_axis+dx/2
x,=np.meshgrid(x_axis,indexing='ij')
xb,=np.meshgrid(xb_axis,indexing='ij')

Ex=np.zeros(shape=(nx,),dtype=np.float64)
Ey=np.zeros(shape=(nx,),dtype=np.float64)
Ez=np.zeros(shape=(nx,),dtype=np.float64)
Bx=np.zeros(shape=(nx,),dtype=np.float64)
By=np.zeros(shape=(nx,),dtype=np.float64)
Bz=np.zeros(shape=(nx,),dtype=np.float64)



def generate_fields(
    laser_a0=10.0,
    laser_FWHM=25*C.femto,
    laser_phase =0.0,
    theta_degree=0.0,
    center_position=0.0,
    polarisation_angle_degree=0.0,
    direction=1.0,
    ):
    #polarisation_angle here is defined in an opposite direction comparied to the polarisation_angle in EPOCH.
    global Ex,Ey,Ez,Bx,By,Bz
    theta_rad=np.radians(theta_degree)
    polarisation_angle_rad=np.radians(polarisation_angle_degree)
    laser_f0=1/laser_lambda
    laser_k0=2*C.pi*laser_f0
    laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
    laser_amp=laser_a0*(C.m_e*laser_omega0*C.speed_of_light)/(C.elementary_charge)
    laser_tau=laser_FWHM/math.sqrt(2*math.log(2)) 
    laser_tau_M=laser_tau/math.cos(theta_rad)
    laser_k0_M=laser_k0*math.cos(theta_rad)
    laser_amp_M=laser_amp*math.cos(theta_rad)
    
    def transversal_field(x:np.ndarray):
        field=np.cos(np.sign(direction)*laser_k0_M*(x-center_position)+laser_phase)*np.exp(-np.square((x-center_position)/(laser_tau_M*C.speed_of_light)))
        return field

    Ey=Ey+laser_amp_M*transversal_field(x_axis)*math.cos(polarisation_angle_rad)*(x<0)
    Ez=Ez+laser_amp_M*transversal_field(x_axis)*math.sin(polarisation_angle_rad)*(x<0)
    By=By-np.sign(direction)*(laser_amp_M/C.speed_of_light)*transversal_field(xb_axis)*math.sin(polarisation_angle_rad)*(x<0)
    Bz=Bz+np.sign(direction)*(laser_amp_M/C.speed_of_light)*transversal_field(xb_axis)*math.cos(polarisation_angle_rad)*(x<0)
    with open(file=os.path.join(working_dir,'Initialize_Field.txt'),mode='a+',encoding='UTF-8') as txt:
        txt.write(
f"""
Add field.
vacuum_length_x_lambda: {vacuum_length_x_lambda}
cells_per_lambda_x: {cells_per_lambda_x}
laser_lambda: {laser_lambda} (m)
laser_a0: {laser_a0}
laser_FWHM: {laser_FWHM} (s)
laser_phase: {laser_phase} (rad)
theta_degree: {theta_degree} (°)
polarisation_angle_degree: {polarisation_angle_degree} (°)
direction: {direction}
"""
        )


def write_field(field:np.ndarray,x_axis=x_axis,name=''):
    field.tofile(os.path.join(working_dir,name))
    return 0
    field_ds=xr.Dataset(
        data_vars={
            name:(["x"], field),
            },
        coords={'x':(["x"], x_axis)}
        )
    field_ds.to_netcdf(path=os.path.join(working_dir,'%s.nc' %(name)))
    return 0
    





generate_fields(
    laser_a0=250,
    laser_FWHM=8*C.femto,
    laser_phase=C.pi/2,
    theta_degree=45,
    center_position=x_min/2,
    polarisation_angle_degree=0,
    direction=1.0,
    )



#write_field(Ex,xb_axis,'Electric_Field_Ex')
write_field(Ey,x_axis,'Electric_Field_Ey')
#write_field(Ez,x_axis,'Electric_Field_Ez')
#write_field(Bx,x_axis,'Magnetic_Field_Bx')
#write_field(By,xb_axis,'Magnetic_Field_By')
write_field(Bz,xb_axis,'Magnetic_Field_Bz')

plt.plot(x_axis/laser_lambda,Ey/laser_Ec)
plt.xlabel(xlabel='x/λ0')
plt.ylabel(ylabel='Ey/Ec')
plt.title(label='Ey')
plt.savefig(os.path.join(working_dir,'Ey_0000.png'))
plt.clf()



