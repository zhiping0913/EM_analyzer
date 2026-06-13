import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/start')
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy.constants as C
import os
import math
import xarray as xr
import matplotlib.pyplot as plt


working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Gaussian_beam_pulse'
os.makedirs(name=working_dir,exist_ok=True)
print(working_dir)
laser_lambda = 0.8*C.micron		# Laser wavelength
laser_f0=1/laser_lambda
laser_k0=2*C.pi*laser_f0
laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_period=laser_lambda/C.speed_of_light
laser_Bc=(C.m_e*laser_omega0)/(C.elementary_charge)
laser_Ec=laser_Bc*C.speed_of_light


vacuum_length_x_lambda=40   #lambda
vacuum_length_y_lambda=20   #lambda
cells_per_lambda_x =200
cells_per_lambda_y =200

nx=round(2*vacuum_length_x_lambda*cells_per_lambda_x)
ny=round(2*vacuum_length_y_lambda*cells_per_lambda_y)

x_min=-vacuum_length_x_lambda*laser_lambda
x_max=vacuum_length_x_lambda*laser_lambda
y_min=-vacuum_length_y_lambda*laser_lambda
y_max=vacuum_length_y_lambda*laser_lambda

dx=(x_max-x_min)/nx
dy=(y_max-y_min)/ny

#xb: Grid x at the boundary of the cell.
#x: Grid x at the center of the cell.
#Ex uses (xb,y)
#Ey uses (x,yb)
#Ez uses (x,y)
#Bx uses (x,yb)
#By uses (xb,y)
#Bz uses (xb,yb)
x_axis=jnp.linspace(start=x_min,stop=x_max,num=nx,endpoint=False)+dx/2
y_axis=jnp.linspace(start=y_min,stop=y_max,num=ny,endpoint=False)+dy/2
xb_axis=x_axis+dx/2
yb_axis=y_axis+dy/2
x,y=jnp.meshgrid(x_axis,y_axis,indexing='ij')
xb,yb=jnp.meshgrid(xb_axis,yb_axis,indexing='ij')


Ex=jnp.zeros(shape=(nx,ny),dtype=jnp.float64)
Ey=jnp.zeros(shape=(nx,ny),dtype=jnp.float64)
Ez=jnp.zeros(shape=(nx,ny),dtype=jnp.float64)
Bx=jnp.zeros(shape=(nx,ny),dtype=jnp.float64)
By=jnp.zeros(shape=(nx,ny),dtype=jnp.float64)
Bz=jnp.zeros(shape=(nx,ny),dtype=jnp.float64)






def generate_fields(
    laser_a0=10.0,
    laser_FWHM=8*C.femto,
    laser_w0=3*C.micro,
    laser_phase =0.0,
    theta_degree=0.0,
    center_position=0.0,
    polarisation_angle_degree=0.0,
    ):
    """
    Args:
        laser_a0: Normalized laser field strength. 
        laser_FWHM: Intensity FWHM. Unit: s.
        laser_w0: Waist radius. Unit: m.
        laser_phase: Phase at the peak of the envelope measured at the focus. cosφ. Unit: rad.
        theta_degree: Incident angle. Unit: °.
        center_position: The distance between the center of the pulse and the origin (0,0,0). Unit: m
        polarisation_angle_degree: The polarisation angle. Suppose the wave is in +x direction, polarisation angle 0° means E is in +y direction, and polarisation angle 90° means E is in +z direction. Unit: °.
    """
    global Ex,Ey,Ez,Bx,By,Bz
    theta_rad=jnp.radians(theta_degree)
    polarisation_angle_rad=jnp.radians(polarisation_angle_degree)
    laser_tau=laser_FWHM/math.sqrt(2*math.log(2)) 
    laser_k=2*jnp.pi/laser_lambda   #unit: 1/m
    laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
    laser_amp=laser_a0*(C.m_e*laser_omega0*C.speed_of_light)/(C.elementary_charge)
    laser_zR=jnp.pi*laser_w0**2/laser_lambda  #unit: m
    @jax.jit
    def transversal_field(xx:jnp.ndarray,yy:jnp.ndarray):
        x_rot=xx*jnp.cos(theta_rad)+yy*jnp.sin(theta_rad)
        y_rot=-xx*jnp.sin(theta_rad)+yy*jnp.cos(theta_rad)
        w_Z=laser_w0*jnp.sqrt(1+jnp.square(x_rot/laser_zR))
        Kappa_Z=x_rot/(jnp.square(x_rot)+jnp.square(laser_zR))   #κ=1/R is the curvature of the wavefront. unit: 1/m
        phi=laser_k*(x_rot-center_position+Kappa_Z*jnp.square(y_rot)/2)
        Gouy_Z=jnp.arctan(x_rot/laser_zR)
        field=jnp.power(1+jnp.square(x_rot/laser_zR),-1/4)*jnp.exp(-jnp.square(y_rot/w_Z))*jnp.exp(-jnp.square(phi/(laser_omega0*laser_tau)))*jnp.cos(phi-0.5*Gouy_Z+laser_phase)
        return field
    @jax.jit
    def longitudinal_E_field(xx:jnp.ndarray,yy:jnp.ndarray):
        x_rot=xx*jnp.cos(theta_rad)+yy*jnp.sin(theta_rad)
        y_rot=-xx*jnp.sin(theta_rad)+yy*jnp.cos(theta_rad)
        w_Z=laser_w0*jnp.sqrt(1+jnp.square(x_rot/laser_zR))
        Kappa_Z=x_rot/(jnp.square(x_rot)+jnp.square(laser_zR))   #κ=1/R is the curvature of the wavefront. unit: 1/m
        phi=laser_k*(x_rot-center_position+Kappa_Z*jnp.square(y_rot)/2)
        Gouy_Z=jnp.arctan(x_rot/laser_zR)
        field=jnp.power(1+jnp.square(x_rot/laser_zR),-3/4)*(y_rot*jnp.cos(polarisation_angle_rad)/laser_zR)*jnp.exp(-jnp.square(y_rot/w_Z))*jnp.exp(-jnp.square(phi/(laser_omega0*laser_tau)))*jnp.cos(phi-1.5*Gouy_Z-jnp.pi/2+laser_phase)
        return field
    @jax.jit
    def longitudinal_B_field(xx:jnp.ndarray,yy:jnp.ndarray):
        x_rot=xx*jnp.cos(theta_rad)+yy*jnp.sin(theta_rad)
        y_rot=-xx*jnp.sin(theta_rad)+yy*jnp.cos(theta_rad)
        w_Z=laser_w0*jnp.sqrt(1+jnp.square(x_rot/laser_zR))
        Kappa_Z=x_rot/(jnp.square(x_rot)+jnp.square(laser_zR))   #κ=1/R is the curvature of the wavefront. unit: 1/m
        phi=laser_k*(x_rot-center_position+Kappa_Z*jnp.square(y_rot)/2)
        Gouy_Z=jnp.arctan(x_rot/laser_zR)
        field=jnp.power(1+jnp.square(x_rot/laser_zR),-3/4)*(-y_rot*jnp.sin(polarisation_angle_rad)/laser_zR)*jnp.exp(-jnp.square(y_rot/w_Z))*jnp.exp(-jnp.square(phi/(laser_omega0*laser_tau)))*jnp.cos(phi-1.5*Gouy_Z-jnp.pi/2+laser_phase)
        return field
    Ex=Ex+laser_amp*(longitudinal_E_field(xb,y)*jnp.cos(theta_rad)-transversal_field(xb,y)*jnp.cos(polarisation_angle_rad)*jnp.sin(theta_rad))
    Ey=Ey+laser_amp*(longitudinal_E_field(x,yb)*jnp.sin(theta_rad)+transversal_field(x,yb)*jnp.cos(polarisation_angle_rad)*jnp.cos(theta_rad))
    Ez=Ez+laser_amp*transversal_field(x,y)*jnp.sin(polarisation_angle_rad)
    Bx=Bx+(laser_amp/C.speed_of_light)*(longitudinal_B_field(x,yb)*jnp.cos(theta_rad)+transversal_field(x,yb)*jnp.sin(polarisation_angle_rad)*jnp.sin(theta_rad))
    By=By+(laser_amp/C.speed_of_light)*(longitudinal_B_field(xb,y)*jnp.sin(theta_rad)-transversal_field(xb,y)*jnp.sin(polarisation_angle_rad)*jnp.cos(theta_rad))
    Bz=Bz+(laser_amp/C.speed_of_light)*transversal_field(xb,yb)*jnp.cos(polarisation_angle_rad)
    print(jnp.max(jnp.abs(jnp.gradient(Ex,xb_axis,axis=0)*dx+jnp.gradient(Ey,yb_axis,axis=1)*dy))/laser_amp)   #∂Ex/∂x+∂Ey/∂y
    print(jnp.max(jnp.abs(jnp.gradient(Bx,x_axis,axis=0)*dx+jnp.gradient(By,y_axis,axis=1)*dy))/(laser_amp/C.speed_of_light))   #∂Bx/∂x+∂By/∂y
    print(jnp.max(jnp.abs(jnp.gradient(By,xb_axis,axis=0)*dx-jnp.gradient(Bx,yb_axis,axis=1)*dy))/(laser_amp/C.speed_of_light))
    Initialize_Field=f"""
Add field.
vacuum_length_x_lambda: {vacuum_length_x_lambda}
vacuum_length_y_lambda: {vacuum_length_y_lambda}
cells_per_lambda_x: {cells_per_lambda_x}
cells_per_lambda_y: {cells_per_lambda_y}
laser_lambda: {laser_lambda} (m)
laser_a0: {laser_a0}
laser_FWHM: {laser_FWHM} (s)
laser_w0: {laser_w0} (m)
laser_phase: {laser_phase} (rad)
theta_degree: {theta_degree} (°)
polarisation_angle_degree: {polarisation_angle_degree} (°)
"""
    print(Initialize_Field)
    with open(file=os.path.join(working_dir,'Initialize_Field.txt'),mode='a+',encoding='UTF-8') as txt:
        txt.write(Initialize_Field)
    return Ex,Ey,Ez,Bx,By,Bz

def write_field(field:jnp.ndarray,x_axis=x_axis,y_axis=y_axis,name=''):
    field_ds=xr.Dataset(
        data_vars={
            name:(["x", "y"], field),
            },
        coords={'x':(["x"], x_axis),'y':(["y"], y_axis)}
        )
    field_ds.to_netcdf(path=os.path.join(working_dir,'%s.nc' %(name)),format="NETCDF4", engine='h5netcdf')
    print(os.path.join(working_dir,'%s.nc' %(name)))
    return 0
    np.asarray(field.transpose(1,0),dtype=np.float64).tofile(os.path.join(working_dir,name))
    print(os.path.join(working_dir,name))
    return 0









generate_fields(
    laser_a0=1,
    laser_FWHM=15*C.femto,
    laser_w0=1*laser_lambda,
    theta_degree=0,
    laser_phase=jnp.pi/2,
    center_position=0,
)
K=-0.005
target_curvature=K/laser_lambda
mask=x-(target_curvature/2)*jnp.square(y)<0
#Ex=smooth_edge_2D(Ex,mask,edge_length=cells_per_lambda_x)
#Ey=smooth_edge_2D(Ey,mask,edge_length=cells_per_lambda_x)
#Ez=smooth_edge_2D(Ez,mask,edge_length=cells_per_lambda_x)
#Bx=smooth_edge_2D(Bx,mask,edge_length=cells_per_lambda_x)
#By=smooth_edge_2D(By,mask,edge_length=cells_per_lambda_x)
#Bz=smooth_edge_2D(Bz,mask,edge_length=cells_per_lambda_x)

#write_field_2D(Field_list=[Ex],x_axis=xb_axis,y_axis=y_axis,name_list=['Ex'],nc_name=os.path.join(working_dir,'Ex.nc'))
write_field(field=Ex,x_axis=xb_axis,y_axis=y_axis,name='Ex')
write_field(field=Ey,x_axis=x_axis,y_axis=yb_axis,name='Ey')
#write_field(field=Ez,x_axis=x_axis,y_axis=y_axis,name='Ez')
#write_field(field=Bx,x_axis=x_axis,y_axis=yb_axis,name='Bx')
#write_field(field=By,x_axis=xb_axis,y_axis=y_axis,name='By')
write_field(field=Bz,x_axis=xb_axis,y_axis=yb_axis,name='Bz')
exit(0)


fig,ax = plt.subplots()
ax.plot(x_axis/laser_lambda,Ey[:,ny//2]/laser_Ec,label='y=0')
ax.set_xlabel('x/λ0')
ax.set_ylabel('Ey/Ec')
ax.set_title('Initial Ey')
plt.savefig(os.path.join(working_dir,'Ey_0000_y=0.png'))
print(os.path.join(working_dir,'Ey_0000_y=0.png'))
plt.close(fig)
plt.clf()
fig,ax = plt.subplots()
ax.plot(yb_axis/laser_lambda,Ey[nx//2,:]/laser_Ec,label='x=0')
ax.set_xlabel('y/λ0')
ax.set_ylabel('Ey/Ec')
ax.set_title('Initial Ey')
plt.savefig(os.path.join(working_dir,'Ey_0000_x=0.png'))
print(os.path.join(working_dir,'Ey_0000_x=0.png'))
plt.close(fig)
plt.clf()

fig,ax = plt.subplots()
pcm=ax.pcolormesh(x_axis/laser_lambda,yb_axis/laser_lambda,Ey.T/laser_Ec,cmap='RdBu')
ax.set_aspect('equal')
ax.set_xlabel('x/λ0')
ax.set_ylabel('y/λ0')
ax.set_title('Initial Ey')
plt.colorbar(pcm).ax.set_ylabel('Ey/Ec')
plt.savefig(os.path.join(working_dir,'Ey_0000.png'))
print(os.path.join(working_dir,'Ey_0000.png'))
plt.close(fig)
plt.clf()
exit(0)