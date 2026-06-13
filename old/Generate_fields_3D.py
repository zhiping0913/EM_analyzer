import numpy as np
import scipy.constants as C
import os
import math

laser_lambda = 0.8*C.micron		# Laser wavelength, microns
vacuum_length_x_lambda=5   #lambda
vacuum_length_y_lambda=5   #lambda
vacuum_length_z_lambda=5   #lambda
cells_per_lambda =50

nx=round(2*vacuum_length_x_lambda*cells_per_lambda)
ny=round(2*vacuum_length_y_lambda*cells_per_lambda)
nz=round(2*vacuum_length_z_lambda*cells_per_lambda)

x_min=-vacuum_length_x_lambda*laser_lambda
x_max=vacuum_length_x_lambda*laser_lambda
y_min=-vacuum_length_y_lambda*laser_lambda
y_max=vacuum_length_y_lambda*laser_lambda
z_min=-vacuum_length_z_lambda*laser_lambda
z_max=vacuum_length_z_lambda*laser_lambda

dx=(x_max-x_min)/nx
dy=(y_max-y_min)/ny
dz=(z_max-z_min)/nz

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
y_axis=np.linspace(start=y_min,stop=y_max,num=ny,endpoint=False)+dy/2
z_axis=np.linspace(start=z_min,stop=z_max,num=nz,endpoint=False)+dz/2
xb_axis=x_axis+dx/2
yb_axis=y_axis+dy/2
zb_axis=z_axis+dz/2
x,y,z=np.meshgrid(x_axis,y_axis,z_axis,indexing='ij')
xb,yb,zb=np.meshgrid(xb_axis,yb_axis,zb_axis,indexing='ij')

E_x=np.zeros(shape=(nx,ny,nz),dtype=np.float64)
E_y=np.zeros(shape=(nx,ny,nz),dtype=np.float64)
E_z=np.zeros(shape=(nx,ny,nz),dtype=np.float64)
B_x=np.zeros(shape=(nx,ny,nz),dtype=np.float64)
B_y=np.zeros(shape=(nx,ny,nz),dtype=np.float64)
B_z=np.zeros(shape=(nx,ny,nz),dtype=np.float64)

def generate_fields(laser_a0=10.0,laser_lambda = laser_lambda,laser_FWHM=25*C.femto,laser_w0=3*C.micro,laser_phase =0.0,theta_degree=0.0,center_position=0.0,polarisation_angle_degree=0.0):
    """

    Args:
        laser_a0
        laser_lambda: .unit: m
        laser_FWHM: Intensity FWHM. Unit: s.
        laser_w0: waist radius. Unit: m.
        laser_phase
        theta_degree
        center_position: The distance between the center of the pulse and the origin (0,0,0). Unit: m
        polarisation_angle_degree: The polarisation angle. Suppose the wave is in +x direction, polarisation angle 0° means E in +y direction, and polarisation angle 0° means E in +z direction. Unit: °.
    """
    global E_x,E_y,E_z,B_x,B_y,B_z
    theta_rad=np.radians(theta_degree)
    polarisation_angle_rad=np.radians(polarisation_angle_degree)
    time=center_position/C.speed_of_light
    laser_tau=laser_FWHM/math.sqrt(2*math.log(2)) 
    laser_k=2*np.pi/laser_lambda   #unit: 1/m
    laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
    laser_amp=laser_a0*(C.m_e*laser_omega0*C.speed_of_light)/(C.elementary_charge)
    laser_zR=np.pi*laser_w0**2/laser_lambda  #unit: m
    laser_divergence=laser_w0/laser_zR
    def transversal_field(xx:np.ndarray,yy:np.ndarray,zz:np.ndarray):
        x_rot=xx*np.cos(theta_rad)+yy*np.sin(theta_rad)
        y_rot=-xx*np.sin(theta_rad)+yy*np.cos(theta_rad)
        y_pol=y_rot*np.cos(polarisation_angle_rad)+zz*np.sin(polarisation_angle_rad)   #E field in y_pol direction
        z_pol=-y_rot*np.sin(polarisation_angle_rad)+zz*np.cos(polarisation_angle_rad)   #B field in z_pol direction
        r=np.sqrt(np.square(y_rot)+np.square(zz))
        w_Z=laser_w0*np.sqrt(1+np.square(x_rot/laser_zR))
        Kappa_Z=x_rot/(np.square(x_rot)+np.square(laser_zR))   #κ=1/R is the curvature of the wavefront. unit: 1/m
        phi=laser_k*(x_rot+Kappa_Z*np.square(r)/2)-laser_omega0*time
        Gouy_Z=np.arctan(x_rot/laser_zR)
        field=(laser_w0/w_Z)*np.exp(-np.square(r/w_Z))*np.exp(-np.square(phi/(laser_omega0*laser_tau)))*np.cos(phi-Gouy_Z+laser_phase)
        return field
    def longitudinal_E_field(xx:np.ndarray,yy:np.ndarray,zz:np.ndarray):
        x_rot=xx*np.cos(theta_rad)+yy*np.sin(theta_rad)
        y_rot=-xx*np.sin(theta_rad)+yy*np.cos(theta_rad)
        y_pol=y_rot*np.cos(polarisation_angle_rad)+zz*np.sin(polarisation_angle_rad)   #E field in y_pol direction
        z_pol=-y_rot*np.sin(polarisation_angle_rad)+zz*np.cos(polarisation_angle_rad)   #B field in z_pol direction
        r=np.sqrt(np.square(y_rot)+np.square(zz))
        w_Z=laser_w0*np.sqrt(1+np.square(x_rot/laser_zR))
        Kappa_Z=x_rot/(np.square(x_rot)+np.square(laser_zR))   #κ=1/R is the curvature of the wavefront. unit: 1/m
        phi=laser_k*(x_rot+Kappa_Z*np.square(r)/2)-laser_omega0*time
        Gouy_Z=np.arctan(x_rot/laser_zR)
        field=((laser_lambda)/(np.pi*w_Z))*(y_pol/w_Z)*np.exp(-np.square(r/w_Z))*np.exp(-np.square(phi/(laser_omega0*laser_tau)))*np.sin(phi-2*Gouy_Z+laser_phase)
        return field
    def longitudinal_B_field(xx:np.ndarray,yy:np.ndarray,zz:np.ndarray):
        x_rot=xx*np.cos(theta_rad)+yy*np.sin(theta_rad)
        y_rot=-xx*np.sin(theta_rad)+yy*np.cos(theta_rad)
        y_pol=y_rot*np.cos(polarisation_angle_rad)+zz*np.sin(polarisation_angle_rad)   #E field in y_pol direction
        z_pol=-y_rot*np.sin(polarisation_angle_rad)+zz*np.cos(polarisation_angle_rad)   #B field in z_pol direction
        r=np.sqrt(np.square(y_rot)+np.square(zz))
        w_Z=laser_w0*np.sqrt(1+np.square(x_rot/laser_zR))
        Kappa_Z=x_rot/(np.square(x_rot)+np.square(laser_zR))   #κ=1/R is the curvature of the wavefront. unit: 1/m
        phi=laser_k*(x_rot+Kappa_Z*np.square(r)/2)-laser_omega0*time
        Gouy_Z=np.arctan(x_rot/laser_zR)
        field=((laser_lambda)/(np.pi*w_Z))*(z_pol/w_Z)*np.exp(-np.square(r/w_Z))*np.exp(-np.square(phi/(laser_omega0*laser_tau)))*np.sin(phi-2*Gouy_Z+laser_phase)
        return field
    E_x=E_x+laser_amp*(longitudinal_E_field(xb,y,z)*np.cos(theta_rad)-transversal_field(xb,y,z)*np.cos(polarisation_angle_rad)*np.sin(theta_rad))
    E_y=E_y+laser_amp*(longitudinal_E_field(x,yb,z)*np.sin(theta_rad)+transversal_field(x,yb,z)*np.cos(polarisation_angle_rad)*np.cos(theta_rad))
    E_z=E_z+laser_amp*transversal_field(x,y,zb)*np.sin(polarisation_angle_rad)
    B_x=B_x+(laser_amp/C.speed_of_light)*(longitudinal_B_field(x,yb,zb)*np.cos(theta_rad)+transversal_field(x,yb,zb)*np.sin(polarisation_angle_rad)*np.sin(theta_rad))
    B_y=B_y+(laser_amp/C.speed_of_light)*(longitudinal_B_field(xb,y,zb)*np.sin(theta_rad)-transversal_field(xb,y,zb)*np.sin(polarisation_angle_rad)*np.cos(theta_rad))
    B_z=B_z+(laser_amp/C.speed_of_light)*transversal_field(xb,yb,z)*np.cos(polarisation_angle_rad)
    print(np.max(np.abs(np.gradient(E_x,xb_axis,axis=0)*dx+np.gradient(E_y,yb_axis,axis=1)*dy+np.gradient(E_z,zb_axis,axis=2)*dz))/laser_amp)
    print(np.max(np.abs(np.gradient(B_x,x_axis,axis=0)*dx+np.gradient(B_y,y_axis,axis=1)*dy+np.gradient(B_z,z_axis,axis=2)*dz))/(laser_amp/C.speed_of_light))
    print(np.max(np.abs(np.gradient(B_y,xb_axis,axis=0)*dx-np.gradient(B_x,yb_axis,axis=1)*dy))/(laser_amp/C.speed_of_light))
working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Try_3D/dat'

generate_fields(
    laser_a0=10.0,
    laser_lambda = laser_lambda,
    laser_FWHM=5*C.femto,
    laser_w0=2*laser_lambda,
    laser_phase =np.pi/2,
    theta_degree=0.0,
    center_position=0,
    polarisation_angle_degree=0
    )





with open(file=os.path.join(working_dir,'E_x'),mode='wb') as f:
    E_x.transpose(2,1,0).tofile(f)
with open(file=os.path.join(working_dir,'E_y'),mode='wb') as f:
    E_y.transpose(2,1,0).tofile(f)
with open(file=os.path.join(working_dir,'E_z'),mode='wb') as f:
    E_z.transpose(2,1,0).tofile(f)
with open(file=os.path.join(working_dir,'B_x'),mode='wb') as f:
    B_x.transpose(2,1,0).tofile(f)
with open(file=os.path.join(working_dir,'B_y'),mode='wb') as f:
    B_y.transpose(2,1,0).tofile(f)
with open(file=os.path.join(working_dir,'B_z'),mode='wb') as f:
    B_z.transpose(2,1,0).tofile(f)


