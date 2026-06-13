import sdf_helper
import pprint
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as C
from scipy.ndimage import rotate,zoom
import os
import math
from scipy import signal
import h5py
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import LogNorm
import pandas as pd
from scipy.signal import hilbert

working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Try_3D/dat'

theta_degree=0
theta_rad=np.radians(theta_degree)

laser_lambda = 0.8*C.micron		# Laser wavelength, microns
laser_k0=2*C.pi/laser_lambda
laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_period=laser_lambda/C.speed_of_light
laser_a0 = 10		# Laser field strength
laser_amp=laser_a0*(C.m_e*laser_omega0*C.speed_of_light)/(C.elementary_charge)
laser_FWHM=5*C.femto   #The full width at half maximum of the intensity.
laser_tau=laser_FWHM/math.sqrt(2*math.log(2)) 
laser_w_0_lambda= 2

highest_harmonic=30

laser_spectrum_peak=C.pi**1.5*laser_amp*(laser_tau*C.speed_of_light)*(laser_w_0_lambda*laser_lambda)**2/2
laser_energy=(C.pi/2)**1.5*laser_amp**2*(laser_tau*C.speed_of_light)*(laser_w_0_lambda*laser_lambda)**2/2

cells_per_lambda =30
vacuum_length_x_lambda=10   #lambda
vacuum_length_y_lambda=5   #lambda
vacuum_length_z_lambda=5   #lambda

continuation_length_lambda=0  #lambda
space_length_lambda=2*(max(vacuum_length_x_lambda,vacuum_length_y_lambda)+continuation_length_lambda)   #lambda
n_field_x=round(2*vacuum_length_x_lambda*cells_per_lambda)
n_field_y=round(2*vacuum_length_y_lambda*cells_per_lambda)
n_field_z=round(2*vacuum_length_z_lambda*cells_per_lambda)
n_continuation=round(space_length_lambda*cells_per_lambda)
n_freq_radius=round(n_continuation*(highest_harmonic/cells_per_lambda))
n_freq_angle=n_freq_radius
print(n_field_x,n_field_y,n_field_z)
highest_harmonic=100

d_x=laser_lambda/cells_per_lambda   #unit: m
d_f=1/(space_length_lambda*laser_lambda)   #unit: 1/m, d_x*d_f=1/n_continuation
d_f_radius=d_f
d_f_angle=2*np.pi/n_freq_angle


grid_x_axis=np.linspace(start=-vacuum_length_x_lambda*laser_lambda,stop=vacuum_length_x_lambda*laser_lambda,num=n_field_x,endpoint=False)
grid_y_axis=np.linspace(start=-vacuum_length_y_lambda*laser_lambda,stop=vacuum_length_y_lambda*laser_lambda,num=n_field_y,endpoint=False)
grid_z_axis=np.linspace(start=-vacuum_length_z_lambda*laser_lambda,stop=vacuum_length_z_lambda*laser_lambda,num=n_field_z,endpoint=False)
grid_center=np.array([np.average(grid_z_axis),np.average(grid_y_axis),np.average(grid_x_axis)])
grid_x_continuation_axis=np.linspace(start=0,stop=space_length_lambda*laser_lambda,num=n_continuation,endpoint=False)
grid_y_continuation_axis=grid_x_continuation_axis
grid_z_continuation_axis=grid_x_continuation_axis

grid_x,grid_y,grid_z=np.meshgrid(grid_x_axis,grid_y_axis,grid_z_axis, indexing='ij')   #grid_x.shape=(n_field_x,n_field_y,n_field_z)



freq_x_axis=np.fft.fftshift(np.fft.fftfreq(n=n_continuation,d=laser_lambda/cells_per_lambda))
freq_y_axis=freq_x_axis
freq_z_axis=freq_x_axis

freq_x,freq_y,freq_z=np.meshgrid(freq_x_axis,freq_y_axis,freq_z_axis, indexing='ij')

freq_continuation_radius = np.sqrt(freq_x**2 + freq_y**2)
freq_continuation_theta = np.arctan2(freq_y,freq_x)

grid_centerline_axis=grid_x_axis
freq_centerline_axis=freq_x_axis 

mask_left = grid_x * np.cos(theta_rad) + grid_y * np.sin(theta_rad) < 0
mask_right = grid_x * np.cos(theta_rad) + grid_y* np.sin(theta_rad) > 0

filter=3.5

def continuation_field(Electric_Field:np.ndarray):
    assert Electric_Field.shape==(n_field_x,n_field_y,n_field_z)
    Electric_Field_continuation=np.zeros(shape=(n_continuation,n_continuation,n_continuation))
    Electric_Field_continuation[round((n_continuation-n_field_x)/2):round((n_continuation+n_field_x)/2),round((n_continuation-n_field_y)/2):round((n_continuation+n_field_y)/2),round((n_continuation-n_field_z)/2):round((n_continuation+n_field_z)/2)]=Electric_Field
    return Electric_Field_continuation

def rotate_field(Electric_Field_Ex:np.ndarray,Electric_Field_Ey:np.ndarray,angle=0.0,name=''):
    assert Electric_Field_Ex.shape==(n_field_x,n_field_y,n_field_z)
    assert Electric_Field_Ey.shape==(n_field_x,n_field_y,n_field_z)
    Electric_Field_transversal=-Electric_Field_Ex*np.sin(angle)+Electric_Field_Ey*np.cos(angle)
    Electric_Field_longitudinal=Electric_Field_Ex*np.cos(angle)+Electric_Field_Ey*np.sin(angle)
    Electric_Field_transversal_rotate=rotate(input=Electric_Field_transversal,angle=np.rad2deg(angle),axes=(1,0),reshape=False,mode='constant',cval=0,order=3)
    Electric_Field_longitudinal_rotate=rotate(input=Electric_Field_longitudinal,angle=np.rad2deg(angle),axes=(1,0),reshape=False,mode='constant',cval=0,order=3)
    fig,ax = plt.subplots()
    pcm=ax.pcolormesh(grid_x_axis/laser_lambda,grid_y_axis/laser_lambda,Electric_Field_transversal_rotate/laser_amp,cmap='RdBu',vmax=1.2,vmin=-1.2)
    ax.set_aspect('equal')
    plt.colorbar(pcm).ax.set_ylabel('Ey(E0)')
    plt.xlabel('x (λ)')
    plt.ylabel('y (λ)')
    plt.savefig(os.path.join(working_dir,'Electric_Field_transversal_rotate_%s.png' %(name)))
    plt.close(fig)
    plt.clf()

    fig,ax = plt.subplots()
    pcm=ax.pcolormesh(grid_x_axis/laser_lambda,grid_y_axis/laser_lambda,Electric_Field_longitudinal_rotate/laser_amp,cmap='RdBu',vmax=1.2,vmin=-1.2)
    ax.set_aspect('equal')
    plt.colorbar(pcm).ax.set_ylabel('Ey(E0)')
    plt.xlabel('x (λ)')
    plt.ylabel('y (λ)')
    plt.savefig(os.path.join(working_dir,'Electric_Field_longitudinal_rotate_%s.png' %(name)))
    plt.close(fig)
    plt.clf()
    return Electric_Field_transversal_rotate, Electric_Field_longitudinal_rotate

def get_polarisation(Electric_Field_Ey:np.ndarray,Electric_Field_Ez:np.ndarray,direction=1,name=''):
    """
        direction. direction>0 means the field travels in +x direction. direction<0 means the field travels in -x direction
    """
    assert Electric_Field_Ey.shape==(n_field_x,n_field_y,n_field_z)
    assert Electric_Field_Ez.shape==(n_field_x,n_field_y,n_field_z)
    assert direction!=0
    Electric_Field_Ey_energy=np.sum(np.square(Electric_Field_Ey))*d_x*d_x*d_x
    Electric_Field_Ez_energy=np.sum(np.square(Electric_Field_Ez))*d_x*d_x*d_x
    print('Energy in y polarisation: %f theoretical total energy' %(Electric_Field_Ey_energy/laser_energy))
    print('Energy in z polarisation: %f theoretical total energy' %(Electric_Field_Ez_energy/laser_energy))
    print('arctan(√(Energy_z/Energy_y))=%f×π' %(np.atan2(np.sqrt(Electric_Field_Ez_energy),np.sqrt(Electric_Field_Ey_energy))/C.pi))
    




def get_spectrum(Electric_Field_Ex:np.ndarray,Electric_Field_Ey:np.ndarray,Electric_Field_Ez:np.ndarray,angle=0.0,name=''):
    assert Electric_Field_Ex.shape==(n_field_x,n_field_y,n_field_z)
    assert Electric_Field_Ey.shape==(n_field_x,n_field_y,n_field_z)
    assert Electric_Field_Ez.shape==(n_field_x,n_field_y,n_field_z)
    Electric_Field_Ex_continuation=continuation_field(Electric_Field_Ex)
    Electric_Field_Ey_continuation=continuation_field(Electric_Field_Ey)
    Electric_Field_Ez_continuation=continuation_field(Electric_Field_Ez)
    Electric_Field_transversal=-Electric_Field_Ex*np.sin(angle)+Electric_Field_Ey*np.cos(angle)
    Electric_Field_Ex_interpolator=RegularGridInterpolator(points=(grid_x_axis,grid_y_axis,grid_z_axis),values=Electric_Field_Ex,method='linear',bounds_error=False,fill_value=0)
    Electric_Field_Ey_interpolator=RegularGridInterpolator(points=(grid_x_axis,grid_y_axis,grid_z_axis),values=Electric_Field_Ey,method='linear',bounds_error=False,fill_value=0)
    Electric_Field_centerline_points=np.vstack((grid_centerline_axis*np.cos(angle),grid_centerline_axis*np.sin(angle),grid_centerline_axis*0)).T+np.tile(grid_center,(len(grid_centerline_axis),1))
    Electric_Field_Ex_centerline=Electric_Field_Ex_interpolator(Electric_Field_centerline_points)
    Electric_Field_Ey_centerline=Electric_Field_Ey_interpolator(Electric_Field_centerline_points)
    Electric_Field_centerline=-Electric_Field_Ex_centerline*np.sin(angle)+Electric_Field_Ey_centerline*np.cos(angle)




    Electric_Field_Ex_spectrum=np.fft.fftshift(np.fft.fftn(Electric_Field_Ex_continuation))*d_x*d_x*d_x
    Electric_Field_Ex_spectrum_squere=np.square(np.abs(Electric_Field_Ex_spectrum))
    Electric_Field_Ey_spectrum=np.fft.fftshift(np.fft.fftn(Electric_Field_Ey_continuation))*d_x*d_x*d_x
    Electric_Field_Ey_spectrum_squere=np.square(np.abs(Electric_Field_Ey_spectrum))
    Electric_Field_Ez_spectrum=np.fft.fftshift(np.fft.fftn(Electric_Field_Ez_continuation))*d_x*d_x*d_x
    Electric_Field_Ez_spectrum_squere=np.square(np.abs(Electric_Field_Ez_spectrum))
    Electric_Field_spectrum_squere=Electric_Field_Ex_spectrum_squere+Electric_Field_Ey_spectrum_squere+Electric_Field_Ez_spectrum_squere
    
    Electric_Field_spectrum_squere_interpolator=RegularGridInterpolator(points=(freq_x_axis,freq_y_axis,freq_z_axis),values=Electric_Field_spectrum_squere,method='linear',bounds_error=False,fill_value=0)
    Electric_Field_spectrum_centerline_points=np.vstack((freq_centerline_axis*np.cos(angle),freq_centerline_axis*np.sin(angle),freq_centerline_axis*0)).T
    Electric_Field_spectrum_squere_centerline=Electric_Field_spectrum_squere_interpolator(Electric_Field_spectrum_centerline_points)

    print(freq_centerline_axis[np.where(Electric_Field_spectrum_squere_centerline==np.max(Electric_Field_spectrum_squere_centerline))[0]]*laser_lambda)
    print(np.max(np.abs(Electric_Field_centerline))/laser_amp)
    print(np.max(Electric_Field_spectrum_squere)/laser_spectrum_peak**2)
    print((np.sum(np.square(np.abs(Electric_Field_Ex)))+np.sum(np.square(np.abs(Electric_Field_Ey)))+np.sum(np.square(np.abs(Electric_Field_Ez))))*d_x*d_x*d_x/laser_energy)
    print(np.sum(np.square(np.abs(Electric_Field_Ex)))*d_x*d_x*d_x/laser_energy)
    print(np.sum(np.square(np.abs(Electric_Field_Ey)))*d_x*d_x*d_x/laser_energy)
    print(np.sum(np.square(np.abs(Electric_Field_Ez)))*d_x*d_x*d_x/laser_energy)
    print(np.sum(Electric_Field_spectrum_squere)*d_f*d_f*d_f/laser_energy)
    print(np.sum(Electric_Field_spectrum_squere*(freq_continuation_radius*laser_lambda>filter))*d_f*d_f*d_f/laser_energy)
    print(np.max(Electric_Field_spectrum_squere_centerline)/laser_spectrum_peak**2)
        
    norm = LogNorm(vmin=1e-6, vmax=1)
    
    fig,ax = plt.subplots()
    plt.plot(grid_centerline_axis/laser_lambda,Electric_Field_centerline/laser_amp)
    plt.ylim(-1.2,1.2)
    plt.xlabel('x (λ)')
    plt.ylabel('E (E0)')
    plt.savefig(os.path.join(working_dir,'Electric_Field_centerline_%s.png' %(name)))
    plt.close(fig)
    plt.clf()

    fig,ax = plt.subplots()
    pcm=ax.pcolormesh(grid_x_axis/laser_lambda,grid_y_axis/laser_lambda,Electric_Field_transversal[:,:,round(n_continuation/2)].T/laser_amp,cmap='RdBu',vmax=1.2,vmin=-1.2)
    plt.colorbar(pcm).ax.set_ylabel('Ey(E0)')
    plt.xlabel('x (λ)')
    plt.ylabel('y (λ)')
    plt.savefig(os.path.join(working_dir,'Electric_Field_transversal_xy_%s.png' %(name)))
    plt.close(fig)
    plt.clf()



    plt.semilogy(freq_centerline_axis*laser_lambda,Electric_Field_spectrum_squere_centerline/laser_spectrum_peak**2)
    plt.xlabel(xlabel='k/k0')
    plt.ylabel(ylabel='I(k)/I0')
    plt.title('%s spectrum' %(name))
    plt.xlim(0,30)
    plt.ylim(1e-6,1)
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_squere_centerline_%s.png' %(name)))
    plt.clf()
    

    fig,ax = plt.subplots()
    pcm=ax.pcolormesh(freq_x_axis*laser_lambda,freq_y_axis*laser_lambda,Electric_Field_spectrum_squere[:,:,round(n_continuation/2)].T/laser_spectrum_peak**2,cmap='hot', norm=norm)
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_xlabel('kx(k0)')
    ax.set_ylabel('ky(k0)')
    plt.colorbar(pcm).ax.set_ylabel('%s spectrum' %(name))
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_squere_xy_%s.png' %(name)))
    plt.close(fig)
    plt.clf()

def get_envelope(Electric_Field:np.ndarray,name=''):
    """
        Assume the wave is in z direction. If not, please rotate the field first.
    """
    assert Electric_Field.shape==(n_field_x,n_field_y,n_field_z)
    Electric_Field_continuation=continuation_field(Electric_Field)
    Electric_Field_analytic=hilbert(Electric_Field_continuation,axis=2)[round((n_continuation-n_field_x)/2):round((n_continuation+n_field_x)/2),round((n_continuation-n_field_y)/2):round((n_continuation+n_field_y)/2),round((n_continuation-n_field_z)/2):round((n_continuation+n_field_z)/2)]
    Electric_Field_analytic_phase=np.angle(Electric_Field_analytic)
    Electric_Field_envelope=np.abs(Electric_Field_analytic)
    Electric_Field_envelope_max=np.max(Electric_Field_envelope)
    Electric_Field_envelope_max_id=tuple(np.array(np.where(Electric_Field_envelope==Electric_Field_envelope_max))[:,0])   #Electric_Field_envelope_max_id=(x_id,y_id,z_id)
    print(Electric_Field_envelope_max/laser_amp)
    print(Electric_Field_envelope_max_id)
    Electric_Field_envelope_max_phase=Electric_Field_analytic_phase[Electric_Field_envelope_max_id]
    print('Phase at the peak: %fπ'%(Electric_Field_envelope_max_phase/np.pi))
    plt.plot(grid_x_axis/laser_lambda,Electric_Field[:,Electric_Field_envelope_max_id[1],Electric_Field_envelope_max_id[2]]/laser_amp,c='b',label='field',linewidth=2)
    plt.plot(grid_x_axis/laser_lambda,Electric_Field_envelope[:,Electric_Field_envelope_max_id[1],Electric_Field_envelope_max_id[2]]/laser_amp,linestyle='--',c='g',label='envelope',linewidth=1)
    plt.plot(grid_x_axis/laser_lambda,-Electric_Field_envelope[:,Electric_Field_envelope_max_id[1],Electric_Field_envelope_max_id[2]]/laser_amp,linestyle='--',c='g',linewidth=1)
    plt.ylim(-1,1)
    plt.legend()
    plt.xlabel(xlabel='x/λ0')
    plt.ylabel(ylabel='E/E0')
    plt.title(label='Field at the centerline %s' %(name))
    plt.savefig(os.path.join(working_dir,'Electric_Field_envelope_centerline_%s.png' %(name)))
    plt.clf()
    return Electric_Field_envelope,Electric_Field_envelope_max,Electric_Field_envelope_max_id,Electric_Field_envelope_max_phase


for i in range(21):
    sdf=sdf_helper.getdata(fname=os.path.join(working_dir,'%0.4d.sdf' %(i)))
    Electric_Field_Ex=sdf.__dict__['Electric_Field_Ex'].data
    Electric_Field_Ey=sdf.__dict__['Electric_Field_Ey'].data
    Electric_Field_Ez=sdf.__dict__['Electric_Field_Ez'].data
    get_spectrum(Electric_Field_Ex,Electric_Field_Ey,Electric_Field_Ez,name='%0.4d' %(i))
    get_envelope(Electric_Field_Ey,name='%0.4d' %(i))


'incident'
'reflection'
'transmission'