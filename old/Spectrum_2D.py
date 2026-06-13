import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/EM_analyzer')
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/EM_analyzer/plot')
import os
from typing import Optional
from joblib import Parallel, delayed
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from scipy.integrate import cumulative_simpson
from jax.scipy.ndimage import map_coordinates
from jax.scipy.special import erf
from jax.numpy.fft import fftfreq,fftshift,ifftshift,fft2,ifft2
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm,Normalize
from matplotlib.axes import Axes
from matplotlib.scale import ScaleBase
import scipy.constants as C
from scipy.signal import peak_widths, hilbert
from scipy.ndimage import zoom
import pandas as pd
import cv2
import gc
from EM_analyzer.read_write import read_sdf,read_nc,read_dat
from EM_analyzer.pretreat_fields import print_array_size
from old.pretreat_fields_2D import square_integral_field_2D, continue_field_2D, write_field_2D, smooth_edge_2D
from EM_analyzer.plot.plot_basic import plot_complex_field,savefig
from EM_analyzer.plot.plot_2D import plot_field_and_profile_2D,plot_2D_field

N=350
D=0.05
Kappa=-0.005
theta_degree=45
#working_dir=os.path.join('/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface/test_adjust_w0/a0=50,W0=16','%d' %theta_degree,'D_%3.2f_Kappa_%+5.3f' %(D,Kappa))
working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Gaussian_beam_pulse/LG20cpl/l=2,p=3'
print(f'working_dir: {working_dir}')

laser_lambda = 0.8*C.micron		# Laser wavelength, unit:m
laser_f0=1/laser_lambda   #unit: m^-1
laser_k0=2*C.pi*laser_f0
laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_period=laser_lambda/C.speed_of_light
laser_Bc=(C.m_e*laser_omega0)/(C.elementary_charge)   #unit: T. 1.338718e+04T for 800nm laser
laser_Ec=laser_Bc*C.speed_of_light   #unit: V/m. 4.013376e+12V/m for 800nm laser
laser_a0 = 1		# Laser field strength
laser_amp=laser_a0*laser_Ec   #unit: V/m
laser_FWHM=10*C.femto   #The full width at half maximum of the intensity.
laser_tau=laser_FWHM/jnp.sqrt(2*jnp.log(2)) 
#laser_tau=laser_period/jnp.sqrt(jnp.pi)   #unit:s. One cycle
laser_Nc=laser_omega0**2*C.m_e*C.epsilon_0/C.elementary_charge**2
laser_w0_lambda= 1
laser_zR_lambda=C.pi*laser_w0_lambda**2
laser_w0=laser_w0_lambda*laser_lambda
laser_zR=laser_zR_lambda*laser_lambda
laser_theta0=1/(C.pi*laser_w0_lambda)
highest_harmonic=100

laser_spectrum_peak=C.pi*laser_amp*(laser_tau*C.speed_of_light)*(laser_w0_lambda*laser_lambda)/2
laser_flux_on_y_peak=laser_amp**2*jnp.sqrt(C.pi/2)*(laser_tau*C.speed_of_light/2)   #unit: V^2/m. ∫E(x,y=0)^2·dx
laser_energy=(C.pi/2)*jnp.square(laser_amp)*laser_tau*C.speed_of_light*laser_w0_lambda*laser_lambda/2
laser_spectrum_on_x_peak=jnp.sqrt(C.pi**3/2)*laser_amp**2*(laser_tau*C.speed_of_light)**2*(laser_w0_lambda*laser_lambda)/4   #unit: V^2·m
laser_spectrum_on_y_peak=jnp.sqrt(C.pi**3/2)*laser_amp**2*(laser_tau*C.speed_of_light)*(laser_w0_lambda*laser_lambda)**2/2   #unit: V^2·m

cells_per_lambda =200
vacuum_length_x_lambda=22.5   #lambda
vacuum_length_y_lambda=22.5   #lambda
continuation_length_lambda=0  #lambda

cells_per_lambda =50
vacuum_length_x_lambda=15   #lambda
vacuum_length_y_lambda=15   #lambda

space_length_lambda=2*(max(vacuum_length_x_lambda,vacuum_length_y_lambda)+continuation_length_lambda)
n_field_x=round(2*vacuum_length_x_lambda*cells_per_lambda)
n_field_y=round(2*vacuum_length_y_lambda*cells_per_lambda)
n_continuation=round(space_length_lambda*cells_per_lambda)
n_freq_radius=round(n_continuation*(highest_harmonic/cells_per_lambda))
n_freq_angle=n_freq_radius*2
n_freq_center=round(2*n_freq_radius)

d_x=laser_lambda/cells_per_lambda   #unit: m
d_f=1/(space_length_lambda*laser_lambda)   #unit: 1/m, d_x*d_f=1/n_continuation. d_f/laser_f0=1/space_length_lambda
d_f_radius=d_f
d_f_angle=2*C.pi/n_freq_angle

x_min=-vacuum_length_x_lambda*laser_lambda
x_max=vacuum_length_x_lambda*laser_lambda
y_min=-vacuum_length_y_lambda*laser_lambda
y_max=vacuum_length_y_lambda*laser_lambda

grid_x_axis=jnp.linspace(start=x_min,stop=x_max,num=n_field_x,endpoint=False)+d_x/2
grid_y_axis=jnp.linspace(start=y_min,stop=y_max,num=n_field_y,endpoint=False)+d_x/2
freq_x_axis=fftshift(fftfreq(n=n_continuation,d=d_x))
freq_y_axis=freq_x_axis

freq_r_axis=fftfreq(n=n_continuation,d=d_x)[0:n_freq_radius]
freq_a_axis=jnp.linspace(start=0,stop=2*C.pi,num=n_freq_angle,endpoint=False)

freq_a,freq_r=jnp.meshgrid(freq_a_axis,freq_r_axis,indexing='ij')
freq_x,freq_y=jnp.meshgrid(freq_x_axis,freq_y_axis, indexing='ij')   #freq_x.shape=(n_continuation,n_continuation)
freq_z=jnp.zeros(shape=(n_continuation,n_continuation))
freq_radius=jnp.hypot(freq_x,freq_y)   #shape=(n_continuation,n_continuation)
freq_theta = jnp.arctan2(freq_y,freq_x)

freq_vector=jnp.array((freq_x,freq_y,freq_z))   #shape=(3,n_continuation,n_continuation)
freq_radius_vector_nonzero=freq_radius.at[freq_radius<d_f/100].set(d_f/100)   #Avoid divided by zero
freq_unit_vector=freq_vector/freq_radius_vector_nonzero[jnp.newaxis, :, :]    #shape=(3,n_continuation,n_continuation)


grid_center_mask=jnp.s_[round((n_continuation-n_field_x)/2):round((n_continuation+n_field_x)/2),round((n_continuation-n_field_y)/2):round((n_continuation+n_field_y)/2)]
freq_center_mask=jnp.s_[n_continuation//2-n_freq_radius:n_continuation//2+n_freq_radius,n_continuation//2-n_freq_radius:n_continuation//2+n_freq_radius]
freq_axis_center_mask=jnp.s_[n_continuation//2-n_freq_radius:n_continuation//2+n_freq_radius]

print('space shape: ',(n_field_x,n_field_y))
print_array_size(freq_x,'freq_x')
print('freq_x[freq_center_mask].shape:',freq_x[freq_center_mask].shape)   #shape=(n_freq_center,n_freq_center)
print('λ0/dx=',laser_lambda/d_x)
print('f0/df=',laser_f0/d_f)

def weighted_quantile(x, f, q):
    f = np.asarray(f)
    x = np.asarray(x)
    cdf = cumulative_simpson(y=f,x=x,initial=0)
    cdf = cdf / cdf[-1]
    return np.interp(q, cdf, x)

def Gaussian(x,x0,w,A):
    return A*jnp.exp(-jnp.square((x-x0)/w))

def theoretical_w_z(z,z_focus,w0):
    zR = laser_k0 * w0**2 /2
    return w0 * jnp.sqrt(1 + jnp.square((z-z_focus)/zR))

def theoretical_Kappa_z(z,w0,z_focus=0):
    zR = laser_k0 * w0**2 /2
    Kappa_Z=(z-z_focus)/(jnp.square(z-z_focus)+jnp.square(zR))
    return Kappa_Z

def gaussian_beam_profile(x_y,A,z_focus,z_center,w0,z_length):
    x,y=x_y
    w_z=theoretical_w_z(z=x,z_focus=z_focus,w0=w0)
    Field_envelope=A*jnp.sqrt(w0/w_z)*jnp.exp(-jnp.square(y/w_z))*jnp.exp(-jnp.square((x-z_center)/z_length))
    return Field_envelope

def w_z_residuals(params, x, y, weights):
    return weights * (y - theoretical_w_z(x, *params))

def theoretical_transversal_field_envelope(x,z,w0,z_focus=0):
    """
        The wave is traveling in +z direction with polarisation in +x direction.
    """
    zR = laser_k0 * w0**2 /2
    w_z=theoretical_w_z(z,z_focus,w0)
    transversal_field_envelope=jnp.power(1+jnp.square((z-z_focus)/zR),-1/4)*jnp.exp(-jnp.square(x/w_z))
    return transversal_field_envelope

def theoretical_transversal_field(x,z,w0,z_focus=0,phi_cep=0):
    """
        The wave is traveling in +z direction with polarisation in +x direction.
    """
    Kappa_z=theoretical_Kappa_z(z,w0,z_focus)
    Gouy_phase=theoretical_Gouy_phase(z,w0,z_focus)
    transversal_field_envelope=theoretical_transversal_field_envelope(x,z,w0,z_focus)
    transversal_field_phase=laser_k0*(z-z_focus)-Gouy_phase/2+laser_k0*Kappa_z*jnp.square(x)/2+phi_cep
    transversal_field=transversal_field_envelope*jnp.cos(transversal_field_phase)
    return transversal_field

def theoretical_transversal_field_maximum(z,w0,z_focus=0):
    """
        The wave is traveling in +z direction with polarisation in +x direction.
    """
    zR = laser_k0 * w0**2 /2
    transversal_field_maximum=jnp.power(1+jnp.square((z-z_focus)/zR),-1/4)
    return transversal_field_maximum

def theoretical_longitudinal_field_envelope(x,z,w0,z_focus=0):
    """
        The wave is traveling in +z direction with polarisation in +x direction.
    """
    zR = laser_k0 * w0**2 /2
    w_z=theoretical_w_z(z,z_focus,w0)
    longitudinal_field_envelope=jnp.power(1+jnp.square((z-z_focus)/zR),-3/4)*jnp.exp(-jnp.square(x/w_z))*x/zR
    return longitudinal_field_envelope

def theoretical_longitudinal_field(x,z,w0,z_focus=0,phi_cep=0):
    """
        The wave is traveling in +z direction with polarisation in +x direction.
    """
    Kappa_z=theoretical_Kappa_z(z,w0,z_focus)
    Gouy_phase=theoretical_Gouy_phase(z,w0,z_focus)
    longitudinal_field_envelope=theoretical_longitudinal_field_envelope(x,z,w0,z_focus)
    longitudinal_field_phase=laser_k0*(z-z_focus)-3*Gouy_phase/2+laser_k0*Kappa_z*jnp.square(x)/2+phi_cep-jnp.pi/2
    longitudinal_field=longitudinal_field_envelope*jnp.cos(longitudinal_field_phase)
    return longitudinal_field

def theoretical_longitudinal_field_maximum(z,w0,z_focus=0):
    """
        The wave is traveling in +z direction with polarisation in +x direction.
    """
    zR = laser_k0 * w0**2 /2
    longitudinal_field_maximum=jnp.power(1+jnp.square((z-z_focus)/zR),-1/4)*jnp.exp(-0.5)/jnp.sqrt(2)*w0/zR
    return longitudinal_field_maximum


def theoretical_Gouy_phase(z,w0,z_focus=0):
    """
        The wave is traveling in +z direction with polarisation in +x direction.
    """
    zR = laser_k0 * w0**2 /2
    return jnp.atan((z-z_focus)/zR)




def plot_2D_field(
    Field:jnp.ndarray,
    ax:Optional[Axes]=None,grid_x_axis=grid_x_axis,grid_y_axis=grid_y_axis,
    normalize=laser_Ec,
    threshold:Optional[float]=None,vmin:Optional[float]=None,vmax:Optional[float]=None,
    xmin:Optional[float]=None,xmax:Optional[float]=None,ymin:Optional[float]=None,ymax:Optional[float]=None,
    cmap='RdBu',
    label=r'$a=\frac{E}{E_c}=\frac{B}{B_c}$',
    return_ax=False,plot_profile=True,name=''):
    assert Field.ndim==2
    assert grid_x_axis.ndim==1
    assert grid_y_axis.ndim==1
    n_x=grid_x_axis.size
    n_y=grid_y_axis.size
    assert Field.shape==(n_x,n_y)
    if threshold!=None:
        Field_masked = np.ma.masked_where(condition=jnp.abs(Field)/normalize <= threshold, a=Field).filled(np.nan)
    else:
        Field_masked=Field
    if vmax is None:
        vmax=1.1*jnp.max(Field_masked)/normalize
    if vmin is None:
        vmin=1.1*jnp.min(Field_masked)/normalize
    Field_max_id=tuple(jnp.asarray(jnp.where(Field==jnp.max(Field)),dtype=np.int32)[:,0])   #Field_max_id=(x_id,y_id)
    x_profile_id=Field_max_id[0]
    y_profile_id=Field_max_id[1]
    norm=Normalize(vmin=vmin,vmax=vmax,clip=True)
    xlabel=r'$\frac{x}{\lambda_0}$'
    ylabel=r'$\frac{y}{\lambda_0}$'
    if ax is None and plot_profile:
        ax_dict=plot_field_and_profile_2D(
            Field=Field_masked/normalize,Field_x_profile_top=Field_masked[:,y_profile_id]/normalize,Field_y_profile_right=Field_masked[x_profile_id,:]/normalize,
            x_axis=grid_x_axis/laser_lambda,y_axis=grid_y_axis/laser_lambda,
            vmin=vmin,vmax=vmax,cmap=cmap,norm=norm,
            xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,
            label=label,xlabel=xlabel,ylabel=ylabel,return_ax=True,name=name
        )
        ax_main=ax_dict['ax_main']
        ax_x_profile_top=ax_dict['ax_x_profile_top']
        ax_y_profile_right=ax_dict['ax_y_profile_right']
        ax_main.axvline(grid_x_axis[x_profile_id]/laser_lambda, color='k', linestyle='--', alpha=0.7)
        ax_main.axhline(grid_y_axis[y_profile_id]/laser_lambda, color='k', linestyle='--', alpha=0.7)
        ax_x_profile_top.axvline(grid_x_axis[x_profile_id]/laser_lambda, color='k', linestyle='--', alpha=0.7)
        ax_y_profile_right.axhline(grid_y_axis[y_profile_id]/laser_lambda, color='k', linestyle='--', alpha=0.7)
        ax_x_profile_top.set_title('y=%.2f·'%(grid_y_axis[y_profile_id]/laser_lambda)+r'$\lambda_0$',fontsize=20)
        ax_y_profile_right.set_title('x=%.2f·'%(grid_x_axis[x_profile_id]/laser_lambda)+r'$\lambda_0$',fontsize=20)
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6),dpi=100)
        pcm=ax.pcolormesh(grid_x_axis/laser_lambda,grid_y_axis/laser_lambda,Field_masked.T/normalize,norm=norm,cmap=cmap, shading='auto')
        ax.set_aspect('equal')
        plt.colorbar(pcm).ax.set_ylabel(label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel, rotation=0)
        ax.set_title(name)
        ax_dict={
            'fig':ax.figure,
            'ax_main':ax,
        }
    if return_ax:
        return ax_dict
    else:
        savefig(fig=ax_dict['fig'],fig_path=os.path.join(working_dir,'%s.png' %(name)))
        return os.path.join(working_dir,'Field_%s.png' %(name))
    # ax is not None and plot_profile is true case is not implemented


def plot_2D_spectrum(Field_spectrum_square:jnp.ndarray,freq_x_axis=freq_x_axis,freq_y_axis=freq_y_axis,name=''):
    assert Field_spectrum_square.ndim==2
    assert freq_x_axis.ndim==1
    assert freq_y_axis.ndim==1
    n_x=freq_x_axis.size
    n_y=freq_y_axis.size
    assert Field_spectrum_square.shape==(n_x,n_y)
    vmin=1e-6
    vmax=1.2
    norm = LogNorm(vmin=vmin, vmax=vmax)
    plot_field_and_profile_2D(Field=Field_spectrum_square/laser_spectrum_peak**2,Field_x_profile_top=Field_spectrum_square[:,n_y//2]/laser_spectrum_peak**2,
                              x_axis=freq_x_axis/laser_f0,y_axis=freq_y_axis/laser_f0,
                              vmin=vmin,vmax=vmax,norm=norm,cmap='hot',
                              xmin=0,xmax=highest_harmonic,ymin=-2,ymax=2,
                              x_profile_top_scale='log',
                              label=r'$I\left(k_x,k_y\right)/I_0$',xlabel=r'$\frac{k_x}{k_0}$',ylabel=r'$\frac{k_y}{k_0}$',x_profile_top_label=r'$I\left(k_x,k_y=0\right)/I_0$',
                              return_ax=False,name=f'Spectrum {name}')
    return 0

def filter_spectrum(Field_spectrum:jnp.ndarray,filter_range=(1.5,highest_harmonic),return_type='spectrum',name=''):
    assert Field_spectrum.shape==(n_continuation,n_continuation) or Field_spectrum.shape==(n_freq_center,n_freq_center)
    assert return_type in ['field','spectrum','both']
    if Field_spectrum.shape==(n_continuation,n_continuation):
        spectrum_mask=(freq_radius*laser_lambda>filter_range[0])&(freq_radius*laser_lambda<filter_range[1])
    else:
        spectrum_mask=(freq_radius[freq_center_mask]*laser_lambda>filter_range[0])&(freq_radius[freq_center_mask]*laser_lambda<filter_range[1])
    Field_filter_spectrum=Field_spectrum*spectrum_mask
    #Field_filter_energy=square_integral_field_2D(Field=Field_filter_spectrum,d_x=d_f,d_y=d_f,complex_array=True)
    #print('total energy: %f theoretical total energy' %(jnp.sum(Field_spectrum_square)*d_f*d_f/laser_energy))
    #print('filter energy: %f theoretical total energy' %(Field_filter_energy/laser_energy))
    if return_type=='spectrum':
        return {'Field_filter_spectrum':Field_filter_spectrum}
    else:
        Field_filter=get_field_from_spectrum(Field_filter_spectrum)
        return {'Field_filter_spectrum':Field_filter_spectrum,'Field_filter':Field_filter}
        plot_2D_field(Field=Field_filter/laser_Ec,name=name,
                     xmin=-5,xmax=5,ymin=-5,ymax=5,vmin=-3,vmax=3
                     )
        return {'Field_filter_spectrum':Field_filter_spectrum,'Field_filter':Field_filter}
        
        

#@profile
def get_envelope(field_or_spectrum:jnp.ndarray,input_type='field',name=''):
    """
        Assume the wave is in x direction. If not, please rotate the field first.
        input_type : {'field', 'spectrum'}, optional
    """
    assert input_type in ['field', 'spectrum']
    if input_type=='field':
        Field=field_or_spectrum
        assert Field.shape==(n_field_x,n_field_y)
        Field_continuation=continue_field_2D(Field,n_continuation_x=n_continuation,n_continuation_y=n_continuation,edge_length=1*cells_per_lambda)
        Field_analytic=hilbert(Field_continuation,axis=0)[grid_center_mask]
    else:
        Field_spectrum=field_or_spectrum
        assert Field_spectrum.shape==(n_continuation,n_continuation)
        h=jnp.sign(freq_x_axis)+1   #shape=(n_continuation,)
        Field_spectrum_hilbert=jnp.einsum('ij,i->ij',Field_spectrum,h)
        Field_analytic=ifft2(ifftshift(Field_spectrum_hilbert))[grid_center_mask]*n_continuation*d_f*n_continuation*d_f
    Field_envelope=jnp.abs(Field_analytic)
    Field_envelope_max=jnp.max(Field_envelope).item()
    Field_envelope_max_id=tuple(jnp.asarray(jnp.where(Field_envelope==Field_envelope_max),dtype=np.int32)[:,0])   #Field_envelope_max_id=(x_id,y_id)
    Field_analytic_phase=jnp.angle(Field_analytic)
    Field_analytic_phase = np.ma.masked_where(condition=jnp.abs(Field_envelope) <= 0.001*Field_envelope_max, a=Field_analytic_phase)
    print('Field_envelope_max: %f×Ec'%(Field_envelope_max/laser_Ec))
    print(Field_envelope_max_id)
    print('Position of the peak: x=%.2fλ0, y=%.2fλ0'%(grid_x_axis[Field_envelope_max_id[0]]/laser_lambda,grid_y_axis[Field_envelope_max_id[1]]/laser_lambda))
    Field_envelope_max_phase=Field_analytic_phase[Field_envelope_max_id]
    print('Phase at the peak: %fπ'%(Field_envelope_max_phase/C.pi))
    return {
        'Field_envelope':Field_envelope,
        'Field_envelope_max':Field_envelope_max,
        'Field_envelope_max_id':Field_envelope_max_id,
        'Field_envelope_max_phase':Field_envelope_max_phase,
    }
    
    Field_analytic_phase_centerline=Field_analytic_phase[:,Field_envelope_max_id[1]]
    Field_analytic_phase_unwrap=jnp.unwrap(Field_analytic_phase,axis=0,period=2*C.pi)
    Field_analytic_phase_unwrap=Field_analytic_phase_unwrap-Field_analytic_phase_unwrap[Field_envelope_max_id]+Field_envelope_max_phase   #Phase relative to the peak. Keep the phase at the peak
    Field_propagate_phase=laser_k0*(grid_x_axis-grid_x_axis[Field_envelope_max_id[0]])+Field_envelope_max_phase
    Field_phase_extra=Field_analytic_phase_unwrap-Field_propagate_phase[:,jnp.newaxis]   # Gouy phase
    Field_phase_gradient=jnp.gradient(Field_phase_extra,grid_x_axis,axis=0)
    zR=-0.5/(jnp.average(Field_phase_gradient[Field_envelope_max_id[0]-5:Field_envelope_max_id[0]+105,Field_envelope_max_id[1]]))
    W0=jnp.sqrt(laser_lambda*zR/C.pi)
    print(zR/laser_lambda)
    print(W0/laser_lambda)
    
    plt.plot(grid_x_axis/laser_lambda,Field_phase_extra[:,Field_envelope_max_id[1]]/C.pi,label='envelope phase',c='b',linewidth=2)
    plt.plot(grid_x_axis/laser_lambda,-0.5*jnp.atan((grid_x_axis-grid_x_axis[Field_envelope_max_id[0]])/zR)/C.pi,label='Gouy phase',linestyle='--',c='g',linewidth=1)
    #plt.xlim(grid_x_axis[Field_envelope_max_id[0]]/laser_lambda-1,grid_x_axis[Field_envelope_max_id[0]]/laser_lambda+1)
    plt.ylim(-1,1)
    plt.xlabel(xlabel='x/λ0')
    plt.ylabel(ylabel='phase/π')
    plt.legend()
    plt.title(label='Phase_%s' %(name))
    plt.savefig(os.path.join(working_dir,'Field_phase_extra_%s.png' %(name)))
    plt.clf()


#@profile
#@jax.jit
def get_envelope_width(field_or_spectrum:jnp.ndarray,input_type='field',grid_x_axis=grid_x_axis,grid_y_axis=grid_y_axis,plot=False,name=''):
    """
        Assume the wave is in x direction. If not, please rotate the field first.
        input_type : {'field', 'spectrum'}, optional
    """
    assert input_type in ['field', 'spectrum']
    n_x=grid_x_axis.size
    n_y=grid_y_axis.size
    d_x=grid_x_axis[1]-grid_x_axis[0]
    d_y=grid_y_axis[1]-grid_y_axis[0]
    Field_envelope_dict=get_envelope(field_or_spectrum=field_or_spectrum,input_type=input_type,name=name)
    grid=jnp.asarray(jnp.meshgrid(grid_x_axis,grid_y_axis, indexing='ij'))   #shape=(2,n_field_x,n_field_y)
    Field_envelope=Field_envelope_dict['Field_envelope']   #shape==(n_field_x,n_field_y)
    Field_envelope_max=Field_envelope_dict['Field_envelope_max']
    Field_envelope_max_id=Field_envelope_dict['Field_envelope_max_id']
    Field_envelope_square=jnp.square(Field_envelope)
    Field_envelope_moment_1=np.average(a=grid,axis=(1,2),weights=Field_envelope_square)   #shape=(2,). average position of the envelope (x,y)
    Field_envelope_moment_2=np.average(a=jnp.square(grid),axis=(1,2),weights=Field_envelope_square)   #shape=(2,)
    Field_envelope_moment_std=jnp.sqrt(Field_envelope_moment_2-jnp.square(Field_envelope_moment_1))
    Field_envelope_moment_1_x_id=(jnp.abs(grid_x_axis - Field_envelope_moment_1[0])).argmin().item()
    Field_envelope_moment_1_y_id=(jnp.abs(grid_y_axis - Field_envelope_moment_1[1])).argmin().item()
    Field_envelope_center_id=Field_envelope_max_id
    #Field_envelope_center_id=(Field_envelope_moment_1_x_id,Field_envelope_moment_1_y_id)
    print(Field_envelope_center_id)
    x_center=grid_x_axis[Field_envelope_center_id[0]]
    y_center=grid_y_axis[Field_envelope_center_id[1]]
    #print('FWHM obtained from envelope_x_std: %ffs'%(Field_envelope_x_std*2*jnp.sqrt(2*jnp.log(2))/C.speed_of_light/C.femto))
    Field_envelope_center_y_profile=Field_envelope[Field_envelope_center_id[0],:]
    Field_envelope_center_x_profile=Field_envelope[:,Field_envelope_center_id[1]]
    Field_envelope_at_moment_1=Field_envelope[Field_envelope_center_id]
    Field_envelope_center_y_profile_peak=jnp.asarray(peak_widths(x=Field_envelope_center_y_profile,peaks=[jnp.argmax(a=Field_envelope_center_y_profile)],rel_height=1-jnp.exp(-1)),dtype=jnp.float64).flatten()
    Field_envelope_center_x_profile_peak=jnp.asarray(peak_widths(x=Field_envelope_center_x_profile,peaks=[jnp.argmax(a=Field_envelope_center_x_profile)],rel_height=1-jnp.sqrt(2)/2)).flatten()
    Field_envelope_center_y_profile_peak_width=Field_envelope_center_y_profile_peak[0]*d_y   #unit: m
    Field_envelope_center_x_profile_peak_width=Field_envelope_center_x_profile_peak[0]*d_x   #unit: m
    x_left,x_right=map_coordinates(input=grid_x_axis,coordinates=[Field_envelope_center_x_profile_peak[-2:]],order=1)   #unit: m
    y_left,y_right=map_coordinates(input=grid_y_axis,coordinates=[Field_envelope_center_y_profile_peak[-2:]],order=1)   #unit: m
    Field_envelope_center_y_profile_peak_width=Field_envelope_center_y_profile_peak_width/2   #unit: m. half width
    Field_envelope_center_x_profile_peak_FWHM=Field_envelope_center_x_profile_peak_width/C.speed_of_light   #unit: s

    print('width at the peak: %fλ0' %(Field_envelope_center_y_profile_peak_width/laser_lambda))
    print('centerline FWHM: %ffs' %(Field_envelope_center_x_profile_peak_FWHM/C.femto))
    print('Field_envelope_center_y_profile_peak,(left,right): (%f,%f)λ0'%(y_left/laser_lambda,y_right/laser_lambda))
    energy_flux_on_y_dict=get_energy_flux_on_y(Field=Field_envelope/jnp.sqrt(2),grid_x_axis=grid_x_axis,grid_y_axis=grid_y_axis,name=name)
    energy_flux_on_y=energy_flux_on_y_dict['energy_flux_on_y']
    energy_flux_on_y_max=energy_flux_on_y_dict['energy_flux_on_y_max']
    energy_flux_on_y_peak_width=energy_flux_on_y_dict['energy_flux_on_y_peak_width']
    energy_flux_on_y_peak_left=energy_flux_on_y_dict['energy_flux_on_y_peak_left']
    energy_flux_on_y_peak_right=energy_flux_on_y_dict['energy_flux_on_y_peak_right']
    energy_flux_on_y_lower=energy_flux_on_y_dict['energy_flux_on_y_lower']
    energy_flux_on_y_upper=energy_flux_on_y_dict['energy_flux_on_y_upper']
    energy_flux_on_y_width=energy_flux_on_y_dict['energy_flux_on_y_width']
    if plot:
        if input_type=='spectrum':
            Field=get_field_from_spectrum(Field_spectrum=field_or_spectrum)
        if input_type=='field':
            Field=field_or_spectrum
        vmax=1.1
        #vmax=Field_envelope_max/laser_Ec
        ax_dict=plot_field_and_profile_2D(
            Field=Field_envelope/laser_Ec,
            Field_x_profile_top=Field[:,Field_envelope_center_id[1]]/laser_Ec,Field_y_profile_right=Field_envelope_center_y_profile/laser_Ec,Field_y_profile_left=jnp.asarray(energy_flux_on_y)/laser_flux_on_y_peak,
            y_profile_left_label=r'$\frac{S(y)}{S_0}$',
            x_axis=grid_x_axis/laser_lambda,y_axis=grid_y_axis/laser_lambda,
            vmin=0,vmax=vmax,cmap='Reds',
            xmin=grid_x_axis[0]/laser_lambda+5,xmax=grid_x_axis[-1]/laser_lambda-5,ymin=-10,ymax=10,
            return_ax=True,name=f'Field envelope {name}')
        fig=ax_dict['fig']
        ax_main=ax_dict['ax_main']
        ax_x_profile_top=ax_dict['ax_x_profile_top']
        ax_y_profile_right=ax_dict['ax_y_profile_right']
        ax_y_profile_left=ax_dict['ax_y_profile_left']
        legend_center=ax_main.plot(x_center/laser_lambda,y_center/laser_lambda, 'x', markersize=12, markeredgewidth=2, label='center',color='black')[0]
        ax_main.plot([x_left/laser_lambda, x_right/laser_lambda], [y_center/laser_lambda, y_center/laser_lambda], '-', linewidth=2, alpha=0.7,color='black')
        ax_main.plot(x_left/laser_lambda, y_center/laser_lambda, '|', markersize=10, markeredgewidth=2,color='black')
        ax_main.plot(x_right/laser_lambda, y_center/laser_lambda, '|', markersize=10, markeredgewidth=2,color='black')
        ax_main.plot([x_center/laser_lambda, x_center/laser_lambda], [y_left/laser_lambda, y_right/laser_lambda], '-', linewidth=2, alpha=0.7,color='black')
        ax_main.plot(x_center/laser_lambda, y_left/laser_lambda, '_', markersize=10, markeredgewidth=2,color='black')
        ax_main.plot(x_center/laser_lambda, y_right/laser_lambda, '_', markersize=10, markeredgewidth=2,color='black')
        rect = Rectangle((x_left/laser_lambda, y_left/laser_lambda), x_right/laser_lambda-x_left/laser_lambda, y_right/laser_lambda-y_left/laser_lambda, fill=False, edgecolor='green', linestyle='--', linewidth=1.5)
        ax_main.axvline(x_center/laser_lambda, color='k', linestyle='--', alpha=0.7)
        ax_main.axhline(y_center/laser_lambda, color='k', linestyle='--', alpha=0.7)
        ax_main.add_patch(rect)
        legend_x_envelope=ax_x_profile_top.plot(grid_x_axis/laser_lambda,Field_envelope_center_x_profile/laser_Ec,c='green',label='envelope',linewidth=1, linestyle='--', alpha=0.7)[0]
        ax_x_profile_top.plot(grid_x_axis/laser_lambda,-Field_envelope_center_x_profile/laser_Ec,c='green',linewidth=1, linestyle='--', alpha=0.7)
        ax_x_profile_top.axvline(x_center/laser_lambda, color='black', linestyle='--', alpha=0.7)
        ax_x_profile_top.axvline(x_left/laser_lambda, color='brown', linestyle='--', alpha=0.7)
        legend_x_width=ax_x_profile_top.axvline(x_right/laser_lambda, color='brown', linestyle='--', alpha=0.7, label=f'FWHM duration={Field_envelope_center_x_profile_peak_FWHM/laser_period:.2f}·'+r'$T_0$')
        legend_x_height=ax_x_profile_top.hlines(y=Field_envelope_center_x_profile_peak[1]/laser_Ec,xmin=x_left/laser_lambda,xmax=x_right/laser_lambda, linestyle='--', alpha=0.7, label='1/'+r'$\sqrt{2}$',color='brown')
        ax_x_profile_top.hlines(y=-Field_envelope_center_x_profile_peak[1]/laser_Ec,xmin=x_left/laser_lambda,xmax=x_right/laser_lambda, linestyle='--', alpha=0.7,color='brown')
        ax_x_profile_top.set_ylim(-vmax,vmax)
        ax_x_profile_top.legend(handles=[legend_x_envelope,legend_x_height],loc='upper left')
        ax_x_profile_top.set_title(f'Field and envelope at y={y_center/laser_lambda:.2f}'+r'$\lambda_0$',fontsize=20)
        ax_y_profile_right.axhline(y_left/laser_lambda, color='pink', linestyle='--', alpha=0.7)
        ax_y_profile_right.axhline(y_center/laser_lambda, color='black', linestyle='--', alpha=0.7)
        legend_y_envelope_width=ax_y_profile_right.axhline(y_right/laser_lambda, color='pink', linestyle='--', alpha=0.7,label=f'envelope half width={Field_envelope_center_y_profile_peak_width/laser_lambda:.2f}·'+r'$\lambda_0$')
        legend_y_envelope_height=ax_y_profile_right.vlines(x=Field_envelope_center_y_profile_peak[1]/laser_Ec,ymin=y_left/laser_lambda,ymax=y_right/laser_lambda, linestyle='--', alpha=0.7, label='1/e',color='pink')
        ax_y_profile_right.legend(handles=[legend_y_envelope_height])
        ax_y_profile_right.set_title(f'Envelope at x={x_center/laser_lambda:.2f}'+r'$\lambda_0$',fontsize=20)
        ax_y_profile_left.set_xlim(max(jnp.asarray(energy_flux_on_y)/laser_flux_on_y_peak),0)
        ax_y_profile_left.axhline(energy_flux_on_y_peak_left/laser_lambda, color='olive', linestyle='--', alpha=0.7)
        ax_y_profile_left.axhline(y_center/laser_lambda, color='black', linestyle='--', alpha=0.7)
        ax_y_profile_left.axhline(energy_flux_on_y_peak_right/laser_lambda, color='olive', linestyle='--', alpha=0.7)
        legend_y_flux_height=ax_y_profile_left.vlines(x=jnp.exp(-2)*energy_flux_on_y_max/laser_flux_on_y_peak,ymin=energy_flux_on_y_peak_left/laser_lambda,ymax=energy_flux_on_y_peak_right/laser_lambda, linestyle='--', alpha=0.7, label='1/e^2',color='black')
        legend_y_flux_width=ax_y_profile_left.axhline(energy_flux_on_y_upper/laser_lambda, color='purple', linestyle='--', alpha=0.7,label=f'flux width={energy_flux_on_y_width/laser_lambda:.2f}·'+r'$\lambda_0$')
        ax_y_profile_left.axhline(energy_flux_on_y_lower/laser_lambda, color='purple', linestyle='--', alpha=0.7)
        ax_y_profile_left.legend(handles=[legend_y_flux_height])
        ax_y_profile_left.set_title('Energy flux')
        ax_legend=fig.add_axes([ax_y_profile_right.get_position().x0,ax_x_profile_top.get_position().y0,ax_y_profile_right.get_position().x1-ax_y_profile_right.get_position().x0,ax_x_profile_top.get_position().y1-ax_x_profile_top.get_position().y0])
        ax_legend.legend(handles=[legend_center,legend_x_width,legend_y_envelope_width,legend_y_flux_width])
        ax_legend.set_facecolor((1, 1, 1, 0))
        ax_legend.set_xticks([])
        ax_legend.set_yticks([])
        savefig(fig,fig_path=os.path.join(working_dir,f'Field_envelope_{name}.png'))

    return {
        'Field_envelope_max':float(Field_envelope_max),
        #'Field_envelope_max_id':Field_envelope_max_id,
        'Field_envelope_max_at_x':float(grid_x_axis[Field_envelope_max_id[0]]),
        'Field_envelope_max_at_y':float(grid_y_axis[Field_envelope_max_id[1]]),
        'Field_envelope_moment_1_at_x':float(Field_envelope_moment_1[0]),
        'Field_envelope_moment_1_at_y':float(Field_envelope_moment_1[1]),
        #'Field_envelope_moment_1_id':(Field_envelope_moment_1_x_id,Field_envelope_moment_1_y_id),
        'Field_envelope_center_y_profile_peak_width':float(Field_envelope_center_y_profile_peak_width),
        'Field_envelope_center_y_profile_peak_left':float(y_left),
        'Field_envelope_center_y_profile_peak_right':float(y_right),
        'Field_envelope_center_x_profile_peak_FWHM':float(Field_envelope_center_x_profile_peak_FWHM),
        'Field_envelope_at_moment_1':float(Field_envelope_at_moment_1),
        #'Field_envelope_center_y_profile':Field_envelope_center_y_profile.tolist(),   #len=n_y
        #'energy_flux_on_y':energy_flux_on_y_dict['energy_flux_on_y'],  #len=n_y
        'energy_flux_on_y_max':energy_flux_on_y_dict['energy_flux_on_y_max'],
        'energy_flux_on_y_peak_width':energy_flux_on_y_dict['energy_flux_on_y_peak_width'],
        'energy_flux_on_y_peak_left':energy_flux_on_y_dict['energy_flux_on_y_peak_left'],
        'energy_flux_on_y_peak_right':energy_flux_on_y_dict['energy_flux_on_y_peak_right'],
        'energy_flux_on_y_median':energy_flux_on_y_dict['energy_flux_on_y_median'],
        'energy_flux_on_y_lower':energy_flux_on_y_dict['energy_flux_on_y_lower'],
        'energy_flux_on_y_upper':energy_flux_on_y_dict['energy_flux_on_y_upper'],
        'energy_flux_on_y_width':energy_flux_on_y_dict['energy_flux_on_y_width'],
    }



#@profile
def get_spectrum(Field:jnp.ndarray,name=''):
    assert Field.shape==(n_field_x,n_field_y)
    Field_continuation=continue_field_2D(Field,n_continuation_x=n_continuation,n_continuation_y=n_continuation,edge_length=1*cells_per_lambda)
    Field_spectrum=fftshift(fft2(Field_continuation))*d_x*d_x   #Unit: V·m for E or V·s for B
    return Field_spectrum
    Field_spectrum_square=jnp.square(jnp.abs(Field_spectrum))
    print('Max amp:',jnp.max(jnp.abs(Field))/laser_amp)
    print('spectrum peak: %f theoretical peak'%(jnp.max(Field_spectrum_square)/laser_spectrum_peak**2))
    print('Total energy (field) %f theoretical energy'%(square_integral_field_2D(Field=Field,d_x=d_x,d_y=d_x)/laser_energy))
    print('Total energy (spectrum) %f theoretical energy'%(square_integral_field_2D(Field=Field_spectrum,d_x=d_f,d_y=d_f,complex_array=True)/laser_energy))
    #plot_2D_field(Field=Field/laser_Ec,name=name)
    plot_2D_spectrum(Field_spectrum_square[freq_center_mask],freq_x_axis[freq_axis_center_mask],freq_y_axis[freq_axis_center_mask],name=name)
    return Field_spectrum

def get_field_from_spectrum(Field_spectrum:jnp.ndarray):
    assert Field_spectrum.shape==(n_continuation,n_continuation) or Field_spectrum.shape==(n_freq_center,n_freq_center)
    if Field_spectrum.shape==(n_continuation,n_continuation):
        Field_spectrum_continuation=Field_spectrum
    else:
        Field_spectrum_continuation=np.zeros(shape=(n_continuation,n_continuation),dtype=np.complex128)
        Field_spectrum_continuation[freq_center_mask]=Field_spectrum
    Field=jnp.real(ifft2(ifftshift(Field_spectrum_continuation))[grid_center_mask])*n_continuation*d_f*n_continuation*d_f
    return Field

def get_x_spectrum(Field:jnp.ndarray,name=''):
    assert Field.shape==(n_field_x,n_field_y)
    Field_spectrum=get_spectrum(Field=Field,name=name)
    Field_spectrum_square=jnp.square(jnp.abs(Field_spectrum))
    Field_spectrum_square_on_x=jnp.sum(Field_spectrum_square,axis=1)*d_f
    Field_spectrum_square_on_y=jnp.sum(Field_spectrum_square,axis=0)*d_f
    Field_spectrum_square_centerline=Field_spectrum_square[:,n_continuation//2]
    print(jnp.max(Field_spectrum_square_on_x)/laser_spectrum_on_x_peak)
    print(jnp.max(Field_spectrum_square_on_y)/laser_spectrum_on_y_peak)
    print(jnp.max(Field_spectrum_square_centerline)/laser_spectrum_peak**2)
    print(freq_x_axis[jnp.where(Field_spectrum_square_on_x==jnp.max(Field_spectrum_square_on_x))[0]]*laser_lambda)
    print(jnp.sum(Field_spectrum_square_on_x)*d_f/laser_energy)
    plt.semilogy(freq_x_axis*laser_lambda,Field_spectrum_square_on_x/laser_spectrum_on_x_peak,label='spectrum_on_x')
    plt.semilogy(freq_x_axis*laser_lambda,Field_spectrum_square_centerline/laser_spectrum_peak**2,label='spectrum_centerline')
    plt.xlabel(xlabel=r'$\frac{k_x}{k_0}$')
    plt.ylabel(ylabel=r'$\frac{I\left(k_x\right)}{I_0}$')
    plt.title('%s spectrum' %(name))
    plt.xlim(0,highest_harmonic)
    plt.ylim(1e-6,1)
    plt.legend()
    plt.savefig(os.path.join(working_dir,'Spectrum_square_on_x_%s.png' %(name)))
    plt.clf()
    return Field_spectrum_square_on_x
    plt.semilogy(freq_x_axis*laser_lambda,Field_spectrum_square_on_y/laser_spectrum_on_y_peak,label='spectrum_on_y')
    plt.xlabel(xlabel=r'$\frac{k_y}{k_0}$')
    plt.ylabel(ylabel='I(ky)/I0')
    plt.title('%s spectrum' %(name))
    plt.xlim(-highest_harmonic,highest_harmonic)
    plt.ylim(1e-6,1)
    plt.legend()
    plt.savefig(os.path.join(working_dir,'Spectrum_square_on_y_%s.png' %(name)))
    plt.clf()


def get_polar_spectrum(field_or_spectrum:jnp.ndarray,input_type='field',name=''):
    assert input_type in ['field', 'spectrum']
    if input_type=='field':
        Field=field_or_spectrum
        assert Field.shape==(n_field_x,n_field_y)
        Field_spectrum=get_spectrum(Field=Field,name=name)
    else:
        Field_spectrum=field_or_spectrum
        assert Field_spectrum.shape==(n_continuation,n_continuation)
    Field_spectrum_square=jnp.square(jnp.abs(Field_spectrum))
    Field_spectrum_square_polar=cv2.warpPolar(src=np.asarray(Field_spectrum_square[freq_center_mask].T),dsize=(n_freq_radius,n_freq_angle),center=(n_freq_radius,n_freq_radius),maxRadius=n_freq_radius,flags=cv2.WARP_POLAR_LINEAR)   #Field_spectrum_square_polar.shape=(n_freq_angle,n_freq_radius)
    Field_spectrum_square_on_radius=jnp.sum(Field_spectrum_square_polar*freq_r,axis=0)*d_f_angle/2
    Field_spectrum_square_on_angle=jnp.sum(Field_spectrum_square_polar*freq_r,axis=1)*d_f_radius
    Field_spectrum_square_angle_1_moment=jnp.average(
        a=freq_a[round(n_freq_angle/4):round(n_freq_angle*3/4),:],
        weights=Field_spectrum_square_polar[round(n_freq_angle/4):round(n_freq_angle*3/4),:],
        axis=0)
    Field_spectrum_square_angle_2_moment=jnp.average(
        a=jnp.square(freq_a[round(n_freq_angle/4):round(n_freq_angle*3/4),:]),
        weights=Field_spectrum_square_polar[round(n_freq_angle/4):round(n_freq_angle*3/4),:],
        axis=0)
    center_angle=Field_spectrum_square_angle_1_moment
    divergent_angle=jnp.sqrt(Field_spectrum_square_angle_2_moment-jnp.square(Field_spectrum_square_angle_1_moment))*2
    #return Field_spectrum_square_polar,Field_spectrum_square_on_radius
    print(jnp.sum(Field_spectrum_square_on_radius)*d_f_radius*2/laser_energy)
    print(jnp.sum(Field_spectrum_square_on_angle)*d_f_angle/laser_energy)
    Field_spectrum_square_on_radius_max_id=jnp.argmax(Field_spectrum_square_on_radius)
    Field_spectrum_square_on_radius_max=Field_spectrum_square_on_radius[Field_spectrum_square_on_radius_max_id]
    print('I(kr) peak at kr/k0=%f, I(kr)_max/I0=%f' %(freq_r_axis[Field_spectrum_square_on_radius_max_id]/laser_f0,Field_spectrum_square_on_radius_max/laser_spectrum_on_x_peak))
    print(freq_a_axis[jnp.where(Field_spectrum_square_on_angle==jnp.max(Field_spectrum_square_on_angle))[0]])
    print(jnp.max(Field_spectrum_square_polar)/laser_spectrum_peak**2)
    print(jnp.max(Field_spectrum_square_on_angle)/(laser_spectrum_on_y_peak*laser_f0/2))
    pd.DataFrame(data={'kr/k0':freq_r_axis/laser_f0,'θ':divergent_angle}).to_hdf(path_or_buf=os.path.join(working_dir,'spectrum.hdf5'),mode='a',key='Field_spectrum_divergent_%s' %(name))
    return Field_spectrum_square_polar,Field_spectrum_square_on_radius
    norm = LogNorm(vmin=1e-6, vmax=1.2,clip=True)
    ax_dict=plot_field_and_profile_2D(
        Field=Field_spectrum_square_polar.T/laser_spectrum_peak**2,
        Field_x_profile_top=Field_spectrum_square_on_radius/laser_spectrum_on_x_peak,
        Field_x_profile_bottom=divergent_angle,
        Field_y_profile_right=Field_spectrum_square_on_angle/(laser_spectrum_on_y_peak*laser_f0/2),
        Field_y_profile_left=Field_spectrum_square_polar[:,round(space_length_lambda)]/laser_spectrum_peak**2,
        x_axis=freq_r_axis/laser_f0,y_axis=freq_a_axis,
        vmin=1e-6,vmax=1.2,norm=norm,xmin=0.5,xmax=highest_harmonic,ymin=np.pi-0.3,ymax=np.pi+0.3,
        cmap='hot',
        label=r'$\frac{I(k_r,k_\theta)}{I_0}$',xlabel=r'$\frac{k_r}{k_0}$',ylabel=r'${\theta}$ (rad)',
        x_profile_top_scale='log',x_profile_top_label=r'$\frac{I\left(k_r\right)}{I_0}$',
        x_profile_bottom_scale='linear',x_profile_bottom_label=r'${\theta}_{std}$ (rad)',
        y_profile_left_scale='log',y_profile_left_label=r'$\frac{I(k_r=k_0,k_\theta)}{I_0}$',
        y_profile_right_scale='log',y_profile_right_label=r'$\frac{I\left(k_\theta\right)}{I_0}$',
        return_ax=True,name=f'Spectrum_polar_{name}'
        )
    fig=ax_dict['fig']
    ax_main=ax_dict['ax_main']
    ax_main.axvline(x=freq_r_axis[round(space_length_lambda)]/laser_f0,linewidth=1,linestyle='--',c='g')
    ax_x_profile_top=ax_dict['ax_x_profile_top']
    ax_x_profile_top.axvline(x=freq_r_axis[round(space_length_lambda)]/laser_f0,linewidth=1,linestyle='--',c='g')
    ax_x_profile_top.set_title('∫I(kr,kθ)·r·dθ',fontsize=20)
    ax_x_profile_bottom=ax_dict['ax_x_profile_bottom']
    ax_x_profile_bottom.axvline(x=freq_r_axis[round(space_length_lambda)]/laser_f0,linewidth=1,linestyle='--',c='g')
    ax_x_profile_bottom.set_ylim(0,0.2)
    ax_y_profile_right=ax_dict['ax_y_profile_right']
    ax_y_profile_right.set_title('∫I(kr,kθ)·r·dr',fontsize=20)
    ax_y_profile_left=ax_dict['ax_y_profile_left']
    ax_y_profile_left.set_title('I(kr=k0,kθ)',fontsize=20)
    savefig(fig,fig_path=os.path.join(working_dir,'Spectrum_square_polar_%s.png' %(name)))
    #pd.DataFrame(data={'I(kr)/I0':Field_spectrum_square_on_radius/laser_spectrum_on_x_peak,'kr/k0':freq_r_axis/laser_f0}).to_hdf(path_or_buf=os.path.join(working_dir,'spectrum.hdf5'),mode='a',key='Field_spectrum_square_on_radius_%s' %(name))
    #pd.DataFrame(data={'kr/k0':freq_r_axis/laser_f0,'θ':divergent_angle}).to_hdf(path_or_buf=os.path.join(working_dir,'spectrum.hdf5'),mode='a',key='Field_spectrum_divergent_%s' %(name))
    return Field_spectrum_square_polar,Field_spectrum_square_on_radius


def get_divergent_angle_from_spectrum(field_or_spectrum:jnp.ndarray,input_type='field',name=''):
    """
        Assume the wave is in x direction. If not, please rotate the field first.
        input_type : {'field', 'spectrum'}, optional
    """
    assert input_type in ['field', 'spectrum']
    if input_type=='field':
        Field=field_or_spectrum
        assert Field.shape==(n_field_x,n_field_y)
        Field_spectrum_square_polar,_=get_polar_spectrum(field_or_spectrum=Field,input_type='field',name=name)
    else:
        Field_spectrum=field_or_spectrum
        assert Field_spectrum.shape==(n_continuation,n_continuation)
        Field_spectrum_square_polar,_=get_polar_spectrum(field_or_spectrum=Field_spectrum,input_type='spectrum',name=name)
    Field_spectrum_square_on_angle=jnp.sum(Field_spectrum_square_polar*freq_r,axis=1)*d_f_radius
    Field_spectrum_square_angle_1_moment=jnp.average(a=freq_a_axis[round(n_freq_angle/4):round(n_freq_angle*3/4)],weights=Field_spectrum_square_on_angle[round(n_freq_angle/4):round(n_freq_angle*3/4)])
    Field_spectrum_square_angle_2_moment=jnp.average(a=jnp.square(freq_a_axis[round(n_freq_angle/4):round(n_freq_angle*3/4)]),weights=Field_spectrum_square_on_angle[round(n_freq_angle/4):round(n_freq_angle*3/4)])
    center_angle=Field_spectrum_square_angle_1_moment
    divergent_angle=jnp.sqrt(Field_spectrum_square_angle_2_moment-jnp.square(Field_spectrum_square_angle_1_moment))*2
    print('center angle: %fθ0' %(center_angle/laser_theta0))
    print(f'divergent angle: {divergent_angle/laser_theta0}θ0={divergent_angle}rad={np.degrees(divergent_angle)}°')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.semilogy(freq_a_axis, Field_spectrum_square_on_angle/(laser_spectrum_on_y_peak*laser_f0/2),c='r',label='Angular spectrum',linewidth=2)
    #ax.axvline(x=Field_spectrum_square_angle_1_moment,ymin=0,ymax=1,linewidth=1,linestyle='-',c='b')
    ax.axvline(x=Field_spectrum_square_angle_1_moment-divergent_angle-C.pi,label=f'divergent angle={np.degrees(divergent_angle):.1f}°',linewidth=1,linestyle='--',c='b')
    ax.axvline(x=Field_spectrum_square_angle_1_moment+divergent_angle-C.pi,linewidth=1,linestyle='--',c='b')
    ax.set_ylim(1e-6,1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    plt.legend()
    plt.title('Angular energy distribution %s' %(name))
    savefig(fig,fig_path=os.path.join(working_dir,'Spectrum_square_polar_distribution_%s.png' %(name)))
    return {
        'center_angle': float(center_angle),   #unit: rad
        'divergent_angle': float(divergent_angle),   #unit: rad
    }

def get_energy_flux_on_y(Field:jnp.ndarray,grid_x_axis=grid_x_axis,grid_y_axis=grid_y_axis,name=''):
    """_summary_
    If input Field is its envelope, please divide it by squr(2) since <(cos(x))^2>=1/2
    Args:
        Field (jnp.ndarray): _description_
        grid_x_axis (_type_, optional): _description_. Defaults to grid_x_axis.
        grid_y_axis (_type_, optional): _description_. Defaults to grid_y_axis.
        name (str, optional): _description_. Defaults to ''.

    Returns:
        _type_: _description_
    """
    assert grid_x_axis.ndim==1
    assert grid_y_axis.ndim==1
    n_x=grid_x_axis.size
    n_y=grid_y_axis.size
    assert Field.shape==(n_x,n_y)
    energy_flux_on_y=jnp.einsum('ij,ij->j',Field,Field)*d_x
    y_center_id=jnp.argmax(a=energy_flux_on_y)
    y_center=grid_y_axis[y_center_id]
    energy_flux_on_y_max=energy_flux_on_y[y_center_id]
    print(energy_flux_on_y_max/laser_flux_on_y_peak)
    energy_flux_on_y_peak=jnp.asarray(peak_widths(x=energy_flux_on_y,peaks=[y_center_id],rel_height=1-jnp.exp(-2))).flatten()
    energy_flux_on_y_peak_width=energy_flux_on_y_peak[0]*d_x/2   #unit: m. half width
    print(f'energy_flux_on_y_peak_width={energy_flux_on_y_peak_width/laser_lambda:.2f}·λ0')
    y_left,y_right=map_coordinates(input=grid_y_axis,coordinates=[energy_flux_on_y_peak[-2:]],order=1)   #unit: m
    y_lower,y_median,y_upper=weighted_quantile(grid_y_axis,energy_flux_on_y,[(1-erf(np.sqrt(1/2)))/2,0.5,(1+erf(np.sqrt(1/2)))/2])   #-w/2 to w/2 contains erf(1/sqrt(2))=0.68 of the total flux energy
    return {
        'energy_flux_on_y':energy_flux_on_y.tolist(),  #len=n_field_y
        'energy_flux_on_y_peak_width':float(energy_flux_on_y_peak_width),
        'energy_flux_on_y_max':float(energy_flux_on_y_max),
        'energy_flux_on_y_peak_left':float(y_left),
        'energy_flux_on_y_peak_right':float(y_right),
        'energy_flux_on_y_median':float(y_median),
        'energy_flux_on_y_lower':float(y_lower),
        'energy_flux_on_y_upper':float(y_upper),
        'energy_flux_on_y_width':float(y_upper-y_lower),
    }
    ax_dict=plot_field_and_profile_2D(
        Field=Field/(laser_Ec*laser_a0),Field_x_profile_top=Field[:,y_center_id]/(laser_Ec*laser_a0),Field_y_profile_right=energy_flux_on_y/laser_flux_on_y_peak,
        x_axis=grid_x_axis/laser_lambda,y_axis=grid_y_axis/laser_lambda,
        vmin=-1,vmax=1,cmap='RdBu',
        label=r'$\frac{E}{E_0}$',y_profile_right_label=r'$\frac{S(y)}{S_0}$',
        return_ax=True,name=f'Field energy flux {name}')
    fig=ax_dict['fig']
    ax_main=ax_dict['ax_main']
    ax_y_profile_right=ax_dict['ax_y_profile_right']
    ax_main.axhline(y_center/laser_lambda, color='k', linestyle='--', alpha=0.7)
    ax_y_profile_right.axhline(y_left/laser_lambda, color='g', linestyle='--', alpha=0.7)
    ax_y_profile_right.axhline(y_center/laser_lambda, color='k', linestyle='--', alpha=0.7)
    legend_y_width=ax_y_profile_right.axhline(y_right/laser_lambda, color='g', linestyle='--', alpha=0.7,label=f'half_width={energy_flux_on_y_peak_width/laser_lambda:.2f}·'+r'$\lambda_0$')
    legend_y_height=ax_y_profile_right.vlines(x=energy_flux_on_y_peak[1]/laser_Ec,ymin=y_left/laser_lambda,ymax=y_right/laser_lambda, linestyle='--', alpha=0.7, label='exp(-2)',color='black')
    ax_y_profile_right.legend(handles=[legend_y_width,legend_y_height])
    ax_y_profile_right.set_xlim(0,1)
    plt.savefig(os.path.join(working_dir,f'energy_flux_{name}.png'))
    print(os.path.join(working_dir,f'energy_flux_{name}.png'))
    plt.close(fig)
    plt.clf()
    return {
        'energy_flux_on_y':energy_flux_on_y.tolist(),  #len=n_field_y
        'energy_flux_on_y_peak_width':energy_flux_on_y_peak_width,
        'energy_flux_on_y_peak_left':y_left,
        'energy_flux_on_y_peak_right':y_right,
    }

#@profile
#@jax.jit
def get_evolution_from_spectrum(
    A_plus_spectrum_vector:jnp.ndarray,
    A_minus_spectrum_vector:jnp.ndarray,
    evolution_time=0.0,
    ):
    r"""
    Args:
        A_plus_spectrum_vector: F\left\{\mathbit{A}_+\right\}=\frac{1}{2}\left(F\left\{\mathbit{E}\right\}-\hat{\mathbit{k}}\times F\left\{c\mathbit{B}\right\}\right).  Unit: V·m
        A_minus_spectrum_vector: F\left\{\mathbit{A}_-\right\}=\frac{1}{2}\left(F\left\{\mathbit{E}\right\}+\hat{\mathbit{k}}\times F\left\{c\mathbit{B}\right\}\right). Unit: V·m. 
        evolution_time: Unit: s
    """
    assert A_plus_spectrum_vector.shape==(3,n_continuation,n_continuation) or A_plus_spectrum_vector.shape==(3,n_freq_center,n_freq_center) 
    assert A_minus_spectrum_vector.shape==(3,n_continuation,n_continuation) or A_minus_spectrum_vector.shape==(3,n_freq_center,n_freq_center)
    print_array_size(A_plus_spectrum_vector,'A_plus_spectrum_vector')
    if A_plus_spectrum_vector.shape==(3,n_continuation,n_continuation):
        time_phase_shift=2*C.pi*freq_radius*C.speed_of_light*evolution_time   #2πc|f|t. shape=(n_continuation,n_continuation)
    else:
        time_phase_shift=2*C.pi*freq_radius[freq_center_mask]*C.speed_of_light*evolution_time   #2πc|f|t. shape=(n_freq_center,n_freq_center)
    A_plus_spectrum_evolution_vector=A_plus_spectrum_vector*jnp.exp(-1j*time_phase_shift)[jnp.newaxis,:,:]
    A_minus_spectrum_evolution_vector=A_minus_spectrum_vector*jnp.exp(1j*time_phase_shift)[jnp.newaxis,:,:]
    Electric_Field_spectrum_evolution_vector=(A_plus_spectrum_evolution_vector+A_minus_spectrum_evolution_vector)   #Unit: V·m
    if A_plus_spectrum_vector.shape==(3,n_continuation,n_continuation):
        freq_unit_vector_local=freq_unit_vector
    else:
        freq_unit_vector_local=freq_unit_vector[(slice(None),) + freq_center_mask]
    Magnetic_Field_spectrum_evolution_vector=jnp.cross(a=freq_unit_vector_local,b=(A_plus_spectrum_evolution_vector-A_minus_spectrum_evolution_vector),axisa=0,axisb=0,axisc=0)   #Unit: V·m
    return Electric_Field_spectrum_evolution_vector,Magnetic_Field_spectrum_evolution_vector

#@profile
#@jax.jit
def get_evolution(
    Electric_Field_Ex=jnp.zeros(shape=(n_field_x,n_field_y)),
    Electric_Field_Ey=jnp.zeros(shape=(n_field_x,n_field_y)),
    Electric_Field_Ez=jnp.zeros(shape=(n_field_x,n_field_y)),
    Magnetic_Field_Bx=jnp.zeros(shape=(n_field_x,n_field_y)),
    Magnetic_Field_By=jnp.zeros(shape=(n_field_x,n_field_y)),
    Magnetic_Field_Bz=jnp.zeros(shape=(n_field_x,n_field_y)),
    evolution_time_list=[0.0],
    window_shift_velocity=jnp.array((0.0,0.0,0.0)),
    grid_x_axis=grid_x_axis,
    grid_y_axis=grid_y_axis,
    name='',
    ):
    """
    Get the evolution of the EM field via phase shift.
    Args:
        Electric_Field_Ex: Unit: V/m
        Electric_Field_Ey: Unit: V/m
        Electric_Field_Ez: Unit: V/m
        Magnetic_Field_Bx: Unit: T
        Magnetic_Field_By: Unit: T
        Magnetic_Field_Bz: Unit: T
        evolution_time_list: List of the evolution time. Unit: s
        window_shift_velocity: The (3d) velocity of the window following the evolution of the field. (vx,vy,vz=0). Unit: m/s
    """
    assert Electric_Field_Ex.shape==(n_field_x,n_field_y)
    assert Electric_Field_Ey.shape==(n_field_x,n_field_y)
    assert Electric_Field_Ez.shape==(n_field_x,n_field_y)
    assert Magnetic_Field_Bx.shape==(n_field_x,n_field_y)
    assert Magnetic_Field_By.shape==(n_field_x,n_field_y)
    assert Magnetic_Field_Bz.shape==(n_field_x,n_field_y)
    window_shift_velocity=jnp.array(window_shift_velocity,dtype=jnp.float64)
    evolution_time_list=jnp.array(evolution_time_list).flatten()
    assert evolution_time_list.size>0
    assert window_shift_velocity.shape==(3,)
    Electric_Field_Ex_spectrum=get_spectrum(Electric_Field_Ex,name=f'{name}_Ex')[freq_center_mask]
    Electric_Field_Ey_spectrum=get_spectrum(Electric_Field_Ey,name=f'{name}_Ey')[freq_center_mask]
    Electric_Field_Ez_spectrum=get_spectrum(Electric_Field_Ez,name=f'{name}_Ez')[freq_center_mask]
    Magnetic_Field_Bx_spectrum=get_spectrum(C.speed_of_light*Magnetic_Field_Bx,name=f'{name}_Bx')[freq_center_mask]   #convert the unit of B to E
    Magnetic_Field_By_spectrum=get_spectrum(C.speed_of_light*Magnetic_Field_By,name=f'{name}_By')[freq_center_mask]
    Magnetic_Field_Bz_spectrum=get_spectrum(C.speed_of_light*Magnetic_Field_Bz,name=f'{name}_Bz')[freq_center_mask]
    Electric_Field_spectrum_vector=jnp.array((Electric_Field_Ex_spectrum,Electric_Field_Ey_spectrum,Electric_Field_Ez_spectrum))   #shape=(3,n_continuation,n_continuation) or (3,n_freq_center,n_freq_center)
    Magnetic_Field_spectrum_vector=jnp.array((Magnetic_Field_Bx_spectrum,Magnetic_Field_By_spectrum,Magnetic_Field_Bz_spectrum))
    freq_unit_vector_cross_Magnetic_Field_spectrum_vector=jnp.cross(a=freq_unit_vector[(slice(None),) + freq_center_mask],b=Magnetic_Field_spectrum_vector,axisa=0,axisb=0,axisc=0)
    A_plus_spectrum_vector=(Electric_Field_spectrum_vector-freq_unit_vector_cross_Magnetic_Field_spectrum_vector)/2
    A_minus_spectrum_vector=(Electric_Field_spectrum_vector+freq_unit_vector_cross_Magnetic_Field_spectrum_vector)/2

    def do_each(evolution_time):
        print(f'evolution time={evolution_time/laser_period:+05.01f}T0')
        return_dict={}
        window_grid_x_axis=grid_x_axis+window_shift_velocity[0]*evolution_time
        window_grid_y_axis=grid_y_axis+window_shift_velocity[1]*evolution_time
        window_phase_shift=2*C.pi*jnp.einsum('ijk,i->jk',freq_vector[(slice(None),) + freq_center_mask],window_shift_velocity)*evolution_time   #2π(fx,fy,fz)·(vx,vy,vz)·t. shape=(n_continuation,n_continuation) or (3,n_freq_center,n_freq_center)
        Electric_Field_spectrum_evolution_vector,Magnetic_Field_spectrum_evolution_vector=get_evolution_from_spectrum(
            A_plus_spectrum_vector=A_plus_spectrum_vector,
            A_minus_spectrum_vector=A_minus_spectrum_vector,
            evolution_time=evolution_time
        )
        Electric_Field_spectrum_evolution_in_window_vector=Electric_Field_spectrum_evolution_vector*jnp.exp(1j*window_phase_shift)[jnp.newaxis,:,:]
        Magnetic_Field_spectrum_evolution_in_window_vector=Magnetic_Field_spectrum_evolution_vector*jnp.exp(1j*window_phase_shift)[jnp.newaxis,:,:]
        Electric_Field_Ex_evolution_spectrum,Electric_Field_Ey_evolution_spectrum,Electric_Field_Ez_evolution_spectrum=Electric_Field_spectrum_evolution_in_window_vector
        Magnetic_Field_Bx_evolution_spectrum,Magnetic_Field_By_evolution_spectrum,Magnetic_Field_Bz_evolution_spectrum=Magnetic_Field_spectrum_evolution_in_window_vector
        Electric_Field_Ex_evolution=get_field_from_spectrum(Electric_Field_Ex_evolution_spectrum)
        Electric_Field_Ey_evolution=get_field_from_spectrum(Electric_Field_Ey_evolution_spectrum)
        Electric_Field_Ez_evolution=get_field_from_spectrum(Electric_Field_Ez_evolution_spectrum)
        Magnetic_Field_Bz_evolution=get_field_from_spectrum(Magnetic_Field_Bz_evolution_spectrum)/C.speed_of_light
        return {
            'Ex_energy': square_integral_field_2D(Electric_Field_Ex_evolution, d_x=d_x, d_y=d_x),
            'Ex_max': float(jnp.max(jnp.abs(Electric_Field_Ex_evolution))),
            'Ey_energy': square_integral_field_2D(Electric_Field_Ey_evolution, d_x=d_x, d_y=d_x),
            'Ey_max': float(jnp.max(jnp.abs(Electric_Field_Ey_evolution))),
            'Ez_energy': square_integral_field_2D(Electric_Field_Ez_evolution, d_x=d_x, d_y=d_x),
            'Ez_max': float(jnp.max(jnp.abs(Electric_Field_Ez_evolution))),
        }
        
        
        plot_2D_field(
            Magnetic_Field_Bz_evolution/laser_Bc,
            grid_x_axis=window_grid_x_axis/laser_lambda,
            grid_y_axis=window_grid_y_axis/laser_lambda,
            vmin=-4,vmax=4,xmin=window_grid_x_axis[0]/laser_lambda+10,xmax=window_grid_x_axis[0]/laser_lambda+35,ymin=-10,ymax=10,
            name=f'Bz_{name}_{evolution_time/laser_period:+05.01f}T0'
            )
        for key in order_interested.keys():
            print(key)
            Magnetic_Field_Bz_nth_evolution=filter_spectrum(Magnetic_Field_Bz_evolution_spectrum,order_interested[key],return_type='both')['Field_filter']
            #continue
            if key=='2':
                return_dict[key]=get_envelope_width(field_or_spectrum=Magnetic_Field_Bz_nth_evolution,input_type='field',grid_x_axis=window_grid_x_axis,grid_y_axis=window_grid_y_axis,plot=True,name=f'Bz_{key}_{name}_{evolution_time/laser_period:+05.01f}T0')
                continue
            else:
                return_dict[key]=get_envelope_width(field_or_spectrum=Magnetic_Field_Bz_nth_evolution,input_type='field',grid_x_axis=window_grid_x_axis,grid_y_axis=window_grid_y_axis,name=f'Bz_{key}_{name}_{evolution_time/laser_period:+05.01f}T0')
        del Electric_Field_spectrum_evolution_in_window_vector
        del Magnetic_Field_spectrum_evolution_in_window_vector
        gc.collect()
        return return_dict


        write_field_2D(
            Field_list=[Magnetic_Field_Bz_evolution],
            x_axis=window_grid_x_axis,y_axis=window_grid_y_axis,
            name_list=['Magnetic_Field_Bz'],
            nc_name=os.path.join(working_dir,f'{name}_{evolution_time/laser_period:+05.01f}T0.nc')
            )
        #return return_dict
        write_field_2D(
            Field_list=[Electric_Field_Ex_evolution,Electric_Field_Ey_evolution,Magnetic_Field_Bz_evolution],
            x_axis=window_grid_x_axis,y_axis=window_grid_y_axis,
            name_list=['Electric_Field_Ex','Electric_Field_Ey','Magnetic_Field_Bz'],
            nc_name=os.path.join(working_dir,f'{name}_time={evolution_time/laser_period:+05.01f}T0.nc')
            )
    task_list=[delayed(do_each)(evolution_time) for evolution_time in evolution_time_list]
    return_list=Parallel(n_jobs=10, backend="loky")(task_list)
    return return_list
    for key in order_interested.keys():
        pd.DataFrame(data=[return_dict[key] for return_dict in return_list],index=evolution_time_list).to_hdf(path_or_buf=os.path.join(working_dir,'waist_fine.hdf5'),key=f'{name}_{key}',mode='a')
    

    return return_list
    return_list=[]
    for evolution_time in evolution_time_list:
        return_list.append(do_each(evolution_time))

data_dict=read_nc(nc_name=os.path.join(working_dir,'Field_t=+00.0T0.nc'),key_name_list=[
    'Electric_Field_Ex',
    'Electric_Field_Ey',
    'Electric_Field_Ez',
    #'Magnetic_Field_Bx',
    #'Magnetic_Field_By',
    #'Magnetic_Field_Bz',
    ])

Electric_Field_Ex=data_dict['Electric_Field_Ex']  #shape=(Nx, Ny, Nz)
Electric_Field_Ey=data_dict['Electric_Field_Ey']  #shape=(Nx, Ny, Nz)
Electric_Field_Ez=data_dict['Electric_Field_Ez']  #shape=(Nx, Ny, Nz)

x_axis=data_dict['x']
y_axis=data_dict['y']
z_axis=data_dict['z']
Nx=x_axis.size
Ny=y_axis.size
Nz=z_axis.size
xc_id=Nx//2
yc_id=Ny//2
zc_id=Nz//2
phi_pol=jnp.radians(30)
phi_cep = 0.0
ax_dict=plot_field_and_profile_2D(
    Field=Electric_Field_Ex[:,yc_id,:]/laser_Ec,
    Field_x_profile_top=Electric_Field_Ex[:,yc_id,zc_id]/laser_Ec,
    Field_y_profile_right=Electric_Field_Ex[xc_id,yc_id,:]/laser_Ec,
    x_axis=x_axis/laser_lambda,
    y_axis=z_axis/laser_lambda,
    vmin=-1,vmax=1,cmap='RdBu',
    xmin=-5,xmax=5,ymin=-5,ymax=5,
    label=r'$E_z$',xlabel=r'$\frac{x}{\lambda_0}$',ylabel=r'$\frac{z}{\lambda_0}$',
    return_ax=True
)
fig=ax_dict['fig']
ax_main=ax_dict['ax_main']
ax_main.axvline(x=x_axis[xc_id]/laser_lambda, color='k', linestyle='--', alpha=0.7)
ax_main.axhline(y=z_axis[zc_id]/laser_lambda, color='k', linestyle='--', alpha=0.7)
ax_y_profile_right=ax_dict['ax_y_profile_right']
ax_y_profile_right.axhline(y=z_axis[zc_id]/laser_lambda, color='k', linestyle='--', alpha=0.7)
ax_y_profile_right.set_title(f'z={z_axis[zc_id]/laser_lambda:.2f}·λ0')
ax_y_profile_right.legend()
ax_x_profile_top=ax_dict['ax_x_profile_top']
#ax_x_profile_top.plot(x_axis/laser_lambda,theoretical_longitudinal_field(x=y_axis[yc_id],z=x_axis,w0=laser_w0,phi_cep=phi_cep),c='orange',label='theoretical Ez',linewidth=2)
ax_x_profile_top.axvline(x=x_axis[xc_id]/laser_lambda, color='k', linestyle='--', alpha=0.7)
#ax_x_profile_top.legend()
ax_x_profile_top.set_title(f'x={x_axis[xc_id]/laser_lambda:.2f}·λ0',fontsize=20)
plt.savefig(os.path.join(working_dir,'Ex_y=0.png'))

exit(0)
evolution_time_list=jnp.linspace(0*laser_period,+100*laser_period,101,endpoint=True)
#evolution_time_list=26*laser_period
order_interested={
    'all':(0,highest_harmonic),
    '1':(0.5,1.5),
    '2':(1.5,2.5),
    '3':(2.5,3.5),
    '4':(3.5,4.5),
    '5':(4.5,5.5),
    '10':(9.5,10.5),
    '15':(14.5,15.5),
    #'[2,200]':(1.5,200.5)
}



#plot 2D Bz and Ne
Bz=read_dat(dat_name='/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface/test_adjust_w0/a0=50,W0=16/Initialize_Field/45/Bz',shape=(n_field_x,n_field_y))
Ne=read_dat(dat_name='/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface/test_adjust_w0/Initialize_Target/D_0.05,Kappa_-0.005/Ne',shape=(n_field_x,n_field_y))
ax_dict=plot_2D_field(Field=Bz,normalize=laser_Bc,vmin=-laser_a0,vmax=laser_a0,return_ax=True,plot_profile=False)
ax_main=ax_dict['ax_main']
ax_dict=plot_2D_field(Field=Ne,ax=ax_main,normalize=laser_Nc,threshold=5,label=r'$\frac{N_e}{N_c}$',vmin=0,vmax=350,cmap='grey_r',plot_profile=False,return_ax=True)
fig=ax_dict['fig']
savefig(fig,os.path.join(working_dir,'fields0000.png'))
exit(0)
name='reflection'
#data_dict=read_nc(nc_name=os.path.join(working_dir,f'{name}_focus_clip_{cells_per_lambda}cpl.nc'),key_name_list=['Electric_Field_Ex','Electric_Field_Ey','Magnetic_Field_Bz','x','y'])
data_dict=read_nc(nc_name=os.path.join(working_dir,f'fields0000_{cells_per_lambda}cpl.nc'),key_name_list=['Electric_Field_Ex','Electric_Field_Ey','Magnetic_Field_Bz','x','y'])
Electric_Field_Ex=data_dict['Electric_Field_Ex']
Electric_Field_Ey=data_dict['Electric_Field_Ey']
Magnetic_Field_Bz=data_dict['Magnetic_Field_Bz']
grid_x_axis=data_dict['x']
grid_y_axis=data_dict['y']




return_dict=get_evolution(
    Electric_Field_Ex=Electric_Field_Ex,
    Electric_Field_Ey=Electric_Field_Ey,
    Magnetic_Field_Bz=Magnetic_Field_Bz,
    evolution_time_list=evolution_time_list,
    window_shift_velocity=(C.speed_of_light,0,0),
    grid_x_axis=grid_x_axis,
    grid_y_axis=grid_y_axis,
    name=name
    )
print(return_dict)

exit(0)


'incident'
'reflection'
'transmission'