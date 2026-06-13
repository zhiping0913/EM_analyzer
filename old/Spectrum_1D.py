import sdf_helper
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp
from jax.scipy.ndimage import map_coordinates
import numpy as np
from typing import Optional
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import scipy.constants as C
import os
import math
from scipy.signal import hilbert, find_peaks, peak_widths
from numpy.fft import fft, fftshift,ifft,ifftshift,fftfreq
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.colors import BASE_COLORS
from matplotlib.axes import Axes
from scipy.interpolate import interp1d
from jax.scipy.special import erf  
import xarray as xr
import time
from EM_analyzer.read_write import read_sdf, read_nc
from plot.plot_basic import savefig
from plot.plot_1D import plot_multiple_1D_fields, plot_twinx
theta_degree=45
theta_rad=jnp.radians(theta_degree)

laser_lambda = 0.8*C.micron		# Laser wavelength
laser_f0=1/laser_lambda
laser_k0=2*C.pi*laser_f0
laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_period=laser_lambda/C.speed_of_light
laser_a0=1		# Laser field strength
laser_Bc=(C.m_e*laser_omega0)/(C.elementary_charge)
laser_Ec=laser_Bc*C.speed_of_light   #4.013376e+12V/m
laser_amp=laser_a0*laser_Ec
laser_FWHM=8*C.femto   #The full width at half maximum of the intensity.
laser_tau=laser_FWHM/jnp.sqrt(2*jnp.log(2)) 
#laser_tau=laser_period/jnp.sqrt(jnp.pi)
laser_Nc=laser_omega0**2*C.m_e*C.epsilon_0/C.elementary_charge**2
laser_S0=C.epsilon_0*C.speed_of_light*laser_amp**2/2   #average Poynting vector

target_N=200

cells_per_lambda =2000
vacuum_length_x_lambda=20   #lambda
continuation_length_lambda=2000   #lambda
space_length_lambda=vacuum_length_x_lambda+2*continuation_length_lambda   #lambda
n_field_x=round(vacuum_length_x_lambda*cells_per_lambda)
n_continuation_x=round(space_length_lambda*cells_per_lambda)

d_x=laser_lambda/cells_per_lambda   #unit: m
d_f=1/(space_length_lambda*laser_lambda)   #unit: 1/m, d_x*d_f=1/n_continuation_x

print('λ0/dx=',laser_lambda/d_x)
print('f0/df=',laser_f0/d_f)

laser_lambda_M=laser_lambda/jnp.cos(theta_rad)
laser_tau_M=laser_tau/jnp.cos(theta_rad)
laser_f0_M=laser_f0*jnp.cos(theta_rad)
laser_k0_M=laser_k0*jnp.cos(theta_rad)
laser_Bc_M=laser_Bc*jnp.cos(theta_rad)
laser_Ec_M=laser_Ec*jnp.cos(theta_rad)
laser_amp_M=laser_amp*jnp.cos(theta_rad)
laser_S0_M=laser_S0*jnp.cos(theta_rad)**2
vacuum_length_x_lambda_M=vacuum_length_x_lambda*jnp.cos(theta_rad)
space_length_lambda_M=space_length_lambda*jnp.cos(theta_rad)   #laser_f0_M/d_f

laser_spectrum_peak_M=laser_amp_M*(jnp.sqrt(C.pi)/2)*(laser_tau_M*C.speed_of_light)*(1-jnp.exp(-laser_k0_M**2*(laser_tau_M*C.speed_of_light)**2))
laser_energy_M=laser_amp_M**2*jnp.sqrt(C.pi/2)*(laser_tau_M*C.speed_of_light/2)*(1-jnp.exp(-laser_k0_M**2*(laser_tau_M*C.speed_of_light)**2/2))
laser_envelope_integral_M=laser_amp_M**2*(laser_tau_M*C.speed_of_light)*jnp.sqrt(C.pi/2)

plasma_cutoff_order=jnp.sqrt(target_N/(jnp.cos(theta_rad))**3)

highest_harmonic=100000


working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface/a0=60/1D/45/ND_a0=0.30'

grid_x_axis=jnp.linspace(start=0,stop=vacuum_length_x_lambda*laser_lambda,num=n_field_x)+d_x/2   #x_M. unit:m. left or right
grid_x_L_axis=grid_x_axis*jnp.cos(theta_rad)
grid_x=jnp.meshgrid(grid_x_axis,indexing='ij')[0]
freq_x_axis=jnp.fft.fftshift(jnp.fft.fftfreq(n=n_continuation_x,d=d_x))   #unit: 1/m
freq_x=jnp.meshgrid(freq_x_axis,indexing='ij')[0]
k_x=2*C.pi*freq_x
freq_radius=jnp.abs(freq_x)
freq_mask=(freq_x/laser_f0_M>1)&(freq_x/laser_f0_M<highest_harmonic)

grid_center_mask=jnp.s_[round((n_continuation_x-n_field_x)/2):round((n_continuation_x+n_field_x)/2)]


filter=0
#freq_x_axis[round(n_continuation_x//2+space_length_lambda_M)]≈laser_f0_M


def linear(x, m, c):
    return m * x + c

def sine_edge(x:jnp.array):
    return jnp.piecewise(
        x,
        [x < -0.5, (x >= -0.5) & (x <= 0.5), x > 0.5],
        [0, lambda t: (1 + jnp.sin(jnp.pi * t)) / 2, 1]
    )


laser_kn=laser_k0

def reverse_field(Electric_Field:jnp.ndarray,name=''):
    assert Electric_Field.shape==(n_field_x,)
    Electric_Field_reverse=jnp.flip(Electric_Field)
    return Electric_Field_reverse

def square_integral_field_1D(Field:jnp.ndarray,d_x=1,complex_array=False):
    assert Field.ndim==1
    if complex_array:
        Field=jnp.asarray(Field,dtype=jnp.complex128)
        square_integral=jnp.real(jnp.einsum('i,i->',Field,jnp.conjugate(Field)))*d_x
    else:
        Field=jnp.asarray(Field,dtype=jnp.float64)
        square_integral=jnp.einsum('i,i->',Field,Field)*d_x
    square_integral=float(square_integral)
    print('∫|Field|^2×dx=%e' %(square_integral))
    return square_integral

def plot_field(
    Electric_Field:jnp.ndarray,
    ax:Optional[Axes]=None,
    grid_x_axis=grid_x_axis,
    alpha=1.0,c='blue',label='field',linestyle='-',linewidth=2,
    return_ax=False,name=''):
    assert Electric_Field.ndim==1
    assert grid_x_axis.ndim==1
    assert Electric_Field.size==grid_x_axis.size
    if ax is None:
        fig,ax = plt.subplots()
    ax.plot(grid_x_axis/laser_lambda_M,Electric_Field/laser_Ec_M,alpha=alpha,c=c,label=label,linewidth=linewidth,linestyle=linestyle)
    ax.set_xlabel('x_L/λ0')
    ax.set_ylabel('a=E/Ec')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5*laser_a0,1.5*laser_a0)
    ax.set_title('Electric Field %s' %(name))
    if return_ax:
        return ax
    else:
        savefig(fig,fig_path=os.path.join(working_dir,'Electric_Field_%s.png' %(name)))


def smooth_edge_1D(Field:jnp.ndarray,mask:jnp.ndarray,edge_length=100):
    n_x=Field.size
    assert mask.size==n_x
    threshold=1e-3
    x_id=jnp.arange(n_x)
    mask=jnp.ones(shape=(n_x,),dtype=jnp.int8)*mask
    sum_x=jnp.sum(mask,axis=0)
    x_min_id=jnp.where(sum_x<threshold, 2*n_x-1, jnp.argmax(mask,axis=0))   #x_min_id=2*n_x-1 if all items in mask is 0, else, x_min_id=the index of the first non-zero item
    x_max_id=jnp.where(sum_x<threshold, -n_x, n_x-1-jnp.argmax(mask[::-1],axis=0))
    x_left_trans=sine_edge((x_id-x_min_id-edge_length/2)/edge_length)
    x_right_trans=sine_edge((x_max_id-edge_length/2-x_id)/edge_length)
    mask_smooth=x_left_trans+x_right_trans-1
    mask_smooth=mask_smooth*(mask_smooth>threshold)
    Field_smooth=Field*mask_smooth
    print(square_integral_field_1D(Field_smooth)/square_integral_field_1D(Field))
    return Field_smooth
    
    

def continuation_field(Electric_Field:jnp.ndarray,n_continuation_x=n_continuation_x,edge_lambda=2,name=''):
    """
        Extend the field array to a shape (n_continuation_x,) for further analysis. The edge of the field array is reduced to 0 to avoid the noice at the edge.
        edge_lambda: the length of the smoothing area at the edge. unit: laser_lambda
    """
    n_x,=Electric_Field.shape
    assert n_x<=n_continuation_x
    grid_center_mask=np.s_[round((n_continuation_x-n_x)/2):round((n_continuation_x+n_x)/2)]
    x_id=np.arange(n_x)
    n_edge_x=edge_lambda*cells_per_lambda
    x_left_trans=0.5 * (1 + erf((x_id - n_edge_x) /n_edge_x))
    x_right_trans= 0.5 * (1 - erf((x_id - (n_x-1-n_edge_x)) / n_edge_x))   #smooth the edge
    Electric_Field_continuation=np.zeros(shape=(n_continuation_x))
    Electric_Field_continuation[grid_center_mask]=Electric_Field*x_left_trans*x_right_trans
    return Electric_Field_continuation
    
def square_integral_field(Field:jnp.ndarray,d_x=d_x,complex_array=False):
    assert Field.ndim==1
    if complex_array:
        square_integral=jnp.real(jnp.einsum('i,i->',Field,jnp.conjugate(Field)))*d_x
    else:
        square_integral=jnp.einsum('i,i->',Field,Field)*d_x
    print(square_integral/laser_energy_M)
    return square_integral

def get_polarisation(Electric_Field_Ey:jnp.ndarray,Electric_Field_Ez:jnp.ndarray,direction=1,name=''):
    """
        direction: direction>0 means the field travels in +x direction. direction<0 means the field travels in -x direction
    """
    assert Electric_Field_Ey.shape==(n_field_x,)
    assert Electric_Field_Ez.shape==(n_field_x,)
    assert direction!=0
    Electric_Field_Ey_envelope=get_envelope(Electric_Field=Electric_Field_Ey,name=name)['Electric_Field_envelope']
    Electric_Field_Ez_envelope=get_envelope(Electric_Field=Electric_Field_Ez,name=name)['Electric_Field_envelope']
    Electric_Field_envelope_square=jnp.square(Electric_Field_Ey_envelope)+jnp.square(Electric_Field_Ez_envelope)
    Electric_Field_envelope_square_max_id=jnp.argmax(Electric_Field_envelope_square)
    polarisation_field=jnp.sign(direction)*jnp.atan2(Electric_Field_Ez,Electric_Field_Ey)
    Electric_Field_Ey_energy=square_integral_field(Field=Electric_Field_Ey,d_x=d_x)
    Electric_Field_Ez_energy=square_integral_field(Field=Electric_Field_Ez,d_x=d_x)
    print('Energy in y polarisation: %f theoretical total energy' %(Electric_Field_Ey_energy/laser_energy_M))
    print('Energy in z polarisation: %f theoretical total energy' %(Electric_Field_Ez_energy/laser_energy_M))
    print('arctan(√(Energy_z/Energy_y))=%f×π' %(jnp.atan2(jnp.sqrt(Electric_Field_Ez_energy),jnp.sqrt(Electric_Field_Ey_energy))/C.pi))
    print('polarisation at the center of the pulse: %f×π' %(polarisation_field[Electric_Field_envelope_square_max_id]/C.pi))
    plt.plot(grid_x_axis/laser_lambda_M,polarisation_field/C.pi)
    plt.xlim(grid_x_axis[Electric_Field_envelope_square_max_id]/laser_lambda_M-1,grid_x_axis[Electric_Field_envelope_square_max_id]/laser_lambda_M+1)
    plt.xlabel('x_L/λ0')
    plt.ylabel('polarisation/π')
    plt.title('polarisation')
    plt.savefig(os.path.join(working_dir,'polarisation_%s.png' %(name)))
    plt.clf()


def filter_spectrum(Electric_Field_spectrum:jnp.ndarray,filter_range=(1.5,highest_harmonic),name=''):
    assert Electric_Field_spectrum.shape==(n_continuation_x,)
    frequency_mask=(freq_radius/laser_f0_M>filter_range[0])&(freq_radius/laser_f0_M<filter_range[1])
    Electric_Field_filter_spectrum=Electric_Field_spectrum*frequency_mask
    Electric_Field_filter=get_field_from_spectrum(Field_spectrum=Electric_Field_filter_spectrum)
    Electric_Field_filter_efficiency=square_integral_field(Field=Electric_Field_filter_spectrum,d_x=d_f,complex_array=True)/laser_energy_M
    #print('total energy: %f theoretical total energy' %(jnp.sum(Electric_Field_spectrum_square)*d_f/laser_energy_M))
    print('filter energy: %f theoretical total energy' %(Electric_Field_filter_efficiency))
    return Electric_Field_filter,Electric_Field_filter_spectrum,float(Electric_Field_filter_efficiency)



def filter_field(Electric_Field:jnp.ndarray,filter_range=(1.5,highest_harmonic),name=''):
    assert Electric_Field.shape==(n_field_x,)
    Electric_Field_spectrum,Electric_Field_spectrum_square=get_spectrum(Field=Electric_Field,name=name)
    Electric_Field_filter,Electric_Field_filter_spectrum,Electric_Field_filter_efficiency=filter_spectrum(Electric_Field_spectrum=Electric_Field_spectrum,filter_range=filter_range,name=name)
    Electric_Field_filter_spectrum_square=jnp.square(jnp.abs(Electric_Field_filter_spectrum))
    spectrum_center_radius=jnp.average(a=freq_radius,weights=Electric_Field_filter_spectrum_square)
    spectrum_width=jnp.sqrt(jnp.average(a=jnp.square(freq_radius),weights=Electric_Field_filter_spectrum_square)-jnp.square(spectrum_center_radius))
    print('spectrum_center_radius %f×f0' %(spectrum_center_radius/laser_f0_M))
    print('spectrum width f_std=%f(m^-1)' %(spectrum_width))
    #print('FWHM duration τ=sqrt(2ln2)/(2*pi*c*f_std)=%ffs' %(jnp.sqrt(2*jnp.log(2))/(2*C.pi*spectrum_width*C.speed_of_light)/C.femto))
    print('spectrum peak: %f theoretical spectrum peak' %(jnp.sqrt(jnp.max(Electric_Field_filter_spectrum_square))/laser_spectrum_peak_M))
    Electric_Field_filter_envelope_dict=get_envelope(Electric_Field_filter,name=name)
    Electric_Field_filter_envelope=Electric_Field_filter_envelope_dict['Electric_Field_envelope']
    Electric_Field_filter_envelope_max=Electric_Field_filter_envelope_dict['Electric_Field_envelope_max']
    Electric_Field_filter_envelope_max_id=Electric_Field_filter_envelope_dict['Electric_Field_envelope_max_id']
    Electric_Field_filter_envelope_peak_width=Electric_Field_filter_envelope_dict['Electric_Field_envelope_peak_width']
    plot_twinx(
        coordinate=grid_x_axis/laser_lambda_M,
        field_dict_list_1=[{'field':Electric_Field/laser_Ec_M,'linestyle':'-','label':'Total'}],
        field_dict_list_2=[
            {'field':Electric_Field_filter/laser_Ec_M,'linestyle':'-','label':'Filter'},
            {'field':Electric_Field_filter_envelope/laser_Ec_M,'linestyle':'--','label':'Filter envelope','color':'green'},
            {'field':-Electric_Field_filter_envelope/laser_Ec_M,'linestyle':'--','label':'Filter envelope','color':'green'},
            ],
        color_1='red',color_2='blue',
        xlabel='x_L/λ0',
        xmin=grid_x_axis[Electric_Field_filter_envelope_max_id]/laser_lambda_M-1.5,xmax=grid_x_axis[Electric_Field_filter_envelope_max_id]/laser_lambda_M+1.5,
        #xmin=34.5,xmax=36.5,
        y1_label='Total a=E/Ec',y1_min=-1.3*laser_a0,y1_max=1.3*laser_a0,
        y2_label='Filter a=E/Ec',y2_min=-1.2*Electric_Field_filter_envelope_max/laser_Ec_M,y2_max=1.2*Electric_Field_filter_envelope_max/laser_Ec_M,
        return_ax=False,name=f'Field_and_filter_{name}',working_dir=working_dir,
    )
    return {
        'Electric_Field_filter':Electric_Field_filter,
        'Electric_Field_filter_spectrum':Electric_Field_filter_spectrum,
        'Electric_Field_filter_efficiency':Electric_Field_filter_efficiency,
    }


def get_analytic(Field:jnp.ndarray):
    assert Field.shape==(n_field_x,)
    Field_continuation=continuation_field(Electric_Field=Field,n_continuation_x=n_continuation_x)
    Field_analytic=hilbert(Field_continuation)[grid_center_mask]
    return Field_analytic

def get_envelope(Electric_Field:jnp.ndarray,direction=1,grid_x_axis=grid_x_axis,name=''):
    """
        direction: direction>0 means the field travels in +x direction. direction<0 means the field travels in -x direction
    """
    assert direction!=0
    assert Electric_Field.size==grid_x_axis.size
    grid_x=grid_x_axis
    Electric_Field_analytic=get_analytic(Electric_Field=Electric_Field)
    Electric_Field_envelope=jnp.abs(Electric_Field_analytic)
    Electric_Field_analytic_phase=jnp.sign(direction)*jnp.angle(Electric_Field_analytic)
    Electric_Field_envelope_square=jnp.square(Electric_Field_envelope)
    Electric_Field_envelope_x_1_moment=jnp.average(a=grid_x,weights=Electric_Field_envelope_square)
    Electric_Field_envelope_x_2_moment=jnp.average(a=jnp.square(grid_x),weights=Electric_Field_envelope_square)
    Electric_Field_envelope_x_std=jnp.sqrt(Electric_Field_envelope_x_2_moment-jnp.square(Electric_Field_envelope_x_1_moment))
    print('Lab frame average position of the envelope: %fλ0' %(Electric_Field_envelope_x_1_moment/laser_lambda_M))
    print('Lab frame FWHM obtained from envelope_x_std: %ffs'%(Electric_Field_envelope_x_std*2*jnp.sqrt(2*jnp.log(2))*jnp.cos(theta_rad)/C.speed_of_light/C.femto))
    Electric_Field_envelope_max=jnp.max(Electric_Field_envelope[1:-1])
    Electric_Field_envelope_max_id=jnp.where(Electric_Field_envelope==Electric_Field_envelope_max)[0].item()
    print(Electric_Field_envelope_max_id)
    print(Electric_Field_envelope_max/laser_amp_M)
    Electric_Field_envelope_max_phase=Electric_Field_analytic_phase[Electric_Field_envelope_max_id]
    print('Phase at the peak: %fπ' %(Electric_Field_envelope_max_phase/jnp.pi))
    Electric_Field_analytic_phase_unwrap=jnp.unwrap(Electric_Field_analytic_phase,period=2*jnp.pi)
    Electric_Field_analytic_phase_unwrap=Electric_Field_analytic_phase_unwrap-Electric_Field_analytic_phase_unwrap[Electric_Field_envelope_max_id]+Electric_Field_envelope_max_phase   #Phase relative to the peak. Keep the phase at the peak
    Electric_Field_envelope_peak=jnp.asarray(peak_widths(x=Electric_Field_envelope,peaks=[Electric_Field_envelope_max_id],rel_height=1-jnp.sqrt(2)/2),dtype=jnp.float64)  #[width,width_height,left_id,right_id]
    x_left,x_right=map_coordinates(input=grid_x_axis,coordinates=[Electric_Field_envelope_peak[-2:]],order=1)   #unit: m. In moving frame
    Electric_Field_envelope_peak_width=d_x*Electric_Field_envelope_peak[0].item()   #unit: m. In moving frame
    print('Lab frame FWHM obtained from envelope:%ffs' %(Electric_Field_envelope_peak_width*jnp.cos(theta_rad)/C.speed_of_light/C.femto))
    print('peak at x_L/λ0=%f' %(grid_x_axis[Electric_Field_envelope_max_id]/laser_lambda_M))
    return {
        'Electric_Field_envelope':Electric_Field_envelope,   #len=grid_x_axis.size
        'Electric_Field_analytic_phase':Electric_Field_analytic_phase,   #len=grid_x_axis.size
        'Electric_Field_envelope_max':float(Electric_Field_envelope_max),
        'Electric_Field_envelope_max_id':Electric_Field_envelope_max_id,
        'Electric_Field_envelope_max_position':float(grid_x_axis[Electric_Field_envelope_max_id]),
        'Electric_Field_envelope_max_phase':float(Electric_Field_envelope_max_phase),
        'Electric_Field_envelope_peak_width':float(Electric_Field_envelope_peak_width),
    }
    fig = plt.figure(figsize=(8, 6))
    ax=fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax = plot_field(Electric_Field,ax=ax,grid_x_axis=grid_x_axis,c='blue',label='field',return_ax=True)
    ax = plot_field(Electric_Field_envelope,ax=ax,grid_x_axis=grid_x_axis,c='g',label='envelope',linestyle='--',linewidth=1,return_ax=True)
    ax = plot_field(-Electric_Field_envelope,ax=ax,grid_x_axis=grid_x_axis,c='g',label=None,linestyle='--',linewidth=1,return_ax=True)
    ax.hlines(y=Electric_Field_envelope_peak[1]/laser_Ec_M,xmin=x_left/laser_lambda_M,xmax=x_right/laser_lambda_M,colors='r',linestyles='--',linewidth=1,label=f'FWHM Duration={Electric_Field_envelope_peak_width/laser_lambda_M:.4f}·T0')
    ax.hlines(y=-Electric_Field_envelope_peak[1]/laser_Ec_M,xmin=x_left/laser_lambda_M,xmax=x_right/laser_lambda_M,colors='r',linestyles='--',linewidth=1)
    #ax.set_xlim(34.5,36.5)
    #ax.set_xlim(grid_x_axis[Electric_Field_envelope_max_id]/laser_lambda_M-0.4,grid_x_axis[Electric_Field_envelope_max_id]/laser_lambda_M+0.4)
    ax.set_ylim(-1.5*laser_a0,1.5*laser_a0)
    ax.legend(loc='upper right')
    ax.set_title(name)
    savefig(fig,fig_path=os.path.join(working_dir,'Electric_Field_envelope_%s.png' %(name)))


def get_dc(Field:jnp.ndarray,name=''):
    assert Field.shape==(n_field_x,)
    Field_square=jnp.square(Field)
    Field_square_spectrum,_=get_spectrum(Field=Field_square,name=name)
    Field_square_dc_spectrum=2*Field_square_spectrum*(freq_radius/laser_f0_M<0.5)
    Field_square_dc=jnp.abs(get_field_from_spectrum(Field_spectrum=Field_square_dc_spectrum,name=name))
    Field_dc=jnp.sqrt(Field_square_dc)
    print(square_integral_field(Field_dc)/laser_envelope_integral_M)
    return Field_dc
    


def get_energy_flux(Electric_Field:jnp.ndarray,Magnetic_Field:jnp.ndarray,name=''):
    assert Electric_Field.shape==(n_field_x,)
    assert Magnetic_Field.shape==(n_field_x,)
    S=Electric_Field*Magnetic_Field/C.mu_0
    S_continuation=continuation_field(S)
    S_spectrum=jnp.fft.fftshift(jnp.fft.fft(S_continuation))
    S_spectrum_square=jnp.square(jnp.abs(S_spectrum))
    S_spectrum_filter=S_spectrum*(freq_radius/laser_f0<1)
    S_average=jnp.real(jnp.fft.ifft(jnp.fft.ifftshift(S_spectrum_filter)))[0:n_field_x]
    S_average_max=jnp.max(jnp.abs(S_average))
    print(S_average_max/laser_S0_M)
    print(jnp.where(jnp.abs(S_average)==S_average_max))
    return S_average
    #plt.plot(grid_x_axis/laser_lambda_M,S/laser_S0_M,c='b',label='S',linewidth=1)
    plt.plot(grid_x_axis/laser_lambda_M,S_average/laser_S0_M,c='r',label='<S>',linewidth=2)
    #plt.xlim(6.6,7.4)
    plt.ylim(-0.002,0.002)
    plt.legend()
    plt.xlabel(xlabel='x_L/λ0')
    plt.ylabel(ylabel='S(S0)')
    plt.title(label='Energy_flux_Sx')
    plt.savefig(os.path.join(working_dir,'Energy_flux_Sx_%s.png' %(name)))
    plt.clf()
    
    return S_average

def get_power(Electric_Field_spectrum:jnp.ndarray,name=''):
    assert Electric_Field_spectrum.shape==(n_continuation_x,)
    Electric_Field_spectrum_square=jnp.square(jnp.abs(Electric_Field_spectrum))
    m, c = curve_fit(linear, jnp.log(freq_x[freq_mask]), jnp.log(Electric_Field_spectrum_square[freq_mask]))[0]
    print(m,c)
    Electric_Field_spectrum_square_fit=jnp.exp(linear(jnp.log(freq_x),m,c))
    plt.loglog(freq_x_axis[n_continuation_x//2:]/laser_f0_M,Electric_Field_spectrum_square[n_continuation_x//2:]/laser_spectrum_peak_M**2,label='spectrum')
    plt.loglog(freq_x[n_continuation_x//2:]/laser_f0_M,Electric_Field_spectrum_square_fit[n_continuation_x//2:]/laser_spectrum_peak_M**2,label='fit, -p=%0.2f'%(m),linestyle='--')
    plt.xlim(0.8,200)
    plt.ylim(1e-8,1)
    plt.xlabel(xlabel='k/k0')
    plt.ylabel(ylabel='I(k)/I0')
    plt.legend()
    plt.title(label='Spectrum')
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_square_fit_%s.png' %(name)))
    plt.clf()

def get_phase_for_harmonic_n(Electric_Field:jnp.ndarray,harmonic_n=1,name=''):
    """
    Assume the field is in +x direction. 
    Args:
        Electric_Field (jnp.ndarray): _description_
        harmonic_n (int, optional): _description_. Defaults to 1.
        name (str, optional): _description_. Defaults to ''.

    Returns:
        _type_: _description_
    """
    assert Electric_Field.shape==(n_field_x,)
    Electric_Field_envelope_dict=get_envelope(Electric_Field,name=name)
    Electric_Field_envelope_max_id=Electric_Field_envelope_dict['Electric_Field_envelope_max_id']
    filter_field_dict=filter_field(Electric_Field,filter_range=(harmonic_n-0.5,harmonic_n+0.5),name=f'{name}_{harmonic_n}')
    Electric_Field_n=filter_field_dict['Electric_Field_filter']
    Electric_Field_n_envelope_dict=get_envelope(Electric_Field_n,name=name)
    Electric_Field_n_phase=Electric_Field_n_envelope_dict['Electric_Field_analytic_phase']
    Electric_Field_n_phase_at_peak=Electric_Field_n_phase[Electric_Field_envelope_max_id]
    print(f'Phase of {harmonic_n}th harmonic at the peak of the total field: {Electric_Field_n_phase_at_peak} rad, {Electric_Field_n_phase_at_peak/jnp.pi} π')
    return Electric_Field_n_phase_at_peak



def get_phase(Electric_Field:jnp.ndarray,harmonic_order=1,name=''):
    """
    Assume the field is in +x direction. 
    Args:
        Electric_Field (jnp.ndarray): _description_
        harmonic_order (int, optional): _description_. Defaults to 1.
        name (str, optional): _description_. Defaults to ''.

    Returns:
        _type_: _description_
    """
    assert Electric_Field.shape==(n_field_x,)
    kn_M=harmonic_order*laser_k0_M
    Electric_Field_spectrum,Electric_Field_spectrum_square=get_spectrum(Field=Electric_Field,name=name)
    harmonic_order_th_mask=jnp.arange(n_continuation_x)[round(n_continuation_x/2+(harmonic_order-0.5)*space_length_lambda_M):round(n_continuation_x/2+(harmonic_order+0.5)*space_length_lambda_M)]
    harmonic_order_th_id=harmonic_order*space_length_lambda_M
    freq_x_order_th=freq_x[harmonic_order_th_mask]
    k_x_order_th=k_x[harmonic_order_th_mask]
    Electric_Field_spectrum_order_th=Electric_Field_spectrum[harmonic_order_th_mask]
    Electric_Field_spectrum_order_th_square=Electric_Field_spectrum_square[harmonic_order_th_mask]
    Electric_Field_spectrum_phase=jnp.unwrap(jnp.angle(Electric_Field_spectrum),period=2*jnp.pi)
    Electric_Field_spectrum_phase=Electric_Field_spectrum_phase-Electric_Field_spectrum_phase[n_continuation_x//2]
    Electric_Field_spectrum_phase_order_th=Electric_Field_spectrum_phase[harmonic_order_th_mask]
    Electric_Field_spectrum_group_delay=jnp.gradient(Electric_Field_spectrum_phase,k_x)   #dφ/dk,unit: m
    Electric_Field_spectrum_group_delay_plus_fn=Electric_Field_spectrum_group_delay[round(n_continuation_x/2+harmonic_order_th_id)]
    Electric_Field_spectrum_group_delay_minus_fn=Electric_Field_spectrum_group_delay[round(n_continuation_x/2-harmonic_order_th_id)]
    Electric_Field_spectrum_group_delay_position_0_fn=-jnp.average(Electric_Field_spectrum_group_delay[harmonic_order_th_mask])
    Electric_Field_spectrum_group_delay_order_th=Electric_Field_spectrum_group_delay[harmonic_order_th_mask]

    Electric_Field_spectrum_order_th_square_max=jnp.max(Electric_Field_spectrum_order_th_square)
    Electric_Field_spectrum_order_th_square_max_id=jnp.argmax(Electric_Field_spectrum_order_th_square)
    print(freq_x_order_th[Electric_Field_spectrum_order_th_square_max_id]/laser_f0_M)
    Electric_Field_spectrum_order_th_square_peak=peak_widths(x=Electric_Field_spectrum_order_th_square,peaks=[Electric_Field_spectrum_order_th_square_max_id],rel_height=0.5)
    Electric_Field_spectrum_group_delay_order_th_coefficients=jnp.polyfit(
        x=k_x_order_th[round(Electric_Field_spectrum_order_th_square_peak[2].item()):round(Electric_Field_spectrum_order_th_square_peak[3].item())],
        y=Electric_Field_spectrum_group_delay_order_th[round(Electric_Field_spectrum_order_th_square_peak[2].item()):round(Electric_Field_spectrum_order_th_square_peak[3].item())],
        deg=2
        )
    Electric_Field_spectrum_group_delay_order_th_fit=jnp.poly1d(c_or_r=Electric_Field_spectrum_group_delay_order_th_coefficients,r=False)(k_x_order_th)
    Electric_Field_spectrum_group_delay_order_th_average=jnp.average(
        a=Electric_Field_spectrum_group_delay_order_th[round(Electric_Field_spectrum_order_th_square_peak[2].item()):round(Electric_Field_spectrum_order_th_square_peak[3].item())],
        #weights=Electric_Field_spectrum_square[round(Electric_Field_spectrum_order_th_square_peak[2].item()):round(Electric_Field_spectrum_order_th_square_peak[3].item())]
        )   #unit: m
    print('Lab frame %d th harmonic at position %fλ0(average)' %(harmonic_order,vacuum_length_x_lambda_M/2+Electric_Field_spectrum_group_delay_order_th_average/laser_lambda_M))
    print('Lab frame %d th harmonic at position %fλ0(center)' %(harmonic_order,vacuum_length_x_lambda_M/2-Electric_Field_spectrum_group_delay_position_0_fn/laser_lambda_M))
    print('Lab frame %d th harmonic at position %fλ0(peak)' %(harmonic_order,vacuum_length_x_lambda_M/2+Electric_Field_spectrum_group_delay_order_th[Electric_Field_spectrum_order_th_square_max_id]/laser_lambda_M))
    Electric_Field_spectrum_intrinsic_phase=Electric_Field_spectrum_phase+k_x*Electric_Field_spectrum_group_delay_position_0_fn
    Electric_Field_spectrum_intrinsic_group_delay=Electric_Field_spectrum_group_delay+Electric_Field_spectrum_group_delay_position_0_fn
    Electric_Field_spectrum_intrinsic_phase_plus_fn=Electric_Field_spectrum_intrinsic_phase[round(n_continuation_x/2+harmonic_order_th_id)]
    Electric_Field_spectrum_intrinsic_phase_minus_fn=Electric_Field_spectrum_intrinsic_phase[round(n_continuation_x/2-harmonic_order_th_id)]
    Electric_Field_spectrum_intrinsic_phase_0_fn=((((Electric_Field_spectrum_intrinsic_phase_plus_fn-Electric_Field_spectrum_intrinsic_phase_minus_fn)/2)/jnp.pi) % 2)*jnp.pi
    Electric_Field_spectrum_intrinsic_phase_order_th=Electric_Field_spectrum_intrinsic_phase[harmonic_order_th_mask]
    Electric_Field_spectrum_intrinsic_phase_order_th_0=Electric_Field_spectrum_intrinsic_phase_order_th[round(0.5*space_length_lambda_M)]
    #Electric_Field_spectrum_intrinsic_phase_order_th=Electric_Field_spectrum_intrinsic_phase_order_th-Electric_Field_spectrum_intrinsic_phase_order_th_0
    print('φn0=%fπ' %(Electric_Field_spectrum_intrinsic_phase_0_fn/jnp.pi))
    print('group_delay dφ(k)/dk=α*(k/kn)^2+β*(k/kn)+x0, kn=%d*k0' %(harmonic_order))
    print('Lab frame x0=%fλ0' %(Electric_Field_spectrum_group_delay_order_th_coefficients[2]/laser_lambda_M))
    print('β=%f' %(Electric_Field_spectrum_group_delay_order_th_coefficients[1]*kn_M))
    print('α=%f' %(Electric_Field_spectrum_group_delay_order_th_coefficients[0]*kn_M**2))
    print('φ_intrinsic_kn+-φ_intrinsic_kn-=%fπ' %((Electric_Field_spectrum_intrinsic_phase_order_th[round(Electric_Field_spectrum_order_th_square_peak[3].item())]-Electric_Field_spectrum_intrinsic_phase_order_th[round(Electric_Field_spectrum_order_th_square_peak[2].item())])/jnp.pi))

    plot_twinx(
        coordinate=freq_x/laser_f0_M,
        field_dict_list_1=[{'field': Electric_Field_spectrum_square/laser_spectrum_peak_M**2}],
        field_dict_list_2=[{'field': Electric_Field_spectrum_group_delay/laser_lambda_M}],
        xmin=0,
        xmax=60,
        xlabel='k/k0',
        y1_label='I(k)/I0',
        y1_scale='log',
        y1_min=1e-6,
        y1_max=1.1,
        y2_label='(dφ/dk)/λ0',
        y2_scale='linear',
        y2_min=(Electric_Field_spectrum_group_delay_order_th_average/laser_lambda_M-1),
        y2_max=(Electric_Field_spectrum_group_delay_order_th_average/laser_lambda_M+1),
        name='Spectrum Group delay %s' %(name),working_dir=working_dir,return_ax=False,
    )
    plot_twinx(
        coordinate=freq_x/laser_f0_M,
        field_dict_list_1=[{'field': Electric_Field_spectrum_square/laser_spectrum_peak_M**2}],
        field_dict_list_2=[{'field': Electric_Field_spectrum_intrinsic_phase/jnp.pi}],
        xmin=0,
        xmax=60,
        xlabel='k/k0',
        y1_label='I(k)/I0',
        y1_scale='log',
        y1_min=1e-6,
        y1_max=1.1,
        y2_label='φ/π',
        y2_scale='linear',
        y2_min=(Electric_Field_spectrum_intrinsic_phase_order_th_0/jnp.pi-1/2),
        y2_max=(Electric_Field_spectrum_intrinsic_phase_order_th_0/jnp.pi+3),
        name='Spectrum Phase %s' %(name),working_dir=working_dir,return_ax=False,
    )

    return (Electric_Field_spectrum_group_delay_order_th_average/laser_lambda-continuation_length_lambda)*jnp.cos(theta_rad)
    
def get_spectrum(Field:jnp.ndarray,name=''):
    assert Field.shape==(n_field_x,) or Field.shape==(n_continuation_x,)
    if Field.shape==(n_field_x,):
        Field_continuation=continuation_field(Field)
    else:
        Field_continuation=Field
    Field_spectrum=jnp.fft.fftshift(jnp.fft.fft(a=fftshift(Field_continuation),n=n_continuation_x,axis=0))*d_x
    Field_spectrum_square=jnp.square(jnp.abs(Field_spectrum))
    print(freq_x_axis[jnp.where(Field_spectrum_square==jnp.max(Field_spectrum_square))[0]]/laser_f0_M)
    print('Total energy %e' %(jnp.sum(Field_spectrum_square)*d_f/laser_energy_M))
    #pd.DataFrame(data={'I(kx)/I0':Field_spectrum_square[n_continuation_x//2:]/laser_spectrum_peak_M**2,'kx/k0':freq_x_axis[n_continuation_x//2:]/laser_f0_M}).to_hdf(path_or_buf=os.path.join(working_dir,'spectrum.hdf5'),mode='a',key='Field_spectrum_square_%s' %(name))
    #return Field_spectrum,Field_spectrum_square
    plt.semilogy(freq_x_axis[n_continuation_x//2:]/laser_f0_M,Field_spectrum_square[n_continuation_x//2:]/laser_spectrum_peak_M**2,label=name)
    plt.axvline(x=jnp.sqrt(target_N),color='r',linestyle='--',linewidth=1,label='k=ωp/c')
    plt.xlim(0,20)
    plt.ylim(1e-6,1.1)
    plt.xlabel(xlabel='kx/k0')
    plt.ylabel(ylabel='I(kx)/I0')
    plt.title(label='Spectrum')
    plt.legend()
    plt.savefig(os.path.join(working_dir,'Field_spectrum_%s.png' %(name)))
    plt.clf()
    return Field_spectrum,Field_spectrum_square

def get_field_from_spectrum(Field_spectrum:jnp.ndarray,name=''):
    assert Field_spectrum.shape==(n_continuation_x,)
    Field_continuation=jnp.real(ifftshift(jnp.fft.ifft(jnp.fft.ifftshift(Field_spectrum))))*n_continuation_x*d_f
    Field=Field_continuation[grid_center_mask]
    return Field

    

def output_field(Electric_Field:jnp.ndarray,grid_axis:jnp.ndarray,name=''):
    """
        Electric_Field (jnp.ndarray): Fields on the grid in the moving frame with coordinate grid_x_L_axis=grid_x_axis*cos(θ)
        grid_axis (jnp.ndarray): Grid in the lab frame.
    """
    assert Electric_Field.shape==(n_field_x,)
    Electric_Field_L=Electric_Field/jnp.cos(theta_rad)
    Electric_Field_L_interp_func = interp1d(x=grid_x_L_axis, y=Electric_Field_L, kind='cubic', fill_value=0)
    Electric_Field_L_output=Electric_Field_L_interp_func(grid_axis)
    fields_ds=xr.Dataset(
        data_vars={
            name:(["x"], Electric_Field_L_output),
            },
        coords={'x':(["x"], grid_axis)}
        )
    fields_ds.to_netcdf(path=os.path.join(working_dir,'%s.nc' %(name)))
    return Electric_Field_L_output
    


def two_pulse_spectrum(Electric_Field_1:jnp.ndarray,Electric_Field_2:jnp.ndarray):
    Electric_Field_1_continuation=continuation_field(Electric_Field_1)
    Electric_Field_2_continuation=continuation_field(Electric_Field_2)
    Electric_Field_concatenate=jnp.concatenate((Electric_Field_1,Electric_Field_2),axis=0)
    Electric_Field_concatenate_continuation=continuation_field(Electric_Field_concatenate)
    _,Electric_Field_1_spectrum_square=get_spectrum(Electric_Field_1_continuation,name='Electric_Field_1')
    _,Electric_Field_2_spectrum_square=get_spectrum(Electric_Field_2_continuation,name='Electric_Field_2')
    _,Electric_Field_concatenate_spectrum_square=get_spectrum(Electric_Field_concatenate_continuation,name='Electric_Field_concatenate')
    plt.semilogy(freq_x_axis[n_continuation_x//2:]/laser_f0_M,Electric_Field_1_spectrum_square[n_continuation_x//2:]/laser_spectrum_peak_M**2,label='1',linestyle='--',c='b',linewidth=1)
    plt.semilogy(freq_x_axis[n_continuation_x//2:]/laser_f0_M,Electric_Field_2_spectrum_square[n_continuation_x//2:]/laser_spectrum_peak_M**2,label='2',linestyle='--',c='g',linewidth=1)
    plt.semilogy(freq_x_axis[n_continuation_x//2:]/laser_f0_M,Electric_Field_concatenate_spectrum_square[n_continuation_x//2:]/laser_spectrum_peak_M**2,label='concatenate',c='r')
    plt.xlim(5,10)
    plt.ylim(1e-6,1e-2)
    plt.xlabel(xlabel='k/k0')
    plt.ylabel(ylabel='I(k)/I0')
    plt.title(label='Spectrum')
    plt.legend()
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_continuation.png'))
    plt.clf()

working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Small_a0/test/without_collision'
Electric_Field_Ey=read_sdf(sdf_name=os.path.join(working_dir,'0020.sdf'),block_name_list=['Electric_Field_Ey'])['Electric_Field_Ey'][:n_field_x]
Electric_Field_Ey=reverse_field(Electric_Field_Ey)
filter_field_dict=filter_field(Electric_Field_Ey,filter_range=(2.5,20.5),name='reflection_[3,20]')
exit(0)
get_envelope(Electric_Field=Electric_Field_Ey,name='reflection')
get_spectrum(Field=Electric_Field_Ey,name='reflection')
exit(0)
Electric_Field_Ey=reverse_field(Electric_Field_Ey)
filter_field_dict=filter_field(Electric_Field_Ey,filter_range=(5.5,highest_harmonic),name='reflection_[6.+∞]')
exit(0)
#Electric_Field_Ey=read_nc(nc_name=os.path.join(working_dir,'0001_transmission_rotate_500cpl.nc'),key_name_list=['Electric_Field_Ey'])['Electric_Field_Ey'][:n_field_x,n_field_x//2]
data_dict=read_nc(nc_name=os.path.join(working_dir,'reflection_300cpl_clip.nc'),key_name_list=['Electric_Field_Ey','Magnetic_Field_Bz','x'])
Electric_Field_Ey=data_dict['Electric_Field_Ey'][0:n_field_x,n_field_x//2]
Magnetic_Field_Bz=data_dict['Magnetic_Field_Bz'][0:n_field_x,n_field_x//2]
grid_x_axis=data_dict['x']

Electric_Field_filter=filter_field(Electric_Field_Ey,(1.5,100),'2+')['Electric_Field_filter']
Electric_Field_envelope=get_envelope(Electric_Field_filter)['Electric_Field_envelope']
Electric_Field_dc=get_dc(Electric_Field_filter)
name='2_dc'
plt.plot(grid_x_axis/laser_lambda_M,Electric_Field_filter/laser_Ec_M,c='b',label='Electric_Field_filter',linewidth=2)
plt.plot(grid_x_axis/laser_lambda_M,Electric_Field_envelope/laser_Ec_M,linestyle='--',c='g',label='envelope',linewidth=1)
plt.plot(grid_x_axis/laser_lambda_M,Electric_Field_dc/laser_Ec_M,linestyle='--',c='r',label='dc',linewidth=1)
plt.legend()
plt.xlabel(xlabel='x_L/λ0')
plt.ylabel(ylabel='a=E/Ec')
plt.title(label='Field_%s' %(name))
plt.savefig(os.path.join(working_dir,'Electric_Field_envelope_%s.png' %(name)))
plt.clf()
exit(0)


'incident'
'reflection,reflected'
'transmission,transmitted'