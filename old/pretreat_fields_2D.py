import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/start')
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax.scipy.special import erf 
import numpy as np
import scipy.constants as C
import os
from joblib import Parallel, delayed
from scipy.ndimage import rotate, map_coordinates
import pandas as pd
import xarray as xr
from read_write import read_sdf,read_nc,read_dat

def sine_edge(x:jnp.array):
    return jnp.piecewise(
        x,
        [x < -0.5, (x >= -0.5) & (x <= 0.5), x > 0.5],
        [0, lambda t: (1 + jnp.sin(np.pi * t)) / 2, 1]
    )


#@profile
def square_integral_field_2D(Field:jnp.ndarray,d_x=1,d_y=1,complex_array=False):
    assert Field.ndim==2
    if complex_array:
        Field=jnp.asarray(Field,dtype=jnp.complex128)
        square_integral=jnp.real(jnp.einsum('ij,ij->',Field,jnp.conjugate(Field)))*d_x*d_y
    else:
        Field=jnp.asarray(Field,dtype=jnp.float64)
        square_integral=jnp.einsum('ij,ij->',Field,Field)*d_x*d_y
    square_integral=float(square_integral)
    print('∬|Field|^2×dx×dy=%e' %(square_integral))
    return square_integral
#@profile
def continue_field_2D(Field:jnp.ndarray,n_continuation_x:int,n_continuation_y:int,center_shift=(0,0),smooth=True,edge_length=100):
    """
        Extend the field array to a shape (n_continuation_x,n_continuation_y) for further analysis. The edge of the field array is reduced to 0 to avoid the noice at the edge.
        edge_length: int. the length (number of cells) of the smoothing area at the edge
    """
    n_x,n_y=Field.shape
    assert n_x<=n_continuation_x-2*jnp.abs(center_shift[0])
    assert n_y<=n_continuation_y-2*jnp.abs(center_shift[1])
    grid_center_mask=jnp.s_[round((n_continuation_x-n_x)/2)+center_shift[0]:round((n_continuation_x-n_x)/2)+n_x+center_shift[0],round((n_continuation_y-n_y)/2)+center_shift[1]:round((n_continuation_y-n_y)/2)+n_y+center_shift[1]]
    Field_continuation=jnp.zeros(shape=(n_continuation_x,n_continuation_y),dtype=Field.dtype)
    if smooth:
        x_id=jnp.arange(n_x)
        y_id=jnp.arange(n_y)
        x_left_trans=0.5 * (1 + erf((x_id - edge_length) /edge_length))
        x_right_trans= 0.5 * (1 - erf((x_id - (n_x-1-edge_length)) / edge_length))
        y_left_trans=0.5 * (1 + erf((y_id - edge_length) /edge_length))
        y_right_trans= 0.5 * (1 - erf((y_id - (n_y-1-edge_length)) / edge_length))   #smooth the edge
        Field_continuation=Field_continuation.at[grid_center_mask].set(jnp.einsum('ij,i,i,j,j->ij',Field,x_left_trans,x_right_trans,y_left_trans,y_right_trans))
    else:
        Field_continuation=Field_continuation.at[grid_center_mask].set(Field)
    return Field_continuation



def rotate_perpendicular_field_2D(Field_z:jnp.ndarray,angle=0.0):
    """
    The direction of the field is perpendicular to the plane of rotation (x-y plane).
    Note that: The output fields' shape is twice the input fields'. output.shape=2*input.shape=(2*input.shape[0],2*input.shape[1])
    Args:
        Field_z (jnp.ndarray): _description_
        angle: The angle from the wave's direction to +x direction.Unit: rad. Negative angle means that the wave vector is in the first or second quadrant.
    """
    n_x,n_y=Field_z.shape
    Field_z_rotate=rotate(input=Field_z,angle=jnp.rad2deg(angle),axes=(0,1),reshape=True,mode='constant',cval=0,order=3)
    Field_z_rotate_continuation=continue_field_2D(Field=Field_z_rotate,n_continuation_x=2*n_x,n_continuation_y=2*n_y,edge_length=max(n_x,n_y)//20)
    return Field_z_rotate_continuation

def rotate_xy(Field_x:jnp.ndarray,Field_y:jnp.ndarray,angle=0.0):
    """_summary_
    Note that: The output fields' shape is twice the input fields'. output.shape=2*input.shape=(2*input.shape[0],2*input.shape[1])
    Args:
        Field_x (jnp.ndarray): _description_
        Field_y (jnp.ndarray): _description_
        angle: The angle from the wave's direction to +x direction.Unit: rad. Negative angle means that the wave vector is in the first or second quadrant.

    Returns:
        _type_: _description_
    """
    assert Field_x.shape==Field_y.shape
    n_x,n_y=Field_x.shape
    Field_y_1=Field_x*jnp.sin(angle)+Field_y*jnp.cos(angle)
    Field_x_1=Field_x*jnp.cos(angle)-Field_y*jnp.sin(angle)
    Field_y_1_rotate=rotate(input=Field_y_1,angle=jnp.rad2deg(angle),axes=(0,1),reshape=True,mode='constant',cval=0,order=3)
    Field_x_1_rotate=rotate(input=Field_x_1,angle=jnp.rad2deg(angle),axes=(0,1),reshape=True,mode='constant',cval=0,order=3)
    Field_y_1_rotate_continuation=continue_field_2D(Field=Field_y_1_rotate,n_continuation_x=2*n_x,n_continuation_y=2*n_y,edge_length=max(n_x,n_y)//20)
    Field_x_1_rotate_continuation=continue_field_2D(Field=Field_x_1_rotate,n_continuation_x=2*n_x,n_continuation_y=2*n_y,edge_length=max(n_x,n_y)//20)
    return Field_x_1_rotate_continuation,Field_y_1_rotate_continuation


def rotate_parallel_field_2D(Field_x:jnp.ndarray,Field_y:jnp.ndarray,x_axis:jnp.ndarray,y_axis:jnp.ndarray,angle=0.0):
    """
    The direction of the field is in the plane of rotation (x-y plane).
    Rotate the field to change the direction of propagation to +x direction. 
    Transversal component
    longitudinal component
    Args:
        Field_x (jnp.ndarray): _description_
        Field_y (jnp.ndarray): _description_
        angle: The angle from the wave's direction to +x direction.Unit: rad. Negative angle means that the wave vector is in the first or second quadrant.

    """
    assert x_axis.ndim==1
    assert y_axis.ndim==1
    n_x=x_axis.size
    n_y=y_axis.size
    assert Field_x.shape==(n_x,n_y)
    assert Field_y.shape==(n_x,n_y)
    x,y=jnp.meshgrid(x_axis,y_axis,indexing='ij')
    d_x=x_axis[1]-x_axis[0]
    d_y=y_axis[1]-y_axis[0]
    Field_x_rotate_continuation,Field_y_rotate_continuation=rotate_xy(Field_x,Field_y,angle)
    x_rotate_continuation,y_rotate_continuation=rotate_xy(x,y,angle)
    Field_y_rotate_max=jnp.max(jnp.abs(Field_y_rotate_continuation))
    Field_y_rotate_max_id=tuple(jnp.array(jnp.where(jnp.abs(Field_y_rotate_continuation)==Field_y_rotate_max))[:,0])
    Field_center_mask=jnp.s_[round(Field_y_rotate_max_id[0]-n_x/2):round(Field_y_rotate_max_id[0]+n_x/2),round(Field_y_rotate_max_id[1]-n_y/2):round(Field_y_rotate_max_id[1]+n_y/2)]
    print('Input field shape: ',Field_y.shape)
    print('transversal_rotate_max_id: ',Field_y_rotate_max_id)
    Field_y_rotate_max_x=x_rotate_continuation[Field_y_rotate_max_id]
    Field_y_rotate_max_y=y_rotate_continuation[Field_y_rotate_max_id]
    x_axis_rotate=jnp.linspace(start=Field_y_rotate_max_x-d_x*n_x/2,stop=Field_y_rotate_max_x+d_x*n_x/2,num=n_x,endpoint=False)
    y_axis_rotate=jnp.linspace(start=Field_y_rotate_max_y-d_y*n_y/2,stop=Field_y_rotate_max_y+d_y*n_y/2,num=n_y,endpoint=False)
    Field_x_rotate=Field_x_rotate_continuation[Field_center_mask]
    Field_y_rotate=Field_y_rotate_continuation[Field_center_mask]
    print('Square integrate (should be close to 1.0)')
    print((square_integral_field_2D(Field_x_rotate)+square_integral_field_2D(Field_y_rotate))/(square_integral_field_2D(Field_x)+square_integral_field_2D(Field_y)))
    return {
        'Field_x_rotate': Field_x_rotate,   #shape=(n_x,n_y)
        'Field_y_rotate':Field_y_rotate,   #shape=(n_x,n_y)
        'Field_y_rotate_max_id':Field_y_rotate_max_id,   #the id of max of transversal_rotate in (2*n_x,2*n_y) grid
        'Field_center_mask':Field_center_mask,   #
        'x_axis_rotate':x_axis_rotate,   #shape=(n_x,)
        'y_axis_rotate':y_axis_rotate,   #shape=(n_y,)
    }

def shift_field_2D(Field:jnp.ndarray,x_axis_0:jnp.ndarray,y_axis_0:jnp.ndarray,x_axis_1:jnp.ndarray,y_axis_1:jnp.ndarray):
    """
    Shift the field from original grid to new grid. 
    Example: Shift the field from the boundary of the grid to the center of the grid.
    Example: Zoom the field with a different resolution.
    Example: Choose fields within our area of interest.
    Args:
        Field (jnp.ndarray): _description_
        x_axis_0 (jnp.ndarray): 1D array. Original x grid.
        y_axis_0 (jnp.ndarray): 1D array. Original y grid.
        x_axis_1 (jnp.ndarray): 1D array. New x grid.
        y_axis_1 (jnp.ndarray): 1D array. New y grid.
    """
    assert Field.ndim==2
    assert x_axis_0.ndim==1
    assert y_axis_0.ndim==1
    assert x_axis_1.ndim==1
    assert y_axis_1.ndim==1
    assert Field.shape==(x_axis_0.size,y_axis_0.size)
    print('Original shape')
    print(Field.shape)
    d_x_0=(x_axis_0[-1]-x_axis_0[0])/(x_axis_0.size-1)
    d_y_0=(y_axis_0[-1]-y_axis_0[0])/(y_axis_0.size-1)
    d_x_1=(x_axis_1[-1]-x_axis_1[0])/(x_axis_1.size-1)
    d_y_1=(y_axis_1[-1]-y_axis_1[0])/(y_axis_1.size-1)
    x_axis_1_in_0_id=(x_axis_1 - x_axis_0[0]) / d_x_0   #the position (id,could be non-integer) of new x in original x
    y_axis_1_in_0_id=(y_axis_1 - y_axis_0[0]) / d_y_0
    x_1_in_0_id,y_1_in_0_id=jnp.meshgrid(x_axis_1_in_0_id,y_axis_1_in_0_id,indexing='ij')
    Field_1=map_coordinates(input=Field, coordinates=jnp.array([x_1_in_0_id,y_1_in_0_id]), order=5, mode='constant', cval=0.0)
    print('New shape')
    print(Field_1.shape)
    print('Square integrate (should be close to 1.0)')
    print(square_integral_field_2D(Field=Field_1,d_x=d_x_1,d_y=d_y_1)/square_integral_field_2D(Field=Field,d_x=d_x_0,d_y=d_y_0))
    return Field_1

def smooth_edge_2D(Field:jnp.ndarray,mask:jnp.ndarray,edge_length=100):
    n_x,n_y=Field.shape
    assert mask.shape==(n_x,n_y)
    threshold=1e-3
    x_id=jnp.arange(n_x)
    y_id=jnp.arange(n_y)
    mask=jnp.ones(shape=(n_x,n_y),dtype=jnp.int8)*mask
    sum_x=jnp.sum(mask,axis=0)   #shape=(n_y,)
    sum_y=jnp.sum(mask,axis=1)   #shape=(n_x,)
    x_min_id=jnp.where(sum_x<threshold, 2*n_x-1, jnp.argmax(mask,axis=0))   #shape=(n_y,)
    x_max_id=jnp.where(sum_x<threshold, -n_x, n_x-1-jnp.argmax(mask[::-1,:],axis=0))   #shape=(n_y,)
    y_min_id=jnp.where(sum_y<threshold, 2*n_y-1, jnp.argmax(mask,axis=1))   #shape=(n_x,)
    y_max_id=jnp.where(sum_y<threshold, -n_y, n_y-1-jnp.argmax(mask[:,::-1],axis=1))   #shape=(n_x,)
    x_left_trans=0.5 * (1 + erf((x_id[:,jnp.newaxis] - x_min_id[jnp.newaxis,:]-edge_length) /edge_length))   #shape==(n_x,n_y)
    x_right_trans= 0.5 * (1 - erf((x_id[:,jnp.newaxis] - x_max_id[jnp.newaxis,:]+edge_length) / edge_length))   #shape==(n_x,n_y)
    y_left_trans=0.5 * (1 + erf((y_id[jnp.newaxis,:] - y_min_id[:,jnp.newaxis]-edge_length) /edge_length))   #shape==(n_x,n_y)
    y_right_trans= 0.5 * (1 - erf((y_id[jnp.newaxis,:] -y_max_id[:,jnp.newaxis]+edge_length) / edge_length))    #shape==(n_x,n_y)
    mask_smooth=x_left_trans+x_right_trans+y_left_trans+y_right_trans-3
    mask_smooth=mask_smooth*(mask_smooth>threshold)
    Field_smooth=Field*mask_smooth
    print(square_integral_field_2D(Field_smooth)/square_integral_field_2D(Field))
    return Field_smooth


    
def write_field_2D(Field_list:list[jnp.ndarray],x_axis:jnp.ndarray,y_axis:jnp.ndarray,name_list=list[str],nc_name='',working_dir=''):
    assert x_axis.ndim==1
    assert y_axis.ndim==1
    assert len(Field_list)==len(name_list)
    data_vars={}
    for Field,name in zip(Field_list,name_list):
        Field=np.asarray(Field,dtype=np.float64)
        assert Field.shape==(x_axis.size,y_axis.size)
        #Field.transpose(1,0).tofile(os.path.join(working_dir,name))
        data_vars[name]=(["x", "y"], Field)
    field_ds=xr.Dataset(
        data_vars=data_vars,
        coords={'x':(["x"], x_axis),'y':(["y"], y_axis)}
        )
    nc_path=os.path.join(working_dir,nc_name)
    field_ds.to_netcdf(path=nc_path,format="NETCDF4", engine='h5netcdf')
    print('Write field to %s' %nc_path)
    return 0

def convert_axes(Field:jnp.ndarray,x_axis=[0],y_axis=[0],z_axis=[0],input_axis='xz',output_axis='xy'):
    """
    Convert the axes of the field from input_axis to output_axis.
    xy: Field.shape=(x_axis.size,y_axis.size). Used in 2D simulations, where wave propagates in x direction, polarization in y direction.
    xz: Field.shape=(x_axis.size,z_axis.size). Used in 2D wave eigensystem, where wave propagates in z direction, polarization in x direction.
    xyz: Field.shape=(x_axis.size,y_axis.size=1,z_axis.size)
    Args:
        Field (jnp.ndarray): The input field. The shape should match the input_axis.
        x_axis (list): The x axis of the field.
        y_axis (list): The y axis of the field.
        z_axis (list): The z axis of the field.
        input_axis (str): The input axis order. Default is 'xz'.
        output_axis (str): The output axis order. Default is 'xy'. 
    Returns:
        jnp.ndarray: The converted field.
    """
    assert input_axis in ['xy','xz','xyz']
    assert output_axis in ['xy','xz','xyz']
    if input_axis=='xy':
        assert Field.shape==(x_axis.size,y_axis.size)
        Field_0=Field.transpose(1,0)   #shape=(y_axis.size,x_axis.size)
        x_axis_0=y_axis
        z_axis_0=x_axis
    elif input_axis=='xz':
        assert Field.shape==(x_axis.size,z_axis.size)
        Field_0=Field   #shape=(x_axis.size,z_axis.size)
        x_axis_0=x_axis
        z_axis_0=z_axis
    elif input_axis=='xyz':
        assert Field.shape==(x_axis.size,1,z_axis.size)
        Field_0=Field.squeeze(axis=1)   #shape=(x_axis.size,z_axis.size)
        x_axis_0=x_axis
        z_axis_0=z_axis
    #Convert to output axis
    if output_axis=='xy':
        Field_1=Field_0.transpose(1,0)   #shape=(x_axis.size,y_axis.size)
        x_axis_1=z_axis_0
        y_axis_1=x_axis_0
        return Field_1, x_axis_1, y_axis_1
    elif output_axis=='xz':
        Field_1=Field_0   #shape=(x_axis.size,z_axis.size)
        x_axis_1=x_axis_0
        z_axis_1=z_axis_0
        return Field_1, x_axis_1, z_axis_1
    elif output_axis=='xyz':
        Field_1=Field_0[:,jnp.newaxis,:]   #shape=(x_axis.size,1,z_axis.size)
        x_axis_1=x_axis_0
        z_axis_1=z_axis_0
        return Field_1, x_axis_1, z_axis_1

theta_degree=45
ND_a0=0.30
D=0.05
def f(Kappa):

    working_dir=os.path.join('/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface/test_adjust_w0/a0=200,W0=1','%d' %theta_degree,'D_%3.2f_Kappa_%+5.3f' %(D,Kappa))
    print(working_dir)
    #data_dict=read_nc(nc_name=os.path.join(working_dir,'reflection_500cpl.nc'),key_name_list=['Electric_Field_Ex','Electric_Field_Ey','Magnetic_Field_Bz','x','y'])
    Field_interest_list=['Electric_Field_Ex','Electric_Field_Ey','Magnetic_Field_Bz']
    data_dict=read_sdf(sdf_name=os.path.join(working_dir,'fields0001.sdf'),block_name_list=Field_interest_list)

    laser_lambda = 0.8*C.micron		# Laser wavelength
    vacuum_length_x_lambda=40
    vacuum_length_y_lambda=40
    cells_per_lambda_x=250
    cells_per_lambda_y=250
    cells_per_lambda_new=250
    n_field_x=round(2*vacuum_length_x_lambda*cells_per_lambda_x)
    n_field_y=round(2*vacuum_length_y_lambda*cells_per_lambda_y)
    d_x=laser_lambda/cells_per_lambda_x
    d_y=laser_lambda/cells_per_lambda_y


    x_min=-vacuum_length_x_lambda*laser_lambda
    x_max=vacuum_length_x_lambda*laser_lambda
    y_min=-vacuum_length_y_lambda*laser_lambda
    y_max=vacuum_length_y_lambda*laser_lambda

    x_axis=jnp.linspace(start=x_min,stop=x_max,num=n_field_x,endpoint=False,dtype=jnp.float64)+d_x/2
    y_axis=jnp.linspace(start=y_min,stop=y_max,num=n_field_y,endpoint=False,dtype=jnp.float64)+d_y/2
    xb_axis=x_axis+d_x/2
    yb_axis=y_axis+d_y/2

    n_field_x_new=round(2*vacuum_length_x_lambda*cells_per_lambda_new)
    n_field_y_new=round(2*vacuum_length_y_lambda*cells_per_lambda_new)
    d_x_new=laser_lambda/cells_per_lambda_new
    d_y_new=d_x_new
    x_axis_new=jnp.linspace(start=x_min,stop=x_max,num=n_field_x_new,endpoint=False,dtype=jnp.float64)+d_x_new/2
    y_axis_new=jnp.linspace(start=y_min,stop=y_max,num=n_field_y_new,endpoint=False,dtype=jnp.float64)+d_y_new/2
    #x_axis_new=data_dict['x']
    #y_axis_new=data_dict['y']

    x,y=jnp.meshgrid(x_axis_new,y_axis_new,indexing='ij')

    Electric_Field_Ex=data_dict['Electric_Field_Ex']
    Electric_Field_Ey=data_dict['Electric_Field_Ey']
    #Electric_Field_Ez=data_dict['Electric_Field_Ez']
    #Magnetic_Field_Bx=data_dict['Magnetic_Field_Bx']
    #Magnetic_Field_By=data_dict['Magnetic_Field_By']
    Magnetic_Field_Bz=data_dict['Magnetic_Field_Bz']
    #Derived_Number_Density_Electron=data_dict['Derived_Number_Density_Electron']
    #Derived_Number_Density_Ion=data_dict['Derived_Number_Density_Ion']
    Electric_Field_Ex_1=shift_field_2D(Field=Electric_Field_Ex,x_axis_0=xb_axis,y_axis_0=y_axis,x_axis_1=x_axis_new,y_axis_1=y_axis_new)
    Electric_Field_Ey_1=shift_field_2D(Field=Electric_Field_Ey,x_axis_0=x_axis,y_axis_0=yb_axis,x_axis_1=x_axis_new,y_axis_1=y_axis_new)
    #Electric_Field_Ez_1=shift_field_2D(Field=Electric_Field_Ez,x_axis_0=x_axis,y_axis_0=y_axis,x_axis_1=x_axis_new,y_axis_1=y_axis_new)
    #Magnetic_Field_Bx_1=shift_field_2D(Field=Magnetic_Field_Bx,x_axis_0=x_axis,y_axis_0=yb_axis,x_axis_1=x_axis_new,y_axis_1=y_axis_new)
    #Magnetic_Field_By_1=shift_field_2D(Field=Magnetic_Field_By,x_axis_0=xb_axis,y_axis_0=y_axis,x_axis_1=x_axis_new,y_axis_1=y_axis_new)
    Magnetic_Field_Bz_1=shift_field_2D(Field=Magnetic_Field_Bz,x_axis_0=xb_axis,y_axis_0=yb_axis,x_axis_1=x_axis_new,y_axis_1=y_axis_new)
    #Derived_Number_Density_Electron_1=shift_field_2D(Field=Derived_Number_Density_Electron,x_axis_0=x_axis,y_axis_0=y_axis,x_axis_1=x_axis_new,y_axis_1=y_axis_new)
    #Derived_Number_Density_Ion_1=shift_field_2D(Field=Derived_Number_Density_Ion,x_axis_0=x_axis,y_axis_0=y_axis,x_axis_1=x_axis_new,y_axis_1=y_axis_new)

    write_field_2D(Field_list=[Electric_Field_Ex_1,Electric_Field_Ey_1,Magnetic_Field_Bz_1],
                x_axis=x_axis_new,y_axis=y_axis_new,
                name_list=['Electric_Field_Ex','Electric_Field_Ey','Magnetic_Field_Bz'],
                nc_name=os.path.join(working_dir,'fields0001_%dcpl.nc'%(cells_per_lambda_new)))
    
    target_curvature=Kappa/laser_lambda
    target_curvature=0
    mask=x-(target_curvature/2)*jnp.square(y)<0
    
    Electric_Field_Ex_reflection=smooth_edge_2D(Electric_Field_Ex_1,mask=mask,edge_length=2*cells_per_lambda_new)
    Electric_Field_Ey_reflection=smooth_edge_2D(Electric_Field_Ey_1,mask=mask,edge_length=2*cells_per_lambda_new)
    Magnetic_Field_Bz_reflection=smooth_edge_2D(Magnetic_Field_Bz_1,mask=mask,edge_length=2*cells_per_lambda_new)

    rotate_parallel_field_2D_dict=rotate_parallel_field_2D(Field_x=Electric_Field_Ex_reflection,Field_y=Electric_Field_Ey_reflection,x_axis=x_axis_new,y_axis=y_axis_new,angle=-3*jnp.pi/4)
    Electric_Field_Ex_reflection_rotate=rotate_parallel_field_2D_dict['Field_x_rotate']
    Electric_Field_Ey_reflection_rotate=rotate_parallel_field_2D_dict['Field_y_rotate']
    Magnetic_Field_Bz_reflection_rotate=rotate_perpendicular_field_2D(Magnetic_Field_Bz_reflection,angle=-3*jnp.pi/4)[rotate_parallel_field_2D_dict['Field_center_mask']]
    x_axis_rotate=rotate_parallel_field_2D_dict['x_axis_rotate']
    y_axis_rotate=rotate_parallel_field_2D_dict['y_axis_rotate']

    write_field_2D(
        Field_list=[Electric_Field_Ex_reflection_rotate,Electric_Field_Ey_reflection_rotate,Magnetic_Field_Bz_reflection_rotate],
        x_axis=x_axis_rotate,y_axis=y_axis_rotate,
        name_list=['Electric_Field_Ex','Electric_Field_Ey','Magnetic_Field_Bz'],nc_name=os.path.join(working_dir,'reflection_%dcpl.nc' %(cells_per_lambda_new)))

    Electric_Field_Ex_transmission=smooth_edge_2D(Electric_Field_Ex_1,mask=~mask,edge_length=2*cells_per_lambda_new)
    Electric_Field_Ey_transmission=smooth_edge_2D(Electric_Field_Ey_1,mask=~mask,edge_length=2*cells_per_lambda_new)
    Magnetic_Field_Bz_transmission=smooth_edge_2D(Magnetic_Field_Bz_1,mask=~mask,edge_length=2*cells_per_lambda_new)

    rotate_parallel_field_2D_dict=rotate_parallel_field_2D(Field_x=Electric_Field_Ex_transmission,Field_y=Electric_Field_Ey_transmission,x_axis=x_axis_new,y_axis=y_axis_new,angle=-1*jnp.pi/4)
    Electric_Field_Ex_transmission_rotate=rotate_parallel_field_2D_dict['Field_x_rotate']
    Electric_Field_Ey_transmission_rotate=rotate_parallel_field_2D_dict['Field_y_rotate']
    Magnetic_Field_Bz_transmission_rotate=rotate_perpendicular_field_2D(Magnetic_Field_Bz_transmission,angle=-1*jnp.pi/4)[rotate_parallel_field_2D_dict['Field_center_mask']]
    x_axis_rotate=rotate_parallel_field_2D_dict['x_axis_rotate']
    y_axis_rotate=rotate_parallel_field_2D_dict['y_axis_rotate']

    write_field_2D(
        Field_list=[Electric_Field_Ex_transmission_rotate,Electric_Field_Ey_transmission_rotate,Magnetic_Field_Bz_transmission_rotate],
        x_axis=x_axis_rotate,y_axis=y_axis_rotate,
        name_list=['Electric_Field_Ex','Electric_Field_Ey','Magnetic_Field_Bz'],nc_name=os.path.join(working_dir,'transmission_%dcpl.nc'%(cells_per_lambda_new)))

if __name__ == "__main__": 
    task_list=[delayed(f)(Kappa) for Kappa in [-0.05,-0.02,0,0.02,0.05]]
    result_list = Parallel(n_jobs=5, verbose=1, backend="loky")(task_list)
    print(result_list)





"transversal"
"longitudinal"
"incident"
'reflection,reflected'
'transmission,transmitted'






