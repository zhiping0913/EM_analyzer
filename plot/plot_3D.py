import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/start')
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/start/plot')
import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.scale import ScaleBase
from plot.plot_basic import savefig
from matplotlib.axes import Axes
from plot.plot_2D import plot_2D_field

def plot_3_2D_profiles(
    Field,x_axis=[0],y_axis=[0],z_axis=[0],
    x_profile_id:Optional[int]=None,y_profile_id:Optional[int]=None,z_profile_id:Optional[int]=None,
    xmin:Optional[float]=None,xmax:Optional[float]=None,
    ymin:Optional[float]=None,ymax:Optional[float]=None,
    zmin:Optional[float]=None,zmax:Optional[float]=None,
    vmin:Optional[float]=None,vmax:Optional[float]=None,
    cmap='RdBu',
    label=r'$a=\frac{E}{E_c}=\frac{B}{B_c}$',xlabel=r'$\frac{x}{\lambda_0}$',ylabel=r'$\frac{y}{\lambda_0}$',zlabel=r'$\frac{z}{\lambda_0}$',
    zoom:float=1.0,
    name='',working_dir='.',
    ):
    Field=np.asarray(Field)
    x_axis=np.asarray(x_axis).flatten()
    y_axis=np.asarray(y_axis).flatten()
    z_axis=np.asarray(z_axis).flatten()
    n_x=x_axis.size
    n_y=y_axis.size
    n_z=z_axis.size
    assert Field.shape==(n_x,n_y,n_z)
    Field_abs=np.abs(Field)
    Field_max=np.max(Field_abs)
    Field_max_id=tuple(np.asarray(np.where(Field_abs==Field_max),dtype=np.int32)[:,0])   #Field_max_id=(x_id,y_id,z_id)
    if x_profile_id is None or x_profile_id<0 or x_profile_id>=n_x:
        x_profile_id=Field_max_id[0]
    if y_profile_id is None or y_profile_id<0 or y_profile_id>=n_y:
        y_profile_id=Field_max_id[1]
    if z_profile_id is None or z_profile_id<0 or z_profile_id>=n_z:
        z_profile_id=Field_max_id[2]
    if xmin is None:
        xmin=np.min(x_axis)
    if xmax is None:
        xmax=np.max(x_axis)
    if ymin is None:    
        ymin=np.min(y_axis)
    if ymax is None:
        ymax=np.max(y_axis)
    if zmin is None:
        zmin=np.min(z_axis)
    if zmax is None:
        zmax=np.max(z_axis)
    if vmin is None:
        vmin=np.min(Field)
    if vmax is None:
        vmax=np.max(Field)
    plot_2D_field(
        Field=Field[x_profile_id,:,:],   #y-z plane
        x_axis=y_axis,y_axis=z_axis,
        xmin=ymin,xmax=ymax,
        ymin=zmin,ymax=zmax,
        vmin=vmin,vmax=vmax,cmap=cmap,zoom=zoom,
        label=label,xlabel=ylabel,ylabel=zlabel,
        name=f'{name} {xlabel}={x_axis[x_profile_id]}',
        working_dir=working_dir
        )
    plot_2D_field(
        Field=Field[:,y_profile_id,:],   #x-z plane
        x_axis=x_axis,y_axis=z_axis,
        xmin=xmin,xmax=xmax,
        ymin=zmin,ymax=zmax,
        vmin=vmin,vmax=vmax,cmap=cmap,zoom=zoom,
        label=label,xlabel=xlabel,ylabel=zlabel,
        name=f'{name} {ylabel}={y_axis[y_profile_id]}',
        working_dir=working_dir
        )
    plot_2D_field(
        Field=Field[:,:,z_profile_id],   #x-y plane
        x_axis=x_axis,y_axis=y_axis,
        xmin=xmin,xmax=xmax,
        ymin=ymin,ymax=ymax,
        vmin=vmin,vmax=vmax,cmap=cmap,zoom=zoom,
        label=label,xlabel=xlabel,ylabel=ylabel,
        name=f'{name} {zlabel}={z_axis[z_profile_id]}',
        working_dir=working_dir
        )

working_dir='.'
