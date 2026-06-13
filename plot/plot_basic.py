import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import Normalize


def savefig(fig=None,fig_path=''):
    if fig is None:
        fig=plt.gcf()
    plt.savefig(fig_path)
    plt.close(fig)
    plt.clf()
    print(fig_path)
    return fig_path

def slice_between(coordinate, min=None, max=None, step:int=1):
    """
    Get a slice object that selects all elements in `coordinate` within the range [min, max].
    
    Parameters:
        coordinate: A sorted 1D array (ascending)
        min: Lower bound, inclusive, None means negative infinity (take the first element)
        max: Upper bound, inclusive, None means positive infinity (take the last element)
        step: Step size for slicing, default is 1 (take every element)
    Returns:
        slice object, can be used to index `coordinate`
    """
    coordinate = np.asarray(coordinate).flatten()  # Ensure it's a numpy array and handle NaN values
    n = coordinate.size
    if n == 0:
        return slice(0, 0)
    
    if min is None:
        start = 0
    else:
        # First index >= min
        start = np.searchsorted(coordinate, min, side='left')
    
    if max is None:
        stop = n
    else:
        # Last index <= max + 1
        stop = np.searchsorted(coordinate, max, side='right')
    
    # Handle possible invalid range
    if start > stop:
        start = stop
    return slice(start, stop, step)

def phase_amp_to_rgb(
    phase:np.ndarray,amplitude:np.ndarray,amplitude_max=None
    ):
    """
    Convert phase and amplitude to rgb image  
    parameters:
    phase: ND array, representing phase data (range should be [-π, π])
    amplitude: ND array, representing amplitude data
    returns:
    rgb_image: (N+1)D array, shape (*shape, 3), representing RGB image
    """
    phase=np.asarray(phase)
    amplitude=np.asarray(amplitude) 
    shape=phase.shape
    assert amplitude.shape==shape, "Phase and amplitude must have the same shape."
    amplitude=np.abs(amplitude)
    phase_normalized=np.mod((phase+np.pi)/(2*np.pi), 1)   #normalize to [0, 1], ['-π', '-π/2', '0', 'π/2', 'π']->[0, 0.25, 0.5, 0.75, 1]
    if amplitude_max is None:
        amplitude_max=np.max(amplitude)
    amplitude_normalized=np.where(amplitude>amplitude_max, 1.0, amplitude/amplitude_max)   #normalize to [0, 1]
    h=phase_normalized  #phase as hue, shape (*shape)
    s=np.ones_like(h)  #saturation fixed to 1, shape (*shape)
    v=amplitude_normalized  #amplitude as value, shape (*shape)
    hsv_image=np.stack((h,s,v),axis=-1)   #shape (*shape, 3)
    rgb_image=hsv_to_rgb(hsv_image)  #shape (*shape, 3)
    return rgb_image 
    



def Plot_complex_field_3D(
    A:np.ndarray=None, 
    phase:np.ndarray=None,amplitude:np.ndarray=None,
    A_max=None,
    x_axis=None,y_axis=None,z_axis=None,
    xlabel='x',ylabel='y',zlabel='z',label='',
    xmin:Optional[float]=None,xmax:Optional[float]=None,ymin:Optional[float]=None,ymax:Optional[float]=None,zmin:Optional[float]=None,zmax:Optional[float]=None,
    ):
    pass  # To be implemented in the future


working_dir="/scratch/gpfs/MIKHAILOVA/zl8336/start/plot"

