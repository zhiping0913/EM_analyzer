import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/EM_analyzer')
from line_profiler import profile
import numpy as np
import jax
from wrapt import partial
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
from scipy.signal.windows import tukey
from scipy.ndimage import map_coordinates
#from jax.scipy.ndimage import map_coordinates
import jax.numpy as jnp
from typing import List, Optional, Tuple, Union
import scipy.constants as C
import string
from jax_array_info import sharding_info
def print_array_size(a:np.ndarray,name=''):
    print(f'array {name}, shape={a.shape}, dtype={a.dtype}, size={a.size}, itemsize={a.itemsize}b, nbytes={a.nbytes/C.gibi}GB')


def print_shard_layout(arr, name=""):
    """Print, on each process, which device every addressable shard of `arr`
    lives on and which index slice it covers. Useful for diagnosing multi-host
    sharding (each process only sees its own local shards)."""
    pid = jax.process_index()
    try:
        shards = arr.addressable_shards
    except AttributeError:
        print(f"[{name}] process {pid}: array has no addressable_shards (not a jax.Array?)", flush=True)
        return
    print(f"[{name}] process {pid} sees {len(shards)} addressable shard(s):", flush=True)
    for i, s in enumerate(shards):
        print(f"  [{name}] shard {i}: device={s.device}  index={s.index}", flush=True)


@jax.jit
def outer_product(arrays: List[jnp.ndarray]) -> jnp.ndarray:
    """
    Compute the outer product of multiple 1D arrays.
    For n arrays with shapes (d0,), (d1,), ..., (d{n-1},), returns an n-dimensional
    array of shape (d0, d1, ..., d{n-1}) where:
    b[i0, i1, ..., i{n-1}] = arrays[0][i0] * arrays[1][i1] * ... * arrays[n-1][i{n-1}]
    Parameters:
    arrays : List[jnp.ndarray]
        List of 1D numpy arrays (lengths may differ)
    Returns:
    jnp.ndarray
        n-dimensional outer product array with shape (d0, d1, ..., d{n-1})
    """
    ndim = len(arrays)
    assert ndim >= 1, "At least one array is required"
    new_shape=[]
    for i, arr in enumerate(arrays):
        arrays[i] = jnp.asarray(arrays[i])
        assert arrays[i].ndim == 1, f"All input arrays must be 1D, but array {i} has shape {arrays[i].shape}"
        new_shape.append(arrays[i].size)
    new_shape=tuple(new_shape)
    #Compute outer product using numpy broadcasting.
    #More memory efficient than einsum for large arrays.
    result = arrays[0].copy()
    if ndim >1:
        for arr in arrays[1:]:
            # Add new axis to result and arr for broadcasting
            result = result[:, jnp.newaxis] * arr[jnp.newaxis, :]
            # Reshape to combine new dimension
            result = result.reshape(-1)   #shape=(d1*d2*...*d(i-1)*di, ) after multiplying with arr of shape (di,)
    return result.reshape(new_shape)
    # Use einsum for efficient computation
    # Build einsum string: for n arrays, we need n subscript letters
    letters = string.ascii_lowercase[:len(arrays)]
    einsum_str = ','.join(letters) + '->' + letters
    return jnp.einsum(einsum_str, *arrays, optimize='optimal')


@jax.jit
def sine_step(x):
    x_eff=jnp.clip(x, -0.5, 0.5)
    y = (1 + jnp.sin(jnp.pi * x_eff)) / 2
    return y
@jax.jit
def sine_step_edge(x,edge_length,edge_start,direction=1):
    """
    Sine step function for edge smoothing.
    Args:
        x (jnp.ndarray): Input array.
        edge_length (int): Length of the edge region.
        edge_start (int): Starting index of the edge region where the edge = 0.
        direction (int): 1 for rising edge, -1 for falling edge.
        When direction=1, the edge starts from 0 at edge_start and rises to 1 at edge_start + edge_length.
        When direction=-1, the edge starts from 1 at edge_start - edge_length and falls to 0 at edge_start.
    Returns:
        jnp.ndarray: Sine step values.
    """
    x_normalized=jnp.sign(direction)*(x - edge_start)/edge_length-0.5
    y=sine_step(x_normalized)
    return y

def get_edge_smooth_window(shape: Tuple[int, ...], edge_length: Union[List[int], Tuple[int, ...],jnp.ndarray]):
    """
    shape: tuple of ints, the shape of the window to be generated
    edge_length: list or tuple of ints, the length of the edge region for each dimension. 
    len(shape) must be equal to len(edge_length). If edge_length[i] = 0, no smoothing is applied along dimension i.
    scipy.signal.windows.tukey
    """
    ndim=len(shape)
    assert len(edge_length)==ndim, f"edge_length length {len(edge_length)} must match shape length {ndim}."
    window_dim_i_list: List[jnp.ndarray]=[]
    for dim_i in range(ndim):
        if edge_length[dim_i]>0 and shape[dim_i]>1:
            id_axis_i=jnp.arange(shape[dim_i])
            window_dim_i=sine_step_edge(x=id_axis_i, edge_length=edge_length[dim_i], edge_start=0, direction=1)*sine_step_edge(x=id_axis_i, edge_length=edge_length[dim_i], edge_start=shape[dim_i], direction=-1)
            window_dim_i_list.append(window_dim_i)
        else:
            window_dim_i_list.append(jnp.ones(shape[dim_i]))
    window=outer_product(window_dim_i_list)
    return window
@partial(jax.jit, static_argnames=('shape','axis','alpha'))
def get_nd_tukey_window(shape: Tuple[int, ...],axis: Union[int, Tuple[int, ...]], alpha: Union[float, Tuple[float, ...]]):
    """
    Get a multi-dimensional Tukey window.
    Parameters:
    shape: tuple of ints, the shape of the window to be generated
    axis: int or tuple of ints, the axis or axes along which to apply the Tukey window. If an int is provided, it is converted to a tuple with one element.
    alpha: float or tuple of floats, the shape parameter of the Tukey window. Default is 0.1.
    Returns:
    jnp.ndarray: The generated multi-dimensional Tukey window.
    """
    ndim=len(shape)
    axis=list(set(np.asarray(axis,dtype=int).flatten()))
    assert all(0 <= ax < ndim for ax in axis), f"All axis values must be in the range [0, {ndim-1}]"
    alpha=np.asarray(alpha, dtype=np.float64)
    if alpha.size==1:
        alpha=np.full((len(axis),), alpha[0], dtype=np.float64)
    assert alpha.size==len(axis), f"alpha size {alpha.size} must match the number of axes {len(axis)}."
    window_dim_i_list: List[jnp.ndarray]=[]
    for dim_i in range(ndim):
        if dim_i in axis:
            window_dim_i=tukey(shape[dim_i], alpha=alpha[axis.index(dim_i)])
            window_dim_i_list.append(window_dim_i)
        else:
            window_dim_i_list.append(jnp.ones(shape[dim_i]))
    window=outer_product(window_dim_i_list)
    return window




def smooth_edge(Field:jnp.ndarray,edge_length: Union[List[int], Tuple[int, ...],int,jnp.ndarray]=10):
    shape=Field.shape
    ndim=Field.ndim
    edge_length=jnp.round(jnp.asarray(edge_length).flatten()).astype(jnp.int64)
    if edge_length.size==1:
        edge_length=jnp.full((ndim,),edge_length[0],dtype=jnp.int64)
    assert edge_length.size==ndim, f"edge_length size {edge_length.size} must match Field ndim {ndim}."
    window=get_edge_smooth_window(shape=shape, edge_length=edge_length)
    Field_smooth=Field*window
    return Field_smooth


def smooth_mask(
    mask: jnp.ndarray,
    edge_length: Union[int, List[int], Tuple[int, ...]] = 10,
) -> jnp.ndarray:
    """
    Smooth the edge of a mask using a Tukey
    (raised-cosine) taper.

    For each dimension, every 1-D slice perpendicular to that dimension is
    examined: the first and last nonzero positions of the mask define the
    extent, and a half-Tukey taper of width ``edge_length`` is applied at
    each boundary.  The per-dimension tapers are multiplied together and then
    multiplied with the binary mask, giving a smooth weight array that is 0
    outside the mask, rises to 1 in the interior, and tapers smoothly at the
    edges.

    Args:
        mask (jnp.ndarray): Boolean or numeric mask.
            Nonzero values define the region of interest.
        edge_length (int or list/tuple of int): Taper width in cells.
            A scalar applies the same width to every dimension; a sequence
            sets the width per dimension.

    Returns:
        jnp.ndarray: the smoothed mask.
    """

    ndim = mask.ndim
    shape = mask.shape

    mask = (jnp.asarray(mask) != 0).astype(jnp.float64)

    edge_length = jnp.round(
        jnp.asarray(edge_length, dtype=jnp.float64).flatten()
    ).astype(jnp.int64)
    if edge_length.size == 1:
        edge_length = jnp.full((ndim,), edge_length[0], dtype=jnp.int64)
    assert edge_length.size == ndim, (
        f"edge_length size {edge_length.size} must match mask ndim {ndim}."
    )

    trans = jnp.ones(shape, dtype=jnp.float64)

    for dim_i in range(ndim):
        el = int(edge_length[dim_i])
        if el <= 0:
            continue
        n = int(shape[dim_i])
        id_axis = jnp.arange(n, dtype=jnp.float64)

        # Which slices perpendicular to dim_i contain at least one nonzero?
        has_nonzero = jnp.any(mask > 0, axis=dim_i)           # shape without dim_i

        # First nonzero index along dim_i for each slice
        min_ids = jnp.argmax(mask > 0, axis=dim_i).astype(jnp.float64)
        min_ids = jnp.where(has_nonzero, min_ids, jnp.inf)     # inf → all-zero slice

        # Last nonzero index along dim_i for each slice
        max_ids = (
            n - 1 - jnp.argmax(jnp.flip(mask, axis=dim_i) > 0, axis=dim_i)
        ).astype(jnp.float64)
        max_ids = jnp.where(has_nonzero, max_ids, -jnp.inf)    # -inf → all-zero slice

        # Broadcast id_axis and per-slice bounds to the full field shape
        other_dims = tuple(d for d in range(ndim) if d != dim_i)
        id_axis_exp  = jnp.expand_dims(id_axis,  axis=other_dims)   # (1,…,n,…,1)
        min_ids_exp  = jnp.expand_dims(min_ids,  axis=dim_i)
        max_ids_exp  = jnp.expand_dims(max_ids,  axis=dim_i)

        # Rising taper: 0 at min_id, 1 at min_id + el  (Tukey cosine)
        t_rise = jnp.clip((id_axis_exp - min_ids_exp) / el, 0.0, 1.0)
        rise   = 0.5 * (1.0 - jnp.cos(jnp.pi * t_rise))
        rise   = jnp.where(min_ids_exp == jnp.inf, 0.0, rise)   # all-zero slice → 0

        # Falling taper: 1 at max_id - el, 0 at max_id  (Tukey cosine)
        t_fall = jnp.clip((max_ids_exp - id_axis_exp) / el, 0.0, 1.0)
        fall   = 0.5 * (1.0 - jnp.cos(jnp.pi * t_fall))
        fall   = jnp.where(max_ids_exp == -jnp.inf, 0.0, fall)

        trans *= rise * fall

    mask_smooth = mask * trans
    return mask_smooth



def stack_Fields(Field_x: Optional[jnp.ndarray]=None, Field_y: Optional[jnp.ndarray]=None, Field_z: Optional[jnp.ndarray]=None):
    """
    Stack field components into a single array.

    Parameters
    ----------
    Field_x, Field_y, Field_z : jnp.ndarray
        Field components, shape (Nx, Ny, Nz)

    Returns
    -------
    Field : jnp.ndarray
        Stacked field, shape (3, Nx, Ny, Nz)
    """
    Field_input_list=[Field_x, Field_y, Field_z]
    assert any(Field_comp is not None for Field_comp in Field_input_list), "At least one field component must be provided."
    Field_i=None
    _Field_input_list=[]
    for Field_comp in Field_input_list:
        if Field_comp is not None:
            Field_i=jnp.asarray(Field_comp)
            _Field_input_list.append(Field_i)
        else:
            _Field_input_list.append(None)
    Field_input_list=_Field_input_list
    dim=Field_i.ndim
    assert dim==1 or dim==2 or dim==3, f"Field component must have at least 1 dimension, got {dim}."
    Field_stack_list=[]
    if dim==1:
        Nx=Field_i.shape[0]
        Ny=1
        Nz=1
        for Field_comp in Field_input_list:
            if Field_comp is None:
                Field_stack_list.append(jnp.zeros((Nx,Ny,Nz)))
            else:
                assert Field_comp.shape == (Nx,), f"Field component shape {Field_comp.shape} does not match expected shape {(Nx,)}."
                Field_stack_list.append(jnp.reshape(Field_comp, (Nx,Ny,Nz)))
    elif dim==2:
        Nx=Field_i.shape[0]
        Ny=Field_i.shape[1]
        Nz=1
        for Field_comp in Field_input_list:
            if Field_comp is None:
                Field_stack_list.append(jnp.zeros((Nx,Ny,Nz)))
            else:
                assert Field_comp.shape == (Nx,Ny), f"Field component shape {Field_comp.shape} does not match expected shape {(Nx,Ny)}."
                Field_stack_list.append(jnp.reshape(Field_comp, (Nx,Ny,Nz)))
    elif dim==3:
        for Field_comp in Field_input_list:
            if Field_comp is None:
                Field_stack_list.append(jnp.zeros_like(Field_i))
            else:
                assert Field_comp.shape == Field_i.shape, f"Field component shape {Field_comp.shape} does not match expected shape {Field_i.shape}."
                Field_stack_list.append(Field_comp)
    Field_stack=jnp.stack(Field_stack_list, axis=0)
    print(f"Stacked field shape: {Field_stack.shape}")
    return Field_stack   #shape=(3, Nx, Ny, Nz)
@partial(jax.jit, static_argnames=('axis'))
def get_norm(Field:jnp.ndarray, axis=0):
    """
    Get the norm of the field.
    Parameters
    ----------
    Field : jnp.ndarray
        Field, shape (3, Nx, Ny, Nz)

    Returns
    -------
    Field_norm : jnp.ndarray
        Norm of the field, shape (Nx, Ny, Nz)
    """
    Field_norm = jnp.linalg.norm(Field, axis=axis)   #shape=(Nx, Ny, Nz)
    return Field_norm

@profile
def get_relative_divergence(
    Field:jnp.ndarray, 
    coordinate_list: List[jnp.ndarray],
    threshold=1e-3
    ):
    """
    Compute ∇·Field in real space.

    Parameters
    ----------
    Field : jnp.ndarray
        Field, shape (ndim, N_dim0, N_dim1, N_dim2), where ndim is the number of dimensions (1, 2, or 3) and N_dim0, N_dim1, N_dim2 are the sizes of each dimension. The first dimension corresponds to the field components (e.g., (Fx, Fy, Fz)).
    coordinate_list : List[jnp.ndarray]
        List of coordinate arrays for each dimension, e.g., [x_coordinate, y_coordinate, z_coordinate]. Each coordinate array should have shape (N_dim_i,). The length of coordinate_list should match the number of dimensions in Field.
    threshold : float
        The field with norm below this threshold*field_norm_max will be considered as zero to avoid numerical instability in divergence calculation.
    Returns
    -------
    divF : jnp.ndarray
        Divergence of the field, shape (Nx, Ny, Nz), unit: (units of Field)/m
    """
    Field=jnp.asarray(Field)   #shape=(ndim, N_dim0, N_dim1, N_dim2)
    ndim=Field.shape[0]
    div_F=jnp.zeros_like(Field[0])   #shape=(N_dim0, N_dim1, N_dim2), initialize divergence array
    assert ndim==len(Field.shape)-1, f"The first dimension of Field {Field.shape[0]} must match the number of dimensions {len(Field.shape)-1}."
    assert ndim==len(coordinate_list), f"Field has {ndim} dimensions but coordinate_list has {len(coordinate_list)} arrays."
    for dim_i in range(ndim):
        coordinate_i=jnp.asarray(coordinate_list[dim_i],dtype=jnp.float64).flatten()
        assert coordinate_i.size == Field.shape[dim_i+1], f"Coordinate array size {coordinate_i.size} does not match field size {Field.shape[dim_i+1]} for dimension {dim_i}."
        N_i=coordinate_i.size
        assert N_i > 0, f"Coordinate array for dimension {dim_i} must have at least one point."
        if N_i > 1:
            dFi_di = jnp.gradient(Field[dim_i], coordinate_i, axis=dim_i)   #shape=(N_dim0, N_dim1, N_dim2), ∂Fi/∂i
        else:
            dFi_di = jnp.zeros_like(Field[dim_i])
        div_F += dFi_di
    F_norm = get_norm(Field, axis=0)   #shape=(Nx, Ny, Nz)
    F_norm_max=jnp.max(jnp.abs(F_norm))
    div_F_relative = jnp.where(F_norm>F_norm_max*threshold, div_F/F_norm, 0.0)   #shape=(Nx, Ny, Nz), set divergence to zero where field norm is small to avoid numerical instability
    return div_F_relative

@profile
def check_divergence(Field:jnp.ndarray, x_coordinate=[0],y_coordinate=[0],z_coordinate=[0],  threshold=5e-1,scale_length=1e-6):
    """
    Check ∇·Field = 0 in real space.
    The denser the grid, the more accurate the calculation.
    Parameters
    ----------
    Field : jnp.ndarray
        Field, shape (3, Nx, Ny, Nz)
    x_coordinate, y_coordinate, z_coordinate : list or jnp.ndarray
        Grid axes for x, y, z directions
    threshold : float
        Relative divergence tolerance
    scale_length : float, unit: m
        Characteristic length scale for normalization

    Raises
    ------
    warning if divergence exceeds threshold
    """
    x_coordinate=jnp.array(x_coordinate).flatten()
    y_coordinate=jnp.array(y_coordinate).flatten()
    z_coordinate=jnp.array(z_coordinate).flatten()
    Nx=x_coordinate.size
    Ny=y_coordinate.size
    Nz=z_coordinate.size
    assert Nx>0 and Ny>0 and Nz>0, "Grid axes must have at least one point."
    div_F_relative=get_relative_divergence(Field=Field, coordinate_list=[x_coordinate,y_coordinate,z_coordinate], threshold=threshold)
    div_F_relative_max=jnp.max(jnp.abs(div_F_relative))
    div_err=div_F_relative_max*scale_length
    print(f"Max relative divergence error: L*|∇·F|/|F| = {div_err:.2e}")
    if div_err > threshold:
        print(f"Warning: Initial Field is not divergence-free: relative L*|∇·F|/|F| = {div_err} > {threshold}")
        return False
    else:
        print(f"Initial Field divergence check passed: relative L*|∇·F|/|F| = {div_err} <= {threshold}")
        return True

@partial(jax.jit, static_argnames=('axis'))
def square_sum(x,axis):
    return jnp.real(jnp.sum(jnp.conj(x)*x,axis=axis))
@profile
def square_integral_field(Field:jnp.ndarray,dr=[1],axis= None):
    """
    Args:
        Field (jnp.ndarray): _description_
        dr (list): [dx,dy,dz], grid spacing in each dimension. If None, defaults to ones.
        axis (int or tuple, optional): Axis or axes along which to compute the integral. If None, computes over all axes. Defaults to None.
        complex_array (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    square_integral=square_sum(Field, axis=axis)*np.prod(dr)
    jax.debug.print('∬|Field|^2×dr={square_integral}', square_integral=square_integral)
    return square_integral

@jax.jit
def _get_coordinate_id_float(coordinate_start,dr,pos):
    id_float=(pos - coordinate_start)/dr
    return id_float

@jax.jit
def _get_coordinate_id_int(coordinate_start,dr,N, pos):
    id_float=_get_coordinate_id_float(coordinate_start,dr,pos)
    id_int=jnp.clip(jnp.round(id_float),0,N-1).astype(jnp.int64)
    return id_int


@profile
def get_coordinate_id_float(coordinate:jnp.ndarray, pos:float|jnp.ndarray):
    """
        Get the floating-point index of a position in a given coordinate array.
        Overflow and underflow are allowed.
    Args:
        coordinate (jnp.ndarray): _description_
        pos (float | jnp.ndarray): _description_

    Returns:
        _type_: _description_
    """
    coordinate=jnp.asarray(coordinate).flatten()
    pos=jnp.asarray(pos)
    assert coordinate.ndim==1, "Coordinate must be 1D array."
    N=coordinate.size
    assert N>=1, "Coordinate must have at least one point."
    if N==1:
        return jnp.zeros_like(pos,dtype=jnp.float64)
    else:
        return _get_coordinate_id_float(coordinate_start=coordinate[0], dr=(coordinate[-1]-coordinate[0])/(N-1), N=N, pos=pos)


@profile
def get_coordinate_id_int(coordinate:jnp.ndarray, pos:float|jnp.ndarray):
    """
        Get the integer index of a position in a given coordinate array.
        Overflow and underflow are clipped to the valid range.
    Args:
        coordinate (jnp.ndarray): _description_
        pos (float | jnp.ndarray): _description_

    Returns:
        _type_: _description_
    """
    coordinate=jnp.asarray(coordinate).flatten()
    pos=jnp.asarray(pos).flatten()
    assert coordinate.ndim==1, "Coordinate must be 1D array."
    N=coordinate.size
    assert N>=1, "Coordinate must have at least one point."
    if N==1:
        id=jnp.zeros_like(pos,dtype=jnp.int64)
    else:
        id=_get_coordinate_id_int(coordinate_start=coordinate[0], dr=(coordinate[-1]-coordinate[0])/(N-1), N=N, pos=pos)
    return id

@jax.jit
def get_closest_coordinate_id(coordinate:jnp.ndarray,pos:jnp.ndarray):
    """Find the index of the closest coordinate for each position.
    Coordinate is not required to be sorted or equally spaced.
    Args:
        coordinate (jnp.ndarray): 1D array of coordinates.
        pos (jnp.ndarray): Array of positions. Can be a scalar or an array of any shape.

    Returns:
        jnp.ndarray: Array of indices of the closest coordinates.
    """
    pos=jnp.asarray(pos)
    distance=coordinate[:,*[jnp.newaxis]*pos.ndim]-pos[jnp.newaxis,...]  # shape=(len(coordinate), *pos.shape),ndim=pos.ndim+1 
    closest_coordinate_id=jnp.argmin(jnp.abs(distance), axis=0)  # shape=pos.shape,ndim=pos.ndim
    return closest_coordinate_id.astype(int)


@profile
def calculate_center(Field:jnp.ndarray, x_coordinate=[0], y_coordinate=[0], z_coordinate=[0],axis=None):
    """
    Calculate the center of the field distribution.

    Parameters
    ----------
    Field : jnp.ndarray
    x_coordinate, y_coordinate, z_coordinate : list or jnp.ndarray
        Grid axes for x, y, z directions
    axis : int or tuple, optional
        Axis or axes along which to compute the center. If None, computes over all axes. Defaults to None.

    Returns
    -------
    center : jnp.ndarray
        Center coordinates, shape depends on axis parameter
    """
    x_coordinate=jnp.array(x_coordinate).flatten()
    y_coordinate=jnp.array(y_coordinate).flatten()
    z_coordinate=jnp.array(z_coordinate).flatten()
    Nx=x_coordinate.size
    Ny=y_coordinate.size
    Nz=z_coordinate.size
    Field=jnp.asarray(Field)
    r=jnp.meshgrid(x_coordinate, y_coordinate, z_coordinate, indexing='ij')  #shape=(3, Nx, Ny, Nz)
    weight=jnp.square(jnp.abs(Field))
    if axis is None:
        assert Field.shape == (Nx, Ny, Nz), f"Field shape {Field.shape} does not match grid shape {(Nx, Ny, Nz)}"
    else:
        axis=jnp.asarray(axis,dtype=jnp.int32)
        assert tuple(Field.shape[ax] for ax in axis) ==(Nx, Ny, Nz), f"Field shape {Field.shape} does not match grid shape {(Nx, Ny, Nz)} along specified axes"
    rc=jnp.average(a=r, weights=weight, axis=axis)
    print(f"Center of the field distribution: {rc}")
    return rc

def pad_field(
    field: jnp.ndarray,
    output_shape: Tuple[int],
    location: Optional[Tuple[int]] = None,
    fill_value: float = 0
    ):
    # Validate inputs
    ndim = field.ndim
    assert len(output_shape) == ndim,f"output_shape length {len(output_shape)} must match array ndim {ndim}"
    # Default location is all zeros (upper-left corner)
    if location is None:
        location = tuple(0 for _ in range(ndim))
    assert len(location) == ndim,f"location length {len(location)} must match array ndim {ndim}"
    
    # Initialize output array with fill_value
    field_pad = jnp.full(output_shape, fill_value, dtype=field.dtype)
    
    # Calculate source and destination slices
    src_slices : List[slice] = []
    dst_slices : List[slice] = []
    dst_axis_start=[]
    dst_axis_end=[]
    for i in range(ndim):
        src_start = 0
        src_end = field.shape[i]
        dst_start = location[i]
        dst_end = location[i] + field.shape[i]
        dst_axis_start.append(location[i])
        dst_axis_end.append(location[i] + output_shape[i])
        # Clip to output bounds
        if dst_start < 0:
            # Negative start means part of source is clipped
            src_start = -dst_start
            dst_start = 0
        if dst_end > output_shape[i]:
            # End beyond output shape means part of source is clipped
            src_end = field.shape[i] - (dst_end - output_shape[i])
            dst_end = output_shape[i]
        # Check if there's any overlap
        if src_start >= src_end or dst_start >= dst_end:
            # No overlap along this dimension, skip copying
            src_slices.append(slice(0, 0))
            dst_slices.append(slice(0, 0))
        else:
            src_slices.append(slice(src_start, src_end))
            dst_slices.append(slice(dst_start, dst_end))
    dst_slices=tuple(dst_slices)
    src_slices=tuple(src_slices)
    pad_slices=dst_slices
    # Copy the overlapping region
    if all(s.start < s.stop for s in src_slices):
        field_pad = field_pad.at[dst_slices].set(field[src_slices])
    print('Input shape:', field.shape)
    print('Output shape:', field_pad.shape)
    return field_pad, pad_slices

def pad_coordinates(
    coordinate_list: List[jnp.ndarray],
    output_shape: Tuple[int],
    location: Optional[Tuple[int]] = None,
) -> List[jnp.ndarray]:
    """
    Pad coordinate arrays to match the output shape of the padded field.
    Parameters:
    -----------
    coordinate_list : List[jnp.ndarray]
        List of 1D coordinate arrays corresponding to each axis of the input array.
        Each array must be 1D with length equal to the corresponding dimension of the input field.
    output_shape : Tuple[int]
        Desired shape of the output array. Must have the same length as coordinate_list.
    location : Optional[Tuple[int]], default=None
        Starting index for each axis where the input field will be placed in the output.
        If None, defaults to (0, 0, ...) (upper-left corner).
        If location[i] < 0, it means padding after the data along coordinate[i], and the data before location[i] will not be shown in the output array.

    Returns:
    --------
    coordinate_pad_list : List[jnp.ndarray]
        List of padded coordinate arrays with lengths matching output_shape
    """
    ndim = len(coordinate_list)
    assert ndim == len(output_shape), f"coordinate_list length {ndim} must match output_shape length {len(output_shape)}"
    if location is None:
        location = tuple(0 for _ in range(ndim))
    assert len(location) == ndim, f"location length {len(location)} must match coordinate_list length {ndim}"
    coordinate_pad_list = []
    for i in range(ndim):
        coordinate_i=jnp.asarray(coordinate_list[i]).flatten()
        Ni = coordinate_i.size
        dxi = (coordinate_i[-1] - coordinate_i[0]) / (Ni - 1) if Ni > 1 else 0.0
        start_id = -location[i]
        end_id = start_id + output_shape[i]
        coordinate_pad_list.append(jnp.linspace(start=coordinate_i[0] + start_id * dxi, stop=coordinate_i[0] + (end_id - 1) * dxi, num=output_shape[i], endpoint=True, dtype=jnp.float64))
    return coordinate_pad_list


def pad_field_with_coordinate(
    field: jnp.ndarray,
    coordinate_list: List[jnp.ndarray],
    output_shape: Tuple[int],
    location: Optional[Tuple[int]] = None,
    fill_value: float = 0
) -> Tuple[jnp.ndarray, List[jnp.ndarray], Tuple[slice]]:
    """
    Pad a N-dimensional array to a specified output shape at given location with a fill value.
    Parameters:
    -----------
    field : jnp.ndarray
        Input array to be padded
    coordinate_list : List[jnp.ndarray]
        List of 1D coordinate arrays corresponding to each axis of the input array.
        Each array must be 1D with length equal to the corresponding dimension of 'field'.
    output_shape : Tuple[int]
        Desired shape of the output array. Must have the same length as 'field.ndim'.
    location : Optional[Tuple[int]], default=None
        Starting index for each axis where the input array will be placed in the output.
        If None, defaults to (0, 0, ...) (upper-left corner).
        If location[i] < 0, it means padding after the data along coordinate[i], and the data before location[i] will not be shown in the output array.
    fill_value : float, default=0
        Value used for padding areas outside the input array.
    
    Returns:
    --------
    field_pad : jnp.ndarray
        Padded array with shape 'output_shape'
    coordinate_pad_list : List[jnp.ndarray]
        List of padded coordinate arrays with lengths matching 'output_shape'
    pad_slices : Tuple[slice]
        Tuple of slices indicating the location of the original data within the padded array
    """
    print('This function will be discarded. Please use smooth_and_pad_field_with_coordinates')
    field_pad, pad_slices = pad_field(field=field, output_shape=output_shape, location=location, fill_value=fill_value)
    coordinate_pad_list = pad_coordinates(coordinate_list=coordinate_list, output_shape=output_shape, location=location)
    return field_pad, coordinate_pad_list, pad_slices

@partial(jax.jit, static_argnames=('pad_width_each_axis','axis','alpha'))
def _smooth_and_pad_field(
    field: jnp.ndarray,
    pad_width_each_axis:tuple[int, ...],
    axis: tuple[int, ...],
    alpha:Union[float, Tuple[float, ...]],
    ) -> jnp.ndarray:
    shape=field.shape
    ndim=len(shape)
    if abs(alpha)<1e-6:
        field_smooth=field
    else:
        window=get_nd_tukey_window(shape=shape, axis=axis, alpha=alpha)
        field_smooth=field*window   #shape=shape
    pad_width=[(0,0)]*ndim
    for axis_i, pad_width_i in zip(axis, pad_width_each_axis):
        pad_width[axis_i]=(pad_width_i, pad_width_i)
    field_pad=jnp.pad(field_smooth, pad_width=pad_width, mode='constant', constant_values=0)
    return field_pad

def smooth_and_pad_field(
    field: jnp.ndarray,
    pad_width_each_axis:tuple[int, ...],
    axis: Union[int, Tuple[int, ...]],
    alpha:Union[float, Tuple[float, ...]]=0.1
    ) -> Tuple[List[jnp.ndarray], Tuple[slice]]:
    """
    Smooth the edges of the input fields and pad them to a specified width.

    Parameters
    ----------
    field : jnp.ndarray
        Input field to be processed.
    axis : int or tuple of ints
        Axis or axes along which to apply the Tukey window for edge smoothing. If an int is provided, it is converted to a tuple with one element.
        If axis is None, no smoothing or padding is applied.
    pad_width_each_axis : tuple of ints or None
        len(pad_width_each_axis) must match len(axis)
        The number of points to pad on each axis in axis argument.
        The padding will be applied symmetrically on both sides of each axis.
    alpha : float or tuple of floats, optional
        The shape parameter of the Tukey window used for edge smoothing for each axis. Default is 0.1. 
        If alpha=0, the Tukey window is equivalent to a rectangular window (no smoothing). If alpha=1, the Tukey window is equivalent to a Hann window (maximum smoothing). For 0 < alpha < 1, the Tukey window has a tapered cosine shape at the edges and is flat in the center. The larger the alpha, the wider the tapered region and the smoother the edges.

    Returns
    -------
    field_pad : jnp.ndarray
        Smoothed and padded field
    """
    shape=field.shape
    ndim=len(shape)
    pad_width_each_axis=tuple(jnp.asarray(pad_width_each_axis,dtype=int).flatten().tolist())
    axis=tuple(jnp.mod(jnp.asarray(axis,dtype=int), ndim).flatten().tolist())
    assert axis is not None, "Axis for smoothing and padding cannot be None."
    assert len(pad_width_each_axis) == len(axis), f"pad_width_each_axis length {len(pad_width_each_axis)} must match axis length {len(axis)}"
    return _smooth_and_pad_field(field=field, pad_width_each_axis=pad_width_each_axis, axis=axis, alpha=alpha)




def smooth_and_pad_field_with_coordinates(
    field: jnp.ndarray, 
    axis: Union[int, Tuple[int, ...]],
    coordinate_each_axis: List[jnp.ndarray],
    pad_width_each_axis:tuple[int, ...],
    alpha: Union[float, Tuple[float, ...]]=0.1,
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], Tuple[slice]]:
    """
    Smooth the edges of the input fields and pad them to a specified width.

    Parameters
    ----------
    field : jnp.ndarray
        Input field to be processed.
    coordinate_each_axis : List[jnp.ndarray]
        List of grid axes for each dimension.
        len(coordinate_each_axis) must match len(axis) and the size of each coordinate array must match the corresponding dimension of the field.
    axis : int or tuple of ints
        Axis or axes along which to apply the Tukey window for edge smoothing. If an int is provided, it is converted to a tuple with one element.
        If axis is None, no smoothing or padding is applied.
    pad_width_each_axis : tuple of ints or None
        len(pad_width_each_axis) must match len(axis)
        The number of points to pad on each axis in axis argument.
        The padding will be applied symmetrically on both sides of each axis.
    alpha : float or tuple of floats, optional
        The shape parameter of the Tukey window used for edge smoothing for each axis. Default is 0.1.
        If alpha=0, the Tukey window is equivalent to a rectangular window (no smoothing). If alpha=1, the Tukey window is equivalent to a Hann window (maximum smoothing). For 0 < alpha < 1, the Tukey window has a tapered cosine shape at the edges and is flat in the center. The larger the alpha, the wider the tapered region and the smoother the edges.
    Returns
    -------
    field_pad : jnp.ndarray
        Smoothed and padded field
    coordinate_pad_list : List[jnp.ndarray]
        List of padded coordinate arrays corresponding to each axis
    pad_slices : Tuple[slice]
        Tuple of slices indicating the location of the original data within the padded array
    """
    field=jnp.asarray(field)
    shape=field.shape
    ndim=len(shape)
    pad_width_each_axis=tuple(jnp.asarray(pad_width_each_axis,dtype=int).flatten().tolist())
    axis=tuple(jnp.mod(jnp.asarray(axis,dtype=int), ndim).flatten().tolist())
    assert axis is not None, "Axis for smoothing and padding cannot be None."
    assert len(axis) > 0, "At least one axis must be specified for edge smoothing and padding."
    assert len(coordinate_each_axis) == len(axis), f"coordinate_each_axis length {len(coordinate_each_axis)} must match axis length {len(axis)}"
    assert len(pad_width_each_axis) == len(axis), f"pad_width_each_axis length {len(pad_width_each_axis)} must match axis length {len(axis)}"
    field_pad=_smooth_and_pad_field(field=field, pad_width_each_axis=pad_width_each_axis, axis=axis, alpha=alpha)
    pad_slices=[]
    coordinate_list=[]
    output_shape=[]
    for i in range(len(axis)):
        axis_i=axis[i]
        pad_width_i=pad_width_each_axis[i]
        assert pad_width_i >= 0, f"pad_width_each_axis must be non-negative, but got {pad_width_i} for axis {axis_i}"
        coordinate_i=jnp.asarray(coordinate_each_axis[i],dtype=jnp.float64).flatten()
        assert coordinate_i.size == shape[axis_i], f"Coordinate array size {coordinate_i.size} must match field size {shape[axis_i]} for axis {axis_i}."
        coordinate_list.append(coordinate_i)
        output_shape.append(shape[axis_i] + 2*pad_width_i)
        pad_slices.append(slice(pad_width_i, pad_width_i + shape[axis_i]))
    pad_slices=tuple(pad_slices)
    coordinate_pad_list=pad_coordinates(coordinate_list=coordinate_list, output_shape=tuple(output_shape), location=pad_width_each_axis)
    return field_pad, coordinate_pad_list, pad_slices

def pad_for_fft(
    field: jnp.ndarray,
    axis: Union[int, Tuple[int, ...]],
    coordinate_each_axis: List[jnp.ndarray],
) -> Tuple[jnp.ndarray, List[jnp.ndarray], Tuple[slice, ...]]:
    """
    Pad a field for FFT computation.

    For each specified axis:
      - If the axis length Ni < 64: no smoothing and no padding.
      - Otherwise: apply a Tukey window (alpha=0.1) for edge smoothing, then
        zero-pad so that the output size is the smallest multiple of 16 strictly
        greater than 2*Ni.  The original data is placed at the center of the
        padded array (asymmetric by at most 1 point when the total padding is odd).

    Parameters
    ----------
    field : jnp.ndarray
        Input field array.
    axis : int or tuple of ints
        Axes along which to pad.
    coordinate_each_axis : List[jnp.ndarray]
        Coordinate array for each axis in `axis`.  Must have the same length
        as `axis`; each coordinate array must match the corresponding axis size.
    Returns
    -------
    field_pad : jnp.ndarray
        Smoothed and zero-padded field.
    coordinate_pad_list : List[jnp.ndarray]
        Padded coordinate arrays (one per entry in `axis`), extended uniformly
        outside the original domain.
    pad_slices : Tuple[slice, ...]
        Slices that recover the original data from `field_pad` along each axis
        in `axis`.
    """
    field = jnp.asarray(field)
    ndim = field.ndim
    axis = np.mod(np.asarray(axis,dtype=int).flatten(), ndim).tolist()  # handle negative axis
    assert len(coordinate_each_axis) == len(axis), (
        f"coordinate_each_axis length {len(coordinate_each_axis)} must match axis length {len(axis)}"
    )

    pad_width: List[Tuple[int, int]] = [(0, 0)] * ndim
    pad_befores: List[int] = []
    out_sizes: List[int] = []
    smooth_axes: List[int] = []
    smooth_alphas: List[float] = []

    for i, ax in enumerate(axis):
        Ni = field.shape[ax]
        if Ni < 64:
            pad_befores.append(0)
            out_sizes.append(Ni)
        else:
            out_size = (2 * Ni // 16 + 1) * 16   # smallest multiple of 16 strictly > 2*Ni
            total_pad = out_size - Ni
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_width = [pw if j != ax else (pad_before, pad_after) for j, pw in enumerate(pad_width)]
            pad_befores.append(pad_before)
            out_sizes.append(out_size)
            smooth_axes.append(ax)
            smooth_alphas.append(0.1)

    if len(smooth_axes) > 0:
        window = get_nd_tukey_window(shape=field.shape, axis=tuple(smooth_axes), alpha=tuple(smooth_alphas))
        field_smoothed = field * window
    else:
        field_smoothed = field

    field_pad = jnp.pad(field_smoothed, pad_width=pad_width, mode='constant', constant_values=0)
    print(f"Pad widths for each axis: {pad_width}")
    print('Input shape:', field.shape)
    print('Output padded shape:', field_pad.shape)
    coordinate_pad_list = pad_coordinates(
        coordinate_list=[jnp.asarray(c, dtype=jnp.float64).flatten() for c in coordinate_each_axis],
        output_shape=tuple(out_sizes),
        location=tuple(pad_befores),
    )

    pad_slices = tuple(
        slice(pad_befores[i], pad_befores[i] + field.shape[ax]) for i, ax in enumerate(axis)
    )
    return field_pad, coordinate_pad_list, pad_slices

#@jax.jit
def _shift_field(field: jnp.ndarray, coordinates: jnp.ndarray) -> jnp.ndarray:
    """
    JIT-compiled core interpolation for shift_field.
    Args:
        field (jnp.ndarray): Input field of any shape.
        coordinates (jnp.ndarray): Fractional index arrays, shape (ndim, *new_shape).
    Returns:
        jnp.ndarray: Interpolated field with shape coordinates.shape[1:].
    """
    return map_coordinates(
        input=field,
        coordinates=coordinates,
        order=5,
        mode='constant',
        cval=0.0,
    )


def shift_field(
    field: jnp.ndarray,
    coordinate_original_each_axis: List[jnp.ndarray],
    coordinate_new_each_axis: List[jnp.ndarray],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    with_sharding=None,
    ):
    """
    Shift the field from original grid to new grid.
    Example: Shift the field from the boundary of the grid to the center of the grid.
    Example: Zoom the field with a different resolution.
    Example: Choose fields within our area of interest.
    Args:
        field (jnp.ndarray): _description_
        coordinate_original_each_axis (List[jnp.ndarray]): 1D arrays. Original grids for each axis.
        coordinate_new_each_axis (List[jnp.ndarray]): 1D arrays. New grids for each axis.
        axis (Union[int, Tuple[int, ...]], optional): The axes along which to shift. If None, shifts along all axes. Defaults to None.
        with_sharding: JAX sharding (e.g. PositionalSharding) to run the entire shifting process on.
            If given, field and coordinate arrays are placed on this sharding before computation.
    """
    if axis is None:
        axis = list(range(field.ndim))
    else:
        axis = np.mod(np.asarray(axis, dtype=int).flatten(), field.ndim).tolist()
    coordinate_original_each_axis = [jnp.asarray(c, dtype=jnp.float64).flatten() for c in coordinate_original_each_axis]
    coordinate_new_each_axis = [jnp.asarray(c, dtype=jnp.float64).flatten() for c in coordinate_new_each_axis]
    assert len(coordinate_original_each_axis) == len(coordinate_new_each_axis) == len(axis), f"Length of coordinate_original_each_axis {len(coordinate_original_each_axis)}, coordinate_new_each_axis {len(coordinate_new_each_axis)}, and axis {len(axis)} must all be the same."
    for i, ax in enumerate(axis):
        assert coordinate_original_each_axis[i].size == field.shape[ax], f"Size of coordinate_original_each_axis[{i}] {coordinate_original_each_axis[i].size} must match field size {field.shape[ax]} along axis {ax}."

    print('Original shape:', field.shape)

    dr_0 = [(c[-1] - c[0]) / (c.size - 1) if c.size > 1 else 1.0 for c in coordinate_original_each_axis]
    dr_1 = [(c[-1] - c[0]) / (c.size - 1) if c.size > 1 else 1.0 for c in coordinate_new_each_axis]

    # Determine new field shape
    new_shape = list(field.shape)
    for i, ax in enumerate(axis):
        new_shape[ax] = coordinate_new_each_axis[i].size

    # Build 1D fractional-index arrays for each field dimension.
    # Shifted axes: map new coordinates into original grid indices.
    # Unshifted axes: identity (0, 1, ..., size-1).
    coord_arrays = []
    for dim in range(field.ndim):
        if dim in axis:
            i = axis.index(dim)
            c0 = coordinate_original_each_axis[i]
            c1 = coordinate_new_each_axis[i]
            coord_arrays.append((c1 - c0[0]) / dr_0[i])
        else:
            coord_arrays.append(jnp.arange(field.shape[dim], dtype=jnp.float64))

    grids = jnp.array(jnp.meshgrid(*coord_arrays, indexing='ij'))   # shape (ndim, *new_shape)

    if with_sharding is not None:
        field = jax.device_put(field, with_sharding)
        grids = jax.device_put(grids, with_sharding)

    field_1 = _shift_field(field, grids)

    print('New shape:', field_1.shape)
    print('Square integrate (should be close to 1.0)')
    print(square_integral_field(Field=field_1, dr=dr_1) / square_integral_field(Field=field, dr=dr_0))

    return field_1






if __name__ == '__main__':
    import math

    def expected_out_size(Ni):
        """Smallest multiple of 16 strictly greater than 2*Ni."""
        return (2 * Ni // 16 + 1) * 16

    # ------------------------------------------------------------------ #
    # Example 1: small axis (Ni < 64) → no padding, no smoothing
    # ------------------------------------------------------------------ #
    print("=== Example 1: small axis (Ni=32 < 64), no padding ===")
    Ni = 32
    field1 = jnp.ones((Ni,))
    x1 = jnp.linspace(0.0, 1.0, Ni)
    fp1, coords1, slices1 = pad_for_fft(field1, axis=0, coordinate_each_axis=[x1])
    assert fp1.shape == (Ni,), f"Expected shape ({Ni},), got {fp1.shape}"
    assert slices1 == (slice(0, Ni),)
    assert jnp.allclose(coords1[0], x1)
    assert jnp.allclose(fp1[slices1], field1)
    print(f"  Ni={Ni}, out_size={fp1.shape[0]}  (expected {Ni}, no change)")
    print(f"  pad_slices={slices1}")
    print(f"  coordinate range: [{coords1[0][0]:.3f}, {coords1[0][-1]:.3f}]")
    print("  PASSED\n")

    # ------------------------------------------------------------------ #
    # Example 2: large axis (Ni=100 >= 64) → padded and centred
    # ------------------------------------------------------------------ #
    print("=== Example 2: large axis (Ni=100 >= 64) ===")
    Ni = 100
    expected = expected_out_size(Ni)
    field2 = jnp.arange(Ni, dtype=jnp.float64)
    x2 = jnp.linspace(-5.0, 5.0, Ni)
    fp2, coords2, slices2 = pad_for_fft(field2, axis=0, coordinate_each_axis=[x2])
    assert fp2.shape[0] == expected, f"Expected out_size={expected}, got {fp2.shape[0]}"
    assert fp2.shape[0] % 16 == 0
    assert fp2.shape[0] > 2 * Ni
    total_pad = expected - Ni
    pad_before = total_pad // 2
    assert slices2 == (slice(pad_before, pad_before + Ni),)
    assert jnp.allclose(fp2[slices2[0]], field2 * tukey(Ni, alpha=0.1))
    dx = x2[1] - x2[0]
    assert jnp.allclose(coords2[0][slices2[0]], x2)
    assert jnp.isclose(coords2[0][1] - coords2[0][0], dx)
    print(f"  Ni={Ni}, out_size={fp2.shape[0]}  (expected {expected})")
    print(f"  pad_before={pad_before}, pad_after={total_pad - pad_before}")
    print(f"  pad_slices={slices2}")
    print(f"  coordinate range: [{coords2[0][0]:.3f}, {coords2[0][-1]:.3f}]  (original: [{x2[0]:.3f}, {x2[-1]:.3f}])")
    print("  PASSED\n")

    # ------------------------------------------------------------------ #
    # Example 3: large axis exactly at a 16-multiple (Ni=128)
    #            2*128=256 is divisible by 16 → out_size must still be > 256
    # ------------------------------------------------------------------ #
    print("=== Example 3: Ni=128, 2*Ni=256 divisible by 16 ===")
    Ni = 128
    expected = expected_out_size(Ni)
    field3 = jnp.ones((Ni,))
    x3 = jnp.linspace(0.0, 1.0, Ni)
    fp3, coords3, slices3 = pad_for_fft(field3, axis=0, coordinate_each_axis=[x3])
    assert fp3.shape[0] == expected
    assert fp3.shape[0] > 2 * Ni
    assert fp3.shape[0] % 16 == 0
    print(f"  Ni={Ni}, 2*Ni={2*Ni}, out_size={fp3.shape[0]}  (expected {expected})")
    print(f"  pad_slices={slices3}")
    print("  PASSED\n")

    # ------------------------------------------------------------------ #
    # Example 4: 2D field, both axes >= 64
    # ------------------------------------------------------------------ #
    print("=== Example 4: 2D field (Nx=80, Ny=100), both padded ===")
    Nx, Ny = 80, 100
    field4 = jnp.ones((Nx, Ny))
    x4 = jnp.linspace(-1.0, 1.0, Nx)
    y4 = jnp.linspace(-2.0, 2.0, Ny)
    fp4, coords4, slices4 = pad_for_fft(field4, axis=(0, 1), coordinate_each_axis=[x4, y4])
    ex, ey = expected_out_size(Nx), expected_out_size(Ny)
    assert fp4.shape == (ex, ey), f"Expected ({ex},{ey}), got {fp4.shape}"
    assert jnp.allclose(fp4[slices4], field4 * tukey(Nx, alpha=0.1)[:, None] * tukey(Ny, alpha=0.1)[None, :])
    print(f"  Input shape: ({Nx},{Ny}), padded shape: {fp4.shape}  (expected ({ex},{ey}))")
    print(f"  pad_slices={slices4}")
    print(f"  x range: [{coords4[0][0]:.3f}, {coords4[0][-1]:.3f}]")
    print(f"  y range: [{coords4[1][0]:.3f}, {coords4[1][-1]:.3f}]")
    print("  PASSED\n")

    # ------------------------------------------------------------------ #
    # Example 5: mixed — axis 0 small (Ni=30), axis 1 large (Ni=100)
    # ------------------------------------------------------------------ #
    print("=== Example 5: mixed 2D (Nx=30 < 64, Ny=100 >= 64) ===")
    Nx, Ny = 30, 100
    field5 = jnp.ones((Nx, Ny))
    x5 = jnp.linspace(0.0, 1.0, Nx)
    y5 = jnp.linspace(-3.0, 3.0, Ny)
    fp5, coords5, slices5 = pad_for_fft(field5, axis=(0, 1), coordinate_each_axis=[x5, y5])
    ey = expected_out_size(Ny)
    assert fp5.shape == (Nx, ey), f"Expected ({Nx},{ey}), got {fp5.shape}"
    assert slices5[0] == slice(0, Nx)        # axis 0: no padding
    assert slices5[1].start > 0              # axis 1: padded
    assert jnp.allclose(coords5[0], x5)     # axis 0 coord unchanged
    print(f"  Input shape: ({Nx},{Ny}), padded shape: {fp5.shape}  (expected ({Nx},{ey}))")
    print(f"  pad_slices={slices5}")
    print(f"  x range: [{coords5[0][0]:.3f}, {coords5[0][-1]:.3f}]  (no change expected)")
    print(f"  y range: [{coords5[1][0]:.3f}, {coords5[1][-1]:.3f}]  (original: [{y5[0]:.3f}, {y5[-1]:.3f}])")
    print("  PASSED\n")

    # ------------------------------------------------------------------ #
    # Example 6: 3D field, pad only one axis using negative index
    # ------------------------------------------------------------------ #
    print("=== Example 6: 3D field (10,20,200), pad only last axis (axis=-1) ===")
    field6 = jnp.ones((10, 20, 200))
    z6 = jnp.linspace(0.0, 1.0, 200)
    fp6, coords6, slices6 = pad_for_fft(field6, axis=-1, coordinate_each_axis=[z6])
    ez = expected_out_size(200)
    assert fp6.shape == (10, 20, ez), f"Expected (10,20,{ez}), got {fp6.shape}"
    assert fp6.shape[2] % 16 == 0 and fp6.shape[2] > 400
    print(f"  Input shape: {field6.shape}, padded shape: {fp6.shape}  (expected (10,20,{ez}))")
    print(f"  pad_slices={slices6}")
    print(f"  z range: [{coords6[0][0]:.4f}, {coords6[0][-1]:.4f}]  (original: [{z6[0]:.4f}, {z6[-1]:.4f}])")
    print("  PASSED\n")

    print("All examples passed.")


