import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336')
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/EM_analyzer')
from line_profiler import profile
import numpy as np
import jax
from wrapt import partial
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
from scipy.signal.windows import tukey
import scipy.constants as C
import jax.numpy as jnp
from typing import List, Optional, Tuple, Union
import string
import xarray as xr
from EM_analyzer.fft_backend import fftn,ifftn,fftfreq
from EM_analyzer.pretreat_fields import pad_for_fft,get_norm,print_shard_layout
from EM_analyzer.Spectral_Maxwell.kgrid import make_k_coordinate_from_r_coordinate,make_r_coordinate_from_k_coordinate
from scipy.signal import hilbert


@partial(jax.jit, static_argnames=('axis'))
def _get_analytic_spectrum_from_field_spectrum(
    spectrum:jnp.ndarray,
    axis: int,
    ):
    """
    Compute the analytic signal from a 0-centered spectrum in Fourier space.
    Args:
        spectrum (jnp.ndarray): 0 centered spectrum in Fourier space.
        axis (int): Axis along which to compute the analytic signal.
    """
    ndim=spectrum.ndim
    shape=spectrum.shape
    freq=fftfreq(n=shape[axis], d=1.0)
    h=1+jnp.sign(freq)
    expand_axis=list(range(ndim))
    expand_axis.pop(axis)
    h=jnp.expand_dims(h, axis=expand_axis)
    analytic_spectrum=spectrum*h
    return analytic_spectrum

def get_analytic_spectrum_from_field_spectrum(
    spectrum:jnp.ndarray,
    axis: Optional[int]=-1,
    ):
    """
    Compute the analytic signal from a 0-centered spectrum in Fourier space.
    Args:
        spectrum (jnp.ndarray): 0 centered spectrum in Fourier space.
        axis (Optional[int], optional): Axis along which to compute the analytic signal. Defaults to -1.
    """
    ndim=spectrum.ndim
    axis=np.mod(np.asarray(axis,dtype=int).flatten()[0], ndim)
    return _get_analytic_spectrum_from_field_spectrum(spectrum=spectrum, axis=axis)

def get_spectrum_from_field_with_coordinate(
    field: jnp.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]]=None,
    r_coordinate_each_axis: Optional[List[jnp.ndarray]]=None,
    out_sharding=None,
    pad=True,
    ):
    """
    Compute the spectrum of a field in Fourier space.
    Parameters
    ----------
    field : jnp.ndarray
        Input field to be processed.
    r_coordinate_each_axis : List[jnp.ndarray]
        List of real space coordinate arrays for each axis. The length of this list should match the number of axes specified in the axis argument.
        If r_coordinate_each_axis is None, it will be set to a list of jnp.arange(shape[axis_i]) for each axis in axis argument.
        len(r_coordinate_each_axis) must match len(axis) and the size of each coordinate array must match the corresponding dimension of the field.
    axis : int or tuple of ints
        Axis or axes along which to compute the spectrum. If an int is provided, it is converted to a tuple with one element.
        If axis is None, then the spectrum is computed along all axes of the input field.
    """
    field=jnp.asarray(field)
    shape=field.shape
    ndim=len(shape)
    if axis is None:
        axis = tuple(range(ndim))
    axis = tuple(np.mod(np.asarray(axis, dtype=int).flatten(), ndim))
    assert len(axis) > 0, "At least one axis must be specified."
    if r_coordinate_each_axis is None:
        r_coordinate_each_axis=[jnp.arange(shape[axis_i]) for axis_i in axis]
    assert len(r_coordinate_each_axis) == len(axis), "The length of r_coordinate_each_axis must match the length of axis."
    if pad:
        field_pad, coordinate_pad_list, pad_slices=pad_for_fft(
            field=field,
            axis=axis,
            coordinate_each_axis=r_coordinate_each_axis,
        )
    else:
        field_pad=field
        coordinate_pad_list=r_coordinate_each_axis
        pad_slices=tuple(slice(None) for _ in axis)
    if out_sharding is not None:
        field_pad = jax.device_put(field_pad, out_sharding)
    print_shard_layout(field_pad, name="field_pad")
    k_coordinate_each_axis=[]
    dr_each_axis=[]
    for coordinate_pad in coordinate_pad_list:
        k_coordinate, dk, dr=make_k_coordinate_from_r_coordinate(coordinate_pad)
        k_coordinate_each_axis.append(k_coordinate)
        dr_each_axis.append(dr)
    spectrum=fftn(field_pad, axes=axis)*jnp.prod(jnp.array(dr_each_axis))
    print_shard_layout(spectrum, name="spectrum")
    return spectrum, k_coordinate_each_axis, pad_slices

def get_field_from_spectrum_with_coordinate(
    spectrum: jnp.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]]=None,
    k_coordinate_each_axis: Optional[List[jnp.ndarray]]=None,
    pad_slices: Optional[Tuple[slice, ...]]=None,
    real=True,
    ):
    """
    Compute the field from its spectrum in Fourier space.
    Parameters
    ----------
    spectrum : jnp.ndarray
        Input spectrum in Fourier space.
    axis : int or tuple of ints
        Axis or axes along which to compute the inverse FFT. If an int is provided, it is converted to a tuple with one element.
        If axis is None, the inverse FFT is computed along all axes of the input spectrum.
    k_coordinate_each_axis : List[jnp.ndarray]
        List of k-space coordinate arrays for each axis in `axis`.
        If None, defaults to jnp.fftfreq(n=shape[axis_i], d=1.0) for each axis.
    pad_slices : tuple of slices or None
        Slices (one per entry in `axis`) indicating the region of the reconstructed
        field that corresponds to the original (unpadded) data, as returned by
        pad_for_fft.  If None, the full array is returned (no slicing).
    real : bool, optional
        If True, return the real part of the reconstructed field. Default is True.
    Returns
    -------
    field : jnp.ndarray
        Reconstructed field in real space (padding removed when pad_slices is given).
    """
    spectrum=jnp.asarray(spectrum,dtype=jnp.complex128)
    shape=spectrum.shape
    ndim=spectrum.ndim
    if axis is None:
        axis = tuple(range(ndim))
    axis = tuple(np.mod(np.asarray(axis, dtype=int).flatten(), ndim))
    assert len(axis) > 0, "At least one axis must be specified."
    if k_coordinate_each_axis is None:
        k_coordinate_each_axis=[fftfreq(n=shape[axis_i], d=1.0) for axis_i in axis]
    assert len(k_coordinate_each_axis) == len(axis), "The length of k_coordinate_each_axis must match the length of axis."
    dk_list=[]
    N_list=[]
    for k_coordinate_i, axis_i in zip(k_coordinate_each_axis, axis):
        k_coordinate_i=jnp.asarray(k_coordinate_i)
        assert k_coordinate_i.size == shape[axis_i], f"The shape of k_coordinate must match the corresponding dimension of the spectrum. Expected shape: {shape[axis_i]}, but got {k_coordinate_i.size}."
        _, _, dk=make_r_coordinate_from_k_coordinate(k_coordinate_i)
        dk_list.append(dk)
        N_list.append(shape[axis_i])
    #dr*dk=2π/N
    # Keep as jax.Array (do NOT call device_get): in multi-process / multi-node
    # runs the array spans non-addressable devices, so materializing here would
    # raise. The caller can do its own process_allgather if it needs numpy.
    field_pad=ifftn(spectrum, axes=axis)*jnp.prod(jnp.array(dk_list)/(2*jnp.pi))*jnp.prod(jnp.array(N_list))
    if pad_slices is None:
        field=field_pad
    else:
        assert len(pad_slices) == len(axis), "The length of pad_slices must match the length of axis."
        slice_list=[slice(None)] * ndim
        for i, ax in enumerate(axis):
            slice_list[ax]=pad_slices[i]
        field=field_pad[tuple(slice_list)]
    if real:
        return jnp.real(field)
    else:
        return field
def get_field_analytic_from_spectrum_with_coordinate(
    spectrum: jnp.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]]=None,
    axis_hilbert: Optional[int]=0,
    k_coordinate_each_axis: Optional[List[jnp.ndarray]]=None,
    pad_slices: Optional[Tuple[slice, ...]]=None,
    ):
    """
    Compute the (complex) analytic signal in real space from a 0-centered
    spectrum. Applies the analytic-spectrum filter (1 + sign(freq)) along
    `axis_hilbert`, then inverse-FFTs along `axis` with `real=False`.

    Args:
        spectrum (jnp.ndarray): 0-centered spectrum in Fourier space.
        axis (Optional[Union[int, Tuple[int, ...]]]): Axes to inverse-FFT.
        axis_hilbert (Optional[int]): Axis carrying the carrier; analytic
            filter is applied along this axis. Defaults to 0.
        k_coordinate_each_axis (Optional[List[jnp.ndarray]]): Physical k-axes
            for each axis in `axis`.
        pad_slices (Optional[Tuple[slice, ...]]): Slices to strip the FFT
            padding from the reconstructed field, one per axis in `axis`.

    Returns:
        jnp.ndarray: Complex analytic field with the original (unpadded) shape.
    """
    spectrum_analytic=get_analytic_spectrum_from_field_spectrum(spectrum=spectrum, axis=axis_hilbert)
    field_analytic=get_field_from_spectrum_with_coordinate(
        spectrum=spectrum_analytic,
        axis=axis,
        k_coordinate_each_axis=k_coordinate_each_axis,
        pad_slices=pad_slices,
        real=False,
    )
    return field_analytic


def get_envelope_from_spectrum_with_coordinate(
    spectrum: jnp.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]]=None,
    axis_hilbert: Optional[int]=0,
    k_coordinate_each_axis: Optional[List[jnp.ndarray]]=None,
    pad_slices: Optional[Tuple[slice, ...]]=None,
    ):
    """
    Envelope (|·|) and instantaneous phase (angle) of the analytic signal
    obtained from `spectrum` via get_field_analytic_from_spectrum_with_coordinate.
    """
    field_analytic = get_field_analytic_from_spectrum_with_coordinate(
        spectrum=spectrum,
        axis=axis,
        axis_hilbert=axis_hilbert,
        k_coordinate_each_axis=k_coordinate_each_axis,
        pad_slices=pad_slices,
    )
    return jnp.abs(field_analytic), jnp.angle(field_analytic)

def get_envelope_from_field_with_coordinate(
    field: jnp.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]]=None,
    axis_hilbert: Optional[int]=0,
    r_coordinate_each_axis: Optional[List[jnp.ndarray]]=None,
    out_sharding=None,
    ):
    """_summary_

    Args:
        field (jnp.ndarray): _description_
        axis (Optional[Union[int, Tuple[int, ...]]], optional): _description_. Defaults to None.
        axis_hilbert (Optional[int], optional): _description_. Defaults to 0.
        r_coordinate_each_axis (Optional[List[jnp.ndarray]], optional): _description_. Defaults to None.
        out_sharding (_type_, optional): _description_. Defaults to None.
    """
    spectrum, k_coordinate_each_axis, pad_slices=get_spectrum_from_field_with_coordinate(
        field=field,
        axis=axis,
        r_coordinate_each_axis=r_coordinate_each_axis,
        out_sharding=out_sharding,
    )
    envelope, phase=get_envelope_from_spectrum_with_coordinate(
        spectrum=spectrum,
        axis=axis,
        axis_hilbert=axis_hilbert,
        k_coordinate_each_axis=k_coordinate_each_axis,
        pad_slices=pad_slices,
    )
    return envelope, phase



def filter_spectrum_by_k_coordinate(
    spectrum: jnp.ndarray,
    k_coordinate_each_axis: List[jnp.ndarray],
    axis: Union[int, Tuple[int, ...]],
    k_min: Optional[float]=0,
    k_max: Optional[float]=100000000000,
    ):
    """
    Filter the spectrum by k coordinate.
    Parameters
    ----------
    spectrum : jnp.ndarray
        Spectrum to be filtered.
    k_coordinate_each_axis : List[jnp.ndarray]
        List of k coordinate arrays for each axis. The length of this list should match the number of axes in the spectra.
        Each k coordinate array should have a shape that matches the corresponding dimension of the spectra.
    k_min : Optional[float], optional
        Minimum k value for filtering. Default is 0.
    k_max : Optional[float], optional
        Maximum k value for filtering. Default is 100000000000.
    Returns
    -------
    spectrum_filter : jnp.ndarray
        Filtered spectrum with the same shape as the input spectrum, but with values set to zero outside the specified k ranges.
    """
    spectrum=jnp.asarray(spectrum,dtype=jnp.complex128)
    shape=spectrum.shape
    ndim=spectrum.ndim
    if axis is None:
        axis = tuple(range(ndim))
    axis = tuple(np.mod(np.asarray(axis, dtype=int).flatten(), ndim))
    assert len(axis) > 0, "At least one axis must be specified."
    assert len(k_coordinate_each_axis) == len(axis), "The length of k_coordinate_each_axis must match the length of axis."
    k_coordinate_list=[]
    for k_coordinate_i, axis_i in zip(k_coordinate_each_axis, axis):
        k_coordinate_i=jnp.asarray(k_coordinate_i)
        assert k_coordinate_i.size == shape[axis_i], f"The shape of k_coordinate must match the corresponding dimension of the spectrum. Expected shape: {shape[axis_i]}, but got {k_coordinate_i.size}."
        k_coordinate_list.append(k_coordinate_i)
    k_grid=jnp.stack(jnp.meshgrid(*k_coordinate_list, indexing='ij'), axis=0)
    k_abs=get_norm(k_grid, axis=0)
    mask=(k_abs>=k_min) & (k_abs<=k_max)
    spectrum_filter=spectrum*mask
    return spectrum_filter


@partial(jax.jit)
def get_energy_flux_from_field_analytic(
    Electric_Field_analytic:jnp.ndarray,
    Magnetic_Field_analytic:jnp.ndarray,
    ):
    """
    Compute the (average) energy flux (Poynting vector) from the analytic signal of the electric and magnetic fields.
    Parameters
    ----------
    Electric_Field_analytic : jnp.ndarray
        Analytic signal of the electric field in Fourier space. The shape should be (3, ...), where the first dimension corresponds to the components of the electric field (Ex, Ey, Ez).
    Magnetic_Field_analytic : jnp.ndarray
        Analytic signal of the magnetic field in Fourier space. The shape should be (3, ...), where the first dimension corresponds to the components of the magnetic field (Bx, By, Bz).
    Returns
    -------
    Poynting_vector : jnp.ndarray
        Energy flux (Poynting vector) computed from the analytic signals of the electric and magnetic fields. The shape will be the same as the input fields, but without the first dimension (i.e., ...).
    """
    Poynting_vector=jnp.real(jnp.cross(Electric_Field_analytic,jnp.conj(Magnetic_Field_analytic),axis=0))/(2*C.mu_0)
    return Poynting_vector


def get_energy_flux_from_field(
    Electric_Field:jnp.ndarray,
    Magnetic_Field:jnp.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]]=None,
    axis_hilbert: Optional[int]=0,
    r_coordinate_each_axis: Optional[List[jnp.ndarray]]=None,
    out_sharding=None,
    ):
    """
    Compute the time-averaged energy flux (Poynting vector) directly from the
    real-valued E and B fields. Internally builds the analytic signals via the
    Hilbert transform in Fourier space along `axis_hilbert` and then calls
    get_energy_flux_from_field_analytic.

    Parameters
    ----------
    Electric_Field, Magnetic_Field : jnp.ndarray
        Shape (3, ...): first dim is the vector component (x, y, z); remaining
        dims are the spatial / temporal grid.
    axis : int or tuple of ints
        Axes along which to FFT/IFFT. Indexed against the input shape (so for
        a (3, Nx) input the spatial axis is 1, not 0).
    axis_hilbert : int
        Which of the FFT'd axes carries the optical carrier (gets the analytic
        signal filter). Indexed against the input shape.
    r_coordinate_each_axis : list of jnp.ndarray, optional
        Real-space coordinate arrays, one per axis in `axis`.
    out_sharding : optional
        Sharding to apply to the padded field inside the spectrum step.

    Returns
    -------
    Poynting_vector : jnp.ndarray
        Shape (3, ...), same trailing dims as the inputs.
    """
    Electric_Field = jnp.asarray(Electric_Field)
    Magnetic_Field = jnp.asarray(Magnetic_Field)
    assert Electric_Field.shape[0] == 3, (
        f"Electric_Field.shape[0] must be 3 (vector component), got {Electric_Field.shape[0]}"
    )
    assert Magnetic_Field.shape[0] == 3, (
        f"Magnetic_Field.shape[0] must be 3 (vector component), got {Magnetic_Field.shape[0]}"
    )
    assert Electric_Field.shape == Magnetic_Field.shape, (
        f"Electric_Field.shape {Electric_Field.shape} and Magnetic_Field.shape "
        f"{Magnetic_Field.shape} must match."
    )
    ndim = Electric_Field.ndim
    if axis is not None:
        axis = tuple(np.mod(np.asarray(axis, dtype=int).flatten(), ndim))
        if r_coordinate_each_axis is not None:
            assert len(r_coordinate_each_axis) == len(axis), (
                f"len(r_coordinate_each_axis)={len(r_coordinate_each_axis)} must "
                f"match len(axis)={len(axis)}."
            )
            for i, ax in enumerate(axis):
                assert jnp.asarray(r_coordinate_each_axis[i]).size == Electric_Field.shape[ax], (
                    f"r_coordinate_each_axis[{i}].size="
                    f"{jnp.asarray(r_coordinate_each_axis[i]).size} must match "
                    f"field.shape[axis={ax}]={Electric_Field.shape[ax]}."
                )
    E_spectrum, k_coordinate_each_axis, pad_slices = get_spectrum_from_field_with_coordinate(
        field=Electric_Field, axis=axis,
        r_coordinate_each_axis=r_coordinate_each_axis,
        out_sharding=out_sharding,
    )
    B_spectrum, _, _ = get_spectrum_from_field_with_coordinate(
        field=Magnetic_Field, axis=axis,
        r_coordinate_each_axis=r_coordinate_each_axis,
        out_sharding=out_sharding,
    )
    E_analytic = get_field_analytic_from_spectrum_with_coordinate(
        spectrum=E_spectrum, axis=axis, axis_hilbert=axis_hilbert,
        k_coordinate_each_axis=k_coordinate_each_axis, pad_slices=pad_slices,
    )
    B_analytic = get_field_analytic_from_spectrum_with_coordinate(
        spectrum=B_spectrum, axis=axis, axis_hilbert=axis_hilbert,
        k_coordinate_each_axis=k_coordinate_each_axis, pad_slices=pad_slices,
    )
    return get_energy_flux_from_field_analytic(
        Electric_Field_analytic=E_analytic,
        Magnetic_Field_analytic=B_analytic,
    )



