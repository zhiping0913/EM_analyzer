#import pyfftw.interfaces.numpy_fft as fft
#import numpy.fft as fft
from functools import partial
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy.fft as fft
import jax.numpy as jnp

"""
    From LightPipes documentation:
    Very useful comment on fftshift and ifftshift found in 
    https://github.com/numpy/numpy/issues/13442
    with x=real and X=fourier space:
        x = ifft(fft(x))
        X = fft(ifft(X))
    and both 0-centered in middle of array:
    ->
    X = fftshift(fft(ifftshift(x)))  # correct magnitude and phase
    x = fftshift(ifft(ifftshift(X)))  # correct magnitude and phase
    X = fftshift(fft(x))  # correct magnitude but wrong phase !
    x = fftshift(ifft(X))  # correct magnitude but wrong phase !
"""
fftshift = fft.fftshift
ifftshift = fft.ifftshift
fftfreq = fft.fftfreq

@partial(jax.jit, static_argnames=('s','axes'))
def fftn(a, s=None, axes=None):
    if axes is None:
        axes = tuple(range(a.ndim))
    axes=np.mod(np.asarray(axes).flatten(), a.ndim)
    return fft.fftshift(fft.fftn(fft.ifftshift(a, axes=axes), s=s, axes=axes), axes=axes)

@partial(jax.jit, static_argnames=('s','axes'))
def ifftn(a, s=None, axes=None):
    if axes is None:
        axes = tuple(range(a.ndim))
    axes=np.mod(np.asarray(axes).flatten(), a.ndim)
    return fft.fftshift(fft.ifftn(fft.ifftshift(a, axes=axes), s=s, axes=axes), axes=axes)


def fftfreq(n, d=1.0):
    return fft.fftshift(fft.fftfreq(n, d=d))


