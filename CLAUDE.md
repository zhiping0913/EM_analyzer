# EM_analyzer

JAX-based toolkit for electromagnetic field generation, spectral Maxwell propagation, envelope/spectrum analysis, and plotting.

Root: `/scratch/gpfs/MIKHAILOVA/zl8336/EM_analyzer`

## sys.path setup (required in every script)

```python
import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336')
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/EM_analyzer')
```

JAX config (set BEFORE importing `Normal_variable_method`):

```python
import jax
jax.config.update('jax_num_cpu_devices', 6)
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
```

---

## 1. Field generation — `Spectral_Maxwell/Generate_Fields/Gaussian_beam_2D.py`

Non-paraxial 2D Gaussian pulse in the physical X–Z plane (x_phys = transverse, z_phys = propagation).

```python
import EM_analyzer.Spectral_Maxwell.Generate_Fields.Gaussian_beam_2D as _GB2D_module
from EM_analyzer.Spectral_Maxwell.Generate_Fields.Gaussian_beam_2D import Gaussian_Beam_2D

# get_pulse() writes .nc/.txt to the module-level working_dir — override first:
_GB2D_module.working_dir = '/your/output/dir'

beam = Gaussian_Beam_2D(
    wavelength   = 0.8e-6,   # m
    w0_lambda    = 3.0,      # waist in units of λ
    phi_pol      = 0.0,      # polarization angle (rad)
    phi_cep      = 0.0,      # CEP (rad)
    a0           = 1.0,      # E_peak / E_c
    r_resolution = 50.0,     # λ/dx
    k_resolution = 50.0,     # k0/dkx
)
EB_dict = beam.get_pulse(FWHM_time=8e-15, time_shift=0.0, theta=0.0)
```

Returned dict:

| key | shape | unit |
|---|---|---|
| `E` | `(3, Nx_phys, 1, Nz_phys)` | V/m |
| `B` | `(3, Nx_phys, 1, Nz_phys)` | T |
| `x_coordinate` | `(Nx_phys,)` (transverse) | m |
| `y_coordinate` | `(1,)` (always `[0.]`) | m |
| `z_coordinate` | `(Nz_phys,)` (propagation) | m |

Useful attributes: `beam.period`, `beam.k0`, `beam.omega0`, `beam.Ec`, `beam.Bc`, `beam.w0`, `beam.z_R`.

---

## 2. Spectral Maxwell solver — `Spectral_Maxwell/Normal_variable_method.py`

Exact (non-paraxial) k-space evolution to any time. Solver lives in the X–Y plane.

```python
from EM_analyzer.Spectral_Maxwell.Normal_variable_method import Spectral_Maxwell_Solver_2D

solver = Spectral_Maxwell_Solver_2D(
    E0x=..., E0y=..., E0z=...,   # each (Nx, Ny), V/m
    B0x=..., B0y=..., B0z=...,   # each (Nx, Ny), T
    x_coordinate=..., y_coordinate=...,  # m
)
result = solver.evolution(
    evolution_time        = 10*T0,                 # s (can be negative)
    window_shift_velocity = (C.speed_of_light, 0), # m/s; shifts output coords
)
# result keys: 'Ex','Ey','Ez','Bx','By','Bz','x_coordinate','y_coordinate'
```

### Coordinate mapping: Gaussian_Beam_2D → Solver_2D

Beam is in x_phys–z_phys; solver is in x_sol–y_sol. Map with a transpose to preserve ∇·E=0:

```
x_sol ← z_phys (propagation)        y_sol ← x_phys (transverse)
E^x_sol = E_phys[2,:,0,:].T         B^x_sol = B_phys[2,:,0,:].T
E^y_sol = E_phys[0,:,0,:].T  (main) B^y_sol = B_phys[0,:,0,:].T
E^z_sol = E_phys[1,:,0,:].T         B^z_sol = B_phys[1,:,0,:].T  (main, out-of-plane B)
```

After `evolution()`, `result['Ey']` is the main oscillating field, shape `(Nz_phys, Nx_phys)`.

---

## 3. Spectrum / envelope — `spectrum.py`

```python
from EM_analyzer.spectrum import (
    get_spectrum_from_field_with_coordinate,
    get_field_from_spectrum_with_coordinate,
    get_envelope_from_field_with_coordinate,
    get_envelope_from_spectrum_with_coordinate,
    filter_spectrum_by_k_coordinate,
)
import jax.numpy as jnp

# Forward FFT (returns 0-centered spectrum + physical k axes + pad info)
spectrum, k_coords, pad_slices = get_spectrum_from_field_with_coordinate(
    field=field, axis=(0, 1), r_coordinate_each_axis=[x_arr, y_arr],
)

# Inverse FFT
field_back = get_field_from_spectrum_with_coordinate(
    spectrum=spectrum, axis=(0, 1),
    k_coordinate_each_axis=k_coords, pad_slices=pad_slices, real=True,
)

# Envelope + instantaneous phase via analytic signal (Hilbert in Fourier space)
envelope, phase = get_envelope_from_field_with_coordinate(
    field                  = jnp.array(Ey),   # any shape
    axis                   = (0,),            # axes to FFT
    axis_hilbert           = 0,               # axis carrying the carrier
    r_coordinate_each_axis = [jnp.array(x_c)],
)

# Bandpass in k-space
spectrum_bp = filter_spectrum_by_k_coordinate(
    spectrum=spectrum, k_coordinate_each_axis=k_coords,
    axis=(0, 1), k_min=0.5e6, k_max=2e7,
)
```

For a beam propagated with `Spectral_Maxwell_Solver_2D`, the carrier oscillates along the propagation axis (`axis=0`), so use `axis=(0,)`, `axis_hilbert=0`, `r_coordinate_each_axis=[x_c]`.

---

## 4. Plotting — `plot/plot_2D.py`

```python
from EM_analyzer.plot.plot_2D import plot_2D_field

plot_2D_field(
    field        = Ey / Ec,                    # (Nx, Ny)
    x_coordinate = x_c / laser_lambda,         # horizontal axis
    y_coordinate = y_c / laser_lambda,         # vertical axis
    vmin=-1.0, vmax=1.0,
    label  = r'$E_y/E_c$',
    xlabel = r'$z/\lambda_0$',
    ylabel = r'$x/\lambda_0$',
    step_x = 2, step_y = 4,                    # downsample for speed
    return_fig  = False,                       # False → save PNG and close
    name        = f'Ey_t={t/T0:+08.2f}T0',
    working_dir = working_dir,
)
# return_fig=True → returns dict with 'fig', 'ax_main' for further editing
```

Convention: `field[i, j]` → `x_coordinate[i]` (horizontal), `y_coordinate[j]` (vertical).

Helpers in `plot/plot_basic.py`:

```python
from EM_analyzer.plot.plot_basic import savefig, slice_between, phase_amp_to_rgb
sl = slice_between(coord, min=-10.0, max=10.0)       # crop a coord range
rgb = phase_amp_to_rgb(phase=np.angle(z), amplitude=np.abs(z))  # complex → RGB
```

---

## 5. I/O — `read_write.py`

```python
from EM_analyzer.read_write import copy_to_dev_shm, cleanup_file, read_dat, write_fields_to_nc
# copy_to_dev_shm: cache .sdf in /dev/shm for fast reads; cleanup_file deletes it.
# read_dat: load a binary .dat with given shape.
# write_fields_to_nc: dump an EB_dict to a NetCDF file (xarray backend).
```

---

## 6. FFT backend — `fft_backend.py`

0-centered FFT wrappers (handle `fftshift`/`ifftshift` automatically):

```python
from EM_analyzer.fft_backend import fftn, ifftn, fftfreq
F = fftn(a, axes=(0, 1))      # 0-centered output
a = ifftn(F, axes=(0, 1))
f = fftfreq(N, d=dx)          # 0-centered frequency axis
```

---

## 7. Other modules

- `pretreat_fields.py` — `outer_product`, `pad_for_fft`, `get_norm`, `square_integral_field`, `check_divergence`, `get_closest_coordinate_id`, Tukey-window helpers.
- `rotate_3D.py` — `Rotation` class for 3D rigid-body field rotation via Euler angles + `map_coordinates` (multi-device JAX sharding).
- `Lorentz/Lorentz_transform.py` — `LorentzTransform(vx, vy, vz)` for boosting fields/4-vectors between inertial frames.
- `Spectral_Maxwell/Angular_spectrum_method.py` — `Vector_angular_spectrum` for stationary monochromatic propagation.
- `Spectral_Maxwell/kgrid.py` — `make_k_coordinate_from_r_coordinate`, `make_r_coordinate_from_k_coordinate`, `grid_k`.

---

## 8. Units reference

```python
import scipy.constants as C, numpy as np
laser_lambda = 0.8e-6
omega0 = 2*np.pi*C.speed_of_light / laser_lambda
Ec     = (C.m_e * omega0 / C.elementary_charge) * C.speed_of_light  # V/m
Bc     =  C.m_e * omega0 / C.elementary_charge                       # T
T0     = laser_lambda / C.speed_of_light                              # s
```

| symbol | meaning | expression |
|---|---|---|
| ω₀ | angular frequency | 2πc/λ |
| Ec | critical E-field | m_e·ω₀·c/e |
| Bc | critical B-field | m_e·ω₀/e |
| a₀ | normalized amplitude | E_peak / Ec |
| z_R | Rayleigh length | π w₀² / λ |
| w₀ | 1/e intensity waist | `w0_lambda * λ` |
