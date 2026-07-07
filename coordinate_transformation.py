"""
Coordinate transformations between Cartesian (x, y[, z]) and 2-D polar (ρ, θ)
or 3-D spherical (r, θ, φ).

`polar_transformation` interpolates a field between Cartesian and 2-D polar
grids with `interpax.interp2d` (cubic by default), and — for `type='vector'`
— additionally rotates the (Fx, Fy) ↔ (F_ρ, F_θ) components at every sample
point.

`spherical_transformation` does the same for 3-D fields on a spherical grid
(polar angle θ from +z axis, azimuth φ in the xy-plane from +x), using
`interpax.interp3d`. `type='vector'` rotates (Fx, Fy, Fz) ↔ (F_r, F_θ, F_φ).

Polar convention:
    x = ρ cos θ,     y = ρ sin θ
    F_ρ =  Fx cos θ + Fy sin θ
    F_θ = -Fx sin θ + Fy cos θ
    ρ ≥ 0.  θ is in radians, treated as 2π-periodic during Polar→Cartesian.

Spherical convention:
    x = r sin θ cos φ,    y = r sin θ sin φ,    z = r cos θ
    r ≥ 0.  θ ∈ [0, π] (polar angle from +z).  φ ∈ [0, 2π) (azimuth).
    φ is treated as 2π-periodic during Spherical→Cartesian.
    Vector components:
        e_r = ( sin θ cos φ,  sin θ sin φ,  cos θ)
        e_θ = ( cos θ cos φ,  cos θ sin φ, -sin θ)
        e_φ = (-sin φ,        cos φ,        0   )
"""

from functools import partial
from typing import Union
import jax
import jax.numpy as jnp
from interpax import interp2d, interp3d

from EM_analyzer.pretreat_fields import square_integral_field


def square_integral_polar(
    field: jnp.ndarray,
    rho_coordinate: jnp.ndarray,
    theta_coordinate: jnp.ndarray,
    rho_axis: int = -2,
    theta_axis: int = -1,
):
    """
    Compute ∬|F|² ρ dρ dθ, summing every axis of `field`.

    Reuses `square_integral_field` after multiplying `field` pointwise by
    √ρ (broadcast along `rho_axis`) — the resulting |·|² therefore picks up
    the polar Jacobian |J| = ρ. The `dr` argument then supplies dρ·dθ.

    Parameters
    ----------
    field : array_like
        Any shape, with a ρ axis (length Nρ) and a θ axis (length Nθ).
    rho_coordinate : (Nρ,) non-negative, ascending
    theta_coordinate : (Nθ,) radians, ascending
    rho_axis, theta_axis : int
        Which axes of `field` are ρ and θ. Default is the last two.
    """
    field = jnp.asarray(field)
    rho   = jnp.asarray(rho_coordinate).flatten()
    theta = jnp.asarray(theta_coordinate).flatten()
    drho   = float(rho[1]   - rho[0])   if rho.size   > 1 else 1.0
    dtheta = float(theta[1] - theta[0]) if theta.size > 1 else 1.0

    rho_axis = int(rho_axis) % field.ndim
    shape_bcast = [1] * field.ndim
    shape_bcast[rho_axis] = rho.size
    rho_bcast = rho.reshape(shape_bcast)

    field_weighted = field * jnp.sqrt(rho_bcast)
    return square_integral_field(field_weighted, dr=[drho, dtheta], axis=None)


def polar_transformation(
    field: jnp.ndarray,
    x_coordinate: jnp.ndarray,
    y_coordinate: jnp.ndarray,
    rho_coordinate: jnp.ndarray,
    theta_coordinate: jnp.ndarray,
    direction: str = 'Cartesian->Polar',
    type: str = 'scalar',
    method: str = 'cubic',
    extrap: Union[bool, float] = False,
):
    """
    Resample a field between the Cartesian (x, y) grid and the polar (ρ, θ) grid.

    Parameters
    ----------
    field : array_like
        - `type='scalar'` :
            shape (Nx, Ny, ...)     for `direction='Cartesian->Polar'`
            shape (Nρ, Nθ, ...)     for `direction='Polar->Cartesian'`
        - `type='vector'` :
            shape (2, Nx, Ny, ...)  for `direction='Cartesian->Polar'`
            shape (2, Nρ, Nθ, ...)  for `direction='Polar->Cartesian'`
            The first axis is the vector component: (Fx, Fy) in Cartesian and
            (F_ρ, F_θ) in polar.
        Any trailing axes (`...`) are batched through the interpolation.
    x_coordinate     : (Nx,) array, ascending
    y_coordinate     : (Ny,) array, ascending
    rho_coordinate   : (Nρ,) array, non-negative and ascending
    theta_coordinate : (Nθ,) array, radians. Treated as 2π-periodic when the
                       source grid is polar.
    direction : {'Cartesian->Polar', 'Polar->Cartesian'}
    type      : {'scalar', 'vector'}
    method    : interpolation method for `interpax.interp2d`
                ('nearest', 'linear', 'cubic', 'cubic2', ...).
    extrap    : passed to `interpax.interp2d`. `False` (default) returns NaN
                outside the source grid.

    Returns
    -------
    array in the target grid layout:
        - scalar Cartesian→Polar :  (Nρ, Nθ, ...)
        - scalar Polar→Cartesian :  (Nx, Ny, ...)
        - vector Cartesian→Polar :  (2, Nρ, Nθ, ...) — (F_ρ, F_θ)
        - vector Polar→Cartesian :  (2, Nx, Ny, ...) — (Fx, Fy)
    """
    assert direction in ('Cartesian->Polar', 'Polar->Cartesian'), (
        f"direction must be 'Cartesian->Polar' or 'Polar->Cartesian', got {direction!r}"
    )
    assert type in ('scalar', 'vector'), (
        f"type must be 'scalar' or 'vector', got {type!r}"
    )

    field            = jnp.asarray(field)
    x_coordinate     = jnp.asarray(x_coordinate).flatten()
    y_coordinate     = jnp.asarray(y_coordinate).flatten()
    rho_coordinate   = jnp.asarray(rho_coordinate).flatten()
    theta_coordinate = jnp.asarray(theta_coordinate).flatten()

    assert bool(jnp.all(rho_coordinate >= 0)), "rho_coordinate must be non-negative."

    Nx, Ny       = x_coordinate.size, y_coordinate.size
    Nrho, Ntheta = rho_coordinate.size, theta_coordinate.size

    if direction == 'Cartesian->Polar':
        result = _cartesian_to_polar(
            field, x_coordinate, y_coordinate, rho_coordinate, theta_coordinate,
            type, method, extrap,
        )
    else:
        result = _polar_to_cartesian(
            field, x_coordinate, y_coordinate, rho_coordinate, theta_coordinate,
            type, method, extrap,
        )

    # Ratio check: integrate the input and output over their native grids and
    # print I1/I0. Faithful interpolation gives ≈ 1; NaN means the target grid
    # extends past the source grid (queries land outside → `extrap=False`).
    dx = float(x_coordinate[1] - x_coordinate[0]) if x_coordinate.size > 1 else 1.0
    dy = float(y_coordinate[1] - y_coordinate[0]) if y_coordinate.size > 1 else 1.0
    rho_axis, theta_axis = (0, 1) if type == 'scalar' else (1, 2)
    if direction == 'Cartesian->Polar':
        I0 = square_integral_field(field, dr=[dx, dy], axis=None)
        I1 = square_integral_polar(
            result, rho_coordinate, theta_coordinate,
            rho_axis=rho_axis, theta_axis=theta_axis,
        )
    else:
        I0 = square_integral_polar(
            field, rho_coordinate, theta_coordinate,
            rho_axis=rho_axis, theta_axis=theta_axis,
        )
        I1 = square_integral_field(result, dr=[dx, dy], axis=None)
    jax.debug.print('polar_transformation Integral ratio I1/I0: {ratio}', ratio=I1 / I0)
    return result


@partial(jax.jit, static_argnames=('type', 'method', 'extrap'))
def _cartesian_to_polar(
    field, x_coordinate, y_coordinate, rho_coordinate, theta_coordinate,
    type, method, extrap,
):
    Nx, Ny       = x_coordinate.size, y_coordinate.size
    Nrho, Ntheta = rho_coordinate.size, theta_coordinate.size
    # Query points on the Cartesian source grid, one per (ρ, θ) target cell:
    #   xq[i, j] = ρ[i] cos θ[j],   yq[i, j] = ρ[i] sin θ[j]
    rho_grid, theta_grid = jnp.meshgrid(rho_coordinate, theta_coordinate, indexing='ij')
    xq = (rho_grid * jnp.cos(theta_grid)).flatten()
    yq = (rho_grid * jnp.sin(theta_grid)).flatten()

    if type == 'scalar':
        assert field.shape[:2] == (Nx, Ny), (
            f"scalar Cartesian->Polar: field.shape[:2]={field.shape[:2]} must equal (Nx,Ny)=({Nx},{Ny})"
        )
        result_flat = interp2d(
            xq=xq, yq=yq,
            x=x_coordinate, y=y_coordinate, f=field,
            method=method, extrap=extrap,
        )
        trailing = field.shape[2:]
        return result_flat.reshape((Nrho, Ntheta) + trailing)

    # vector
    assert field.shape[0] == 2, (
        f"vector: field.shape[0] must be 2 (Fx, Fy), got {field.shape[0]}"
    )
    assert field.shape[1:3] == (Nx, Ny), (
        f"vector Cartesian->Polar: field.shape[1:3]={field.shape[1:3]} must equal (Nx,Ny)=({Nx},{Ny})"
    )
    # interp2d expects `f` shape (Nx, Ny, ...): move component axis to the end.
    f_moved = jnp.moveaxis(field, 0, -1)                                 # (Nx, Ny, ..., 2)
    result_flat = interp2d(
        xq=xq, yq=yq,
        x=x_coordinate, y=y_coordinate, f=f_moved,
        method=method, extrap=extrap,
    )                                                                    # (Nq, ..., 2)
    trailing = field.shape[3:]
    result = result_flat.reshape((Nrho, Ntheta) + trailing + (2,))
    result = jnp.moveaxis(result, -1, 0)                                 # (2, Nρ, Nθ, ...)

    # Vector rotation:
    #   [F_ρ]   [ cos θ   sin θ] [Fx]
    #   [F_θ] = [-sin θ   cos θ] [Fy]
    # R has axes (out_component, in_component, θ); result has axes
    # (in_component, ρ, θ, ...).
    cos_t = jnp.cos(theta_coordinate)
    sin_t = jnp.sin(theta_coordinate)
    R = jnp.stack([
        jnp.stack([ cos_t, sin_t]),
        jnp.stack([-sin_t, cos_t]),
    ])                                                                   # (2, 2, Nθ)
    return jnp.einsum('ijk,jrk...->irk...', R, result)                   # (2, Nρ, Nθ, ...)


@partial(jax.jit, static_argnames=('type', 'method', 'extrap'))
def _polar_to_cartesian(
    field, x_coordinate, y_coordinate, rho_coordinate, theta_coordinate,
    type, method, extrap,
):
    Nx, Ny       = x_coordinate.size, y_coordinate.size
    Nrho, Ntheta = rho_coordinate.size, theta_coordinate.size
    # Query points on the polar source grid, one per (x, y) target cell:
    #   ρ_q  = √(x² + y²),   θ_q = atan2(y, x)  (wrapped into theta_coordinate's range)
    xg, yg = jnp.meshgrid(x_coordinate, y_coordinate, indexing='ij')
    rho_q_flat   = jnp.sqrt(xg**2 + yg**2).flatten()
    theta_q_raw  = jnp.arctan2(yg, xg).flatten()
    # Bring atan2's [-π, π) output into [θ_start, θ_start + 2π). interp2d's
    # `period` argument handles the periodic wrap on the interpolation side,
    # but pre-shifting to the coordinate's own range gives nicer diagnostics.
    theta_start  = theta_coordinate[0]
    theta_q_flat = jnp.mod(theta_q_raw - theta_start, 2 * jnp.pi) + theta_start
    # For interp2d(x=ρ, y=θ, f=...), period is (period_x, period_y) → periodic in θ only.
    period = (None, float(2 * jnp.pi))

    if type == 'scalar':
        assert field.shape[:2] == (Nrho, Ntheta), (
            f"scalar Polar->Cartesian: field.shape[:2]={field.shape[:2]} must equal (Nρ,Nθ)=({Nrho},{Ntheta})"
        )
        result_flat = interp2d(
            xq=rho_q_flat, yq=theta_q_flat,
            x=rho_coordinate, y=theta_coordinate, f=field,
            method=method, extrap=extrap, period=period,
        )
        trailing = field.shape[2:]
        return result_flat.reshape((Nx, Ny) + trailing)

    # vector
    assert field.shape[0] == 2, (
        f"vector: field.shape[0] must be 2 (F_ρ, F_θ), got {field.shape[0]}"
    )
    assert field.shape[1:3] == (Nrho, Ntheta), (
        f"vector Polar->Cartesian: field.shape[1:3]={field.shape[1:3]} must equal (Nρ,Nθ)=({Nrho},{Ntheta})"
    )
    f_moved = jnp.moveaxis(field, 0, -1)                                 # (Nρ, Nθ, ..., 2)
    result_flat = interp2d(
        xq=rho_q_flat, yq=theta_q_flat,
        x=rho_coordinate, y=theta_coordinate, f=f_moved,
        method=method, extrap=extrap, period=period,
    )                                                                    # (Nq, ..., 2)
    trailing = field.shape[3:]
    result = result_flat.reshape((Nx, Ny) + trailing + (2,))
    result = jnp.moveaxis(result, -1, 0)                                 # (2, Nx, Ny, ...)

    # Vector rotation (inverse of the C→P rotation, evaluated at θ_q(x, y)):
    #   [Fx]   [cos θ_q   -sin θ_q] [F_ρ]
    #   [Fy] = [sin θ_q    cos θ_q] [F_θ]
    # R has axes (out_component, in_component, x, y); result has axes
    # (in_component, x, y, ...).
    theta_grid = jnp.arctan2(yg, xg)                                     # (Nx, Ny)
    cos_t = jnp.cos(theta_grid)
    sin_t = jnp.sin(theta_grid)
    R = jnp.stack([
        jnp.stack([cos_t, -sin_t]),
        jnp.stack([sin_t,  cos_t]),
    ])                                                                   # (2, 2, Nx, Ny)
    return jnp.einsum('ijab,jab...->iab...', R, result)                  # (2, Nx, Ny, ...)


# ─────────────────────────────────────────────────────────────────────────────
# 3-D spherical (r, θ, φ) ↔ Cartesian (x, y, z)
# ─────────────────────────────────────────────────────────────────────────────


def square_integral_spherical(
    field: jnp.ndarray,
    r_coordinate: jnp.ndarray,
    theta_coordinate: jnp.ndarray,
    phi_coordinate: jnp.ndarray,
    r_axis: int = -3,
    theta_axis: int = -2,
    phi_axis: int = -1,
):
    """
    Compute ∭|F|² r² sin θ dr dθ dφ, summing every axis of `field`.

    Multiplies `field` pointwise by r · √(sin θ) (each broadcast along its
    own axis) so |·|² picks up the spherical Jacobian |J| = r² sin θ, then
    delegates the dr·dθ·dφ · Σ step to `square_integral_field`.

    Parameters
    ----------
    field : array_like
        Any shape, with an r axis (Nr), θ axis (Nθ), φ axis (Nφ).
    r_coordinate : (Nr,) non-negative, ascending
    theta_coordinate : (Nθ,) radians in [0, π], ascending
    phi_coordinate : (Nφ,) radians, ascending
    r_axis, theta_axis, phi_axis : int
        Which axes of `field` hold r, θ, φ. Default is the last three.
    """
    field = jnp.asarray(field)
    r     = jnp.asarray(r_coordinate).flatten()
    theta = jnp.asarray(theta_coordinate).flatten()
    phi   = jnp.asarray(phi_coordinate).flatten()
    dr     = float(r[1]     - r[0])     if r.size     > 1 else 1.0
    dtheta = float(theta[1] - theta[0]) if theta.size > 1 else 1.0
    dphi   = float(phi[1]   - phi[0])   if phi.size   > 1 else 1.0

    r_axis     = int(r_axis)     % field.ndim
    theta_axis = int(theta_axis) % field.ndim

    shape_r = [1] * field.ndim
    shape_r[r_axis] = r.size
    r_bcast = r.reshape(shape_r)

    shape_theta = [1] * field.ndim
    shape_theta[theta_axis] = theta.size
    sqrt_sin_theta_bcast = jnp.sqrt(jnp.sin(theta)).reshape(shape_theta)

    field_weighted = field * r_bcast * sqrt_sin_theta_bcast
    return square_integral_field(
        field_weighted, dr=[dr, dtheta, dphi], axis=None,
    )


def spherical_transformation(
    field: jnp.ndarray,
    x_coordinate: jnp.ndarray,
    y_coordinate: jnp.ndarray,
    z_coordinate: jnp.ndarray,
    r_coordinate: jnp.ndarray,
    theta_coordinate: jnp.ndarray,
    phi_coordinate: jnp.ndarray,
    direction: str = 'Cartesian->Spherical',
    type: str = 'scalar',
    method: str = 'cubic',
    extrap: Union[bool, float] = False,
):
    """
    Resample a field between the Cartesian (x, y, z) grid and the spherical
    (r, θ, φ) grid.

    Parameters
    ----------
    field : array_like
        - `type='scalar'` :
            (Nx, Ny, Nz, ...)     for `direction='Cartesian->Spherical'`
            (Nr, Nθ, Nφ, ...)     for `direction='Spherical->Cartesian'`
        - `type='vector'` :
            (3, Nx, Ny, Nz, ...)  for `direction='Cartesian->Spherical'` — (Fx, Fy, Fz)
            (3, Nr, Nθ, Nφ, ...)  for `direction='Spherical->Cartesian'` — (F_r, F_θ, F_φ)
    x, y, z_coordinate : 1-D ascending
    r_coordinate       : 1-D non-negative ascending
    theta_coordinate   : 1-D in [0, π] ascending (polar angle from +z)
    phi_coordinate     : 1-D radians ascending (treated as 2π-periodic on the
                         spherical-source side)
    direction : {'Cartesian->Spherical', 'Spherical->Cartesian'}
    type      : {'scalar', 'vector'}
    method    : interpax.interp3d method ('nearest', 'linear', 'cubic', ...)
    extrap    : passed to interpax.interp3d; `False` returns NaN outside knots.

    Returns
    -------
    array on the target grid, layout mirroring `field` conventions above.
    """
    assert direction in ('Cartesian->Spherical', 'Spherical->Cartesian'), (
        f"direction must be 'Cartesian->Spherical' or 'Spherical->Cartesian', got {direction!r}"
    )
    assert type in ('scalar', 'vector'), (
        f"type must be 'scalar' or 'vector', got {type!r}"
    )

    field            = jnp.asarray(field)
    x_coordinate     = jnp.asarray(x_coordinate).flatten()
    y_coordinate     = jnp.asarray(y_coordinate).flatten()
    z_coordinate     = jnp.asarray(z_coordinate).flatten()
    r_coordinate     = jnp.asarray(r_coordinate).flatten()
    theta_coordinate = jnp.asarray(theta_coordinate).flatten()
    phi_coordinate   = jnp.asarray(phi_coordinate).flatten()

    assert bool(jnp.all(r_coordinate >= 0)), "r_coordinate must be non-negative."

    Nx, Ny, Nz              = x_coordinate.size, y_coordinate.size, z_coordinate.size
    Nr, Ntheta, Nphi        = r_coordinate.size, theta_coordinate.size, phi_coordinate.size

    if direction == 'Cartesian->Spherical':
        result = _cartesian_to_spherical(
            field, x_coordinate, y_coordinate, z_coordinate,
            r_coordinate, theta_coordinate, phi_coordinate,
            type, method, extrap,
        )
    else:
        result = _spherical_to_cartesian(
            field, x_coordinate, y_coordinate, z_coordinate,
            r_coordinate, theta_coordinate, phi_coordinate,
            type, method, extrap,
        )

    # Ratio check: integrate the input and output over their native grids.
    dx = float(x_coordinate[1] - x_coordinate[0]) if x_coordinate.size > 1 else 1.0
    dy = float(y_coordinate[1] - y_coordinate[0]) if y_coordinate.size > 1 else 1.0
    dz = float(z_coordinate[1] - z_coordinate[0]) if z_coordinate.size > 1 else 1.0
    r_axis, theta_axis, phi_axis = (0, 1, 2) if type == 'scalar' else (1, 2, 3)
    if direction == 'Cartesian->Spherical':
        I0 = square_integral_field(field, dr=[dx, dy, dz], axis=None)
        I1 = square_integral_spherical(
            result, r_coordinate, theta_coordinate, phi_coordinate,
            r_axis=r_axis, theta_axis=theta_axis, phi_axis=phi_axis,
        )
    else:
        I0 = square_integral_spherical(
            field, r_coordinate, theta_coordinate, phi_coordinate,
            r_axis=r_axis, theta_axis=theta_axis, phi_axis=phi_axis,
        )
        I1 = square_integral_field(result, dr=[dx, dy, dz], axis=None)
    jax.debug.print('spherical_transformation Integral ratio I1/I0: {ratio}', ratio=I1 / I0)
    return result


@partial(jax.jit, static_argnames=('type', 'method', 'extrap'))
def _cartesian_to_spherical(
    field, x_coordinate, y_coordinate, z_coordinate,
    r_coordinate, theta_coordinate, phi_coordinate,
    type, method, extrap,
):
    Nx, Ny, Nz       = x_coordinate.size, y_coordinate.size, z_coordinate.size
    Nr, Ntheta, Nphi = r_coordinate.size, theta_coordinate.size, phi_coordinate.size
    # Query points on the Cartesian source grid, one per (r, θ, φ) target cell:
    #   xq = r sinθ cosφ,   yq = r sinθ sinφ,   zq = r cosθ
    r_grid, th_grid, ph_grid = jnp.meshgrid(
        r_coordinate, theta_coordinate, phi_coordinate, indexing='ij',
    )
    sin_th = jnp.sin(th_grid)
    xq = (r_grid * sin_th * jnp.cos(ph_grid)).flatten()
    yq = (r_grid * sin_th * jnp.sin(ph_grid)).flatten()
    zq = (r_grid * jnp.cos(th_grid)).flatten()

    if type == 'scalar':
        assert field.shape[:3] == (Nx, Ny, Nz), (
            f"scalar Cartesian->Spherical: field.shape[:3]={field.shape[:3]} "
            f"must equal (Nx, Ny, Nz)=({Nx}, {Ny}, {Nz})"
        )
        result_flat = interp3d(
            xq=xq, yq=yq, zq=zq,
            x=x_coordinate, y=y_coordinate, z=z_coordinate, f=field,
            method=method, extrap=extrap,
        )
        trailing = field.shape[3:]
        return result_flat.reshape((Nr, Ntheta, Nphi) + trailing)

    # vector
    assert field.shape[0] == 3, (
        f"vector: field.shape[0] must be 3 (Fx, Fy, Fz), got {field.shape[0]}"
    )
    assert field.shape[1:4] == (Nx, Ny, Nz), (
        f"vector Cartesian->Spherical: field.shape[1:4]={field.shape[1:4]} "
        f"must equal (Nx, Ny, Nz)=({Nx}, {Ny}, {Nz})"
    )
    f_moved = jnp.moveaxis(field, 0, -1)                                 # (Nx, Ny, Nz, ..., 3)
    result_flat = interp3d(
        xq=xq, yq=yq, zq=zq,
        x=x_coordinate, y=y_coordinate, z=z_coordinate, f=f_moved,
        method=method, extrap=extrap,
    )                                                                    # (Nq, ..., 3)
    trailing = field.shape[4:]
    result = result_flat.reshape((Nr, Ntheta, Nphi) + trailing + (3,))
    result = jnp.moveaxis(result, -1, 0)                                 # (3, Nr, Nθ, Nφ, ...)

    # Vector rotation (Cartesian → Spherical), evaluated at target (θ, φ):
    #   [F_r]     [ sinθ cosφ,  sinθ sinφ,   cosθ] [Fx]
    #   [F_θ]  =  [ cosθ cosφ,  cosθ sinφ,  -sinθ] [Fy]
    #   [F_φ]     [-sinφ,       cosφ,        0   ] [Fz]
    # sin/cos of θ and φ only depend on (θ, φ) — cache them at the target grid.
    tg, pg   = jnp.meshgrid(theta_coordinate, phi_coordinate, indexing='ij')
    sin_th_g = jnp.sin(tg);  cos_th_g = jnp.cos(tg)
    sin_ph_g = jnp.sin(pg);  cos_ph_g = jnp.cos(pg)
    zero     = jnp.zeros_like(sin_th_g)
    R = jnp.stack([
        jnp.stack([ sin_th_g * cos_ph_g,  sin_th_g * sin_ph_g,   cos_th_g]),
        jnp.stack([ cos_th_g * cos_ph_g,  cos_th_g * sin_ph_g,  -sin_th_g]),
        jnp.stack([-sin_ph_g,             cos_ph_g,              zero    ]),
    ])                                                                   # (3, 3, Nθ, Nφ)
    # result: (3, Nr, Nθ, Nφ, ...);  einsum contracts j (input component),
    # broadcasts (θ, φ) via matching labels s, t; keeps r free (only in result).
    return jnp.einsum('ijst,jrst...->irst...', R, result)                # (3, Nr, Nθ, Nφ, ...)


@partial(jax.jit, static_argnames=('type', 'method', 'extrap'))
def _spherical_to_cartesian(
    field, x_coordinate, y_coordinate, z_coordinate,
    r_coordinate, theta_coordinate, phi_coordinate,
    type, method, extrap,
):
    Nx, Ny, Nz       = x_coordinate.size, y_coordinate.size, z_coordinate.size
    Nr, Ntheta, Nphi = r_coordinate.size, theta_coordinate.size, phi_coordinate.size
    # Query points on the spherical source grid, one per (x, y, z) target cell:
    #   r_q = √(x²+y²+z²),   θ_q = atan2(√(x²+y²), z) ∈ [0, π],
    #   φ_q = atan2(y, x)   ∈ [-π, π],  wrapped into phi_coordinate's range.
    xg, yg, zg = jnp.meshgrid(x_coordinate, y_coordinate, z_coordinate, indexing='ij')
    r_q_flat  = jnp.sqrt(xg**2 + yg**2 + zg**2).flatten()
    th_q_flat = jnp.arctan2(jnp.sqrt(xg**2 + yg**2), zg).flatten()
    ph_q_raw  = jnp.arctan2(yg, xg).flatten()
    phi_start = phi_coordinate[0]
    ph_q_flat = jnp.mod(ph_q_raw - phi_start, 2 * jnp.pi) + phi_start
    # interp3d(x=r, y=θ, z=φ) → periodic only in φ.
    period = (None, None, float(2 * jnp.pi))

    if type == 'scalar':
        assert field.shape[:3] == (Nr, Ntheta, Nphi), (
            f"scalar Spherical->Cartesian: field.shape[:3]={field.shape[:3]} "
            f"must equal (Nr, Nθ, Nφ)=({Nr}, {Ntheta}, {Nphi})"
        )
        result_flat = interp3d(
            xq=r_q_flat, yq=th_q_flat, zq=ph_q_flat,
            x=r_coordinate, y=theta_coordinate, z=phi_coordinate, f=field,
            method=method, extrap=extrap, period=period,
        )
        trailing = field.shape[3:]
        return result_flat.reshape((Nx, Ny, Nz) + trailing)

    # vector
    assert field.shape[0] == 3, (
        f"vector: field.shape[0] must be 3 (F_r, F_θ, F_φ), got {field.shape[0]}"
    )
    assert field.shape[1:4] == (Nr, Ntheta, Nphi), (
        f"vector Spherical->Cartesian: field.shape[1:4]={field.shape[1:4]} "
        f"must equal (Nr, Nθ, Nφ)=({Nr}, {Ntheta}, {Nphi})"
    )
    f_moved = jnp.moveaxis(field, 0, -1)                                 # (Nr, Nθ, Nφ, ..., 3)
    result_flat = interp3d(
        xq=r_q_flat, yq=th_q_flat, zq=ph_q_flat,
        x=r_coordinate, y=theta_coordinate, z=phi_coordinate, f=f_moved,
        method=method, extrap=extrap, period=period,
    )                                                                    # (Nq, ..., 3)
    trailing = field.shape[4:]
    result = result_flat.reshape((Nx, Ny, Nz) + trailing + (3,))
    result = jnp.moveaxis(result, -1, 0)                                 # (3, Nx, Ny, Nz, ...)

    # Vector rotation (Spherical → Cartesian), evaluated at query (θ_q, φ_q):
    #   [Fx]   [sinθ cosφ,  cosθ cosφ,  -sinφ] [F_r]
    #   [Fy] = [sinθ sinφ,  cosθ sinφ,   cosφ] [F_θ]
    #   [Fz]   [cosθ,       -sinθ,        0  ] [F_φ]
    # θ_q depends on (x, y, z); φ_q depends on (x, y). Build R on the full
    # (Nx, Ny, Nz) grid so the einsum stays clean.
    th_xyz = jnp.arctan2(jnp.sqrt(xg**2 + yg**2), zg)                    # (Nx, Ny, Nz)
    ph_xyz = jnp.arctan2(yg, xg)
    sin_th = jnp.sin(th_xyz);  cos_th = jnp.cos(th_xyz)
    sin_ph = jnp.sin(ph_xyz);  cos_ph = jnp.cos(ph_xyz)
    zero   = jnp.zeros_like(sin_th)
    R = jnp.stack([
        jnp.stack([sin_th * cos_ph,  cos_th * cos_ph,  -sin_ph]),
        jnp.stack([sin_th * sin_ph,  cos_th * sin_ph,   cos_ph]),
        jnp.stack([cos_th,          -sin_th,            zero  ]),
    ])                                                                   # (3, 3, Nx, Ny, Nz)
    # result: (3, Nx, Ny, Nz, ...); contract j (input component); keep (x, y, z)
    # via matching labels; keep trailing axes via ellipsis.
    return jnp.einsum('ijabc,jabc...->iabc...', R, result)               # (3, Nx, Ny, Nz, ...)
