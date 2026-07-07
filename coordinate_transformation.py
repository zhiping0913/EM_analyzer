"""
Coordinate transformations between Cartesian (x, y) and 2-D polar (ρ, θ).

`polar_transformation` interpolates a field between the two grids with
`interpax.interp2d` (cubic by default), and — for `type='vector'` — additionally
rotates the (Fx, Fy) ↔ (F_ρ, F_θ) components at every sample point.

The polar convention:
    x = ρ cos θ,     y = ρ sin θ
    F_ρ =  Fx cos θ + Fy sin θ
    F_θ = -Fx sin θ + Fy cos θ
    ρ ≥ 0. θ is in radians and is treated as 2π-periodic during Polar→Cartesian
    interpolation, so it is fine to have θ span [0, 2π), [-π, π), etc.
"""

from typing import Union
import jax.numpy as jnp
from interpax import interp2d


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
        return _cartesian_to_polar(
            field, x_coordinate, y_coordinate, rho_coordinate, theta_coordinate,
            Nx, Ny, Nrho, Ntheta, type, method, extrap,
        )
    else:
        return _polar_to_cartesian(
            field, x_coordinate, y_coordinate, rho_coordinate, theta_coordinate,
            Nx, Ny, Nrho, Ntheta, type, method, extrap,
        )


def _cartesian_to_polar(
    field, x_coordinate, y_coordinate, rho_coordinate, theta_coordinate,
    Nx, Ny, Nrho, Ntheta, type, method, extrap,
):
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


def _polar_to_cartesian(
    field, x_coordinate, y_coordinate, rho_coordinate, theta_coordinate,
    Nx, Ny, Nrho, Ntheta, type, method, extrap,
):
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
