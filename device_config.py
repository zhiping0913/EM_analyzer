"""
Single source of truth for the EM_analyzer JAX backend configuration.

Different modules in the codebase have historically set their own JAX
platform / cpu-device-count / sharding requirements at import time. That
created a race — whichever module was imported first "won", and the losers
either silently mis-configured or asserted their way out. For example:

    - Spectral_Maxwell/Normal_variable_method.py wants 2 GPUs or 6 CPUs
      (2 × 3 mesh: EM axis × channel axis).
    - rotate_3D.py wants 3 CPUs (channel axis).
    - Gaussian_beam_2D.py imports both.

`configure_jax_backend()` picks a configuration that satisfies every
downstream sharding requirement (6 CPU devices, or ≥2 GPUs). It is
idempotent — the first call performs the actual `jax.config.update` calls
and caches the result; subsequent calls just return the cached info.

Rules
-----
1. If the caller pinned CPU before we ran (via `JAX_PLATFORMS=cpu` env var
   or a prior `jax.config.update('jax_platform_name', 'cpu')`), stay on
   CPU.
2. Otherwise: if `nvidia-smi` reports ≥ `n_gpu_min` GPUs (default 2), use
   GPU.
3. On CPU, request `n_cpu_default` devices (default 6). Skip that
   `jax.config.update` if `JAX_NUM_CPU_DEVICES` is already set — multi-
   process slurm jobs use a per-process count.

Usage in a downstream module
----------------------------
    from EM_analyzer.device_config import configure_jax_backend
    info = configure_jax_backend()
    USE_GPU = info['USE_GPU']
    # …now safe to `import jax.numpy as jnp` and query `jax.devices()`.
"""

from __future__ import annotations
import os
import subprocess
from typing import Optional


# Cached after the first successful call. See configure_jax_backend().
USE_GPU:             Optional[bool] = None
BACKEND:             Optional[str]  = None
LOCAL_DEVICE_COUNT:  Optional[int]  = None
GLOBAL_DEVICE_COUNT: Optional[int]  = None

# Sensible defaults. Callers may override via configure_jax_backend(…).
DEFAULT_N_GPU_MIN     = 2
DEFAULT_N_CPU_DEVICES = 6


def _detect_gpu_count() -> int:
    """Count NVIDIA GPUs via `nvidia-smi` without initializing JAX."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return sum(1 for line in result.stdout.strip().split('\n') if line.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return 0


def configure_jax_backend(
    n_gpu_min: int = DEFAULT_N_GPU_MIN,
    n_cpu_devices: int = DEFAULT_N_CPU_DEVICES,
    verbose: bool = True,
) -> dict:
    """Idempotently configure the JAX backend.

    Parameters
    ----------
    n_gpu_min : int
        Minimum GPU count needed to switch to GPU mode. Below this we stay
        on CPU. Default 2 (Normal_variable_method's 2-GPU sharding path).
    n_cpu_devices : int
        Requested CPU device count when we are the ones setting it.
        Default 6 (covers Normal_variable_method's 2×3 mesh and leaves
        room for rotate_3D's 3-channel mesh on the same devices).
    verbose : bool
        Print a one-line summary of the chosen backend on the first call.

    Returns
    -------
    dict with keys USE_GPU, BACKEND, LOCAL_DEVICE_COUNT, GLOBAL_DEVICE_COUNT.
    """
    global USE_GPU, BACKEND, LOCAL_DEVICE_COUNT, GLOBAL_DEVICE_COUNT

    import jax   # local import — first call happens before the caller uses jax

    if USE_GPU is not None:
        # Already configured; refresh live counts and return.
        LOCAL_DEVICE_COUNT  = jax.local_device_count()
        GLOBAL_DEVICE_COUNT = jax.device_count()
        return {
            'USE_GPU':             USE_GPU,
            'BACKEND':             BACKEND,
            'LOCAL_DEVICE_COUNT':  LOCAL_DEVICE_COUNT,
            'GLOBAL_DEVICE_COUNT': GLOBAL_DEVICE_COUNT,
        }

    # 1. Honor a prior CPU pin.
    user_forced_cpu = (
        os.environ.get('JAX_PLATFORMS', '').lower() == 'cpu'
        or jax.config.values.get('jax_platform_name') == 'cpu'
    )

    # 2. GPU probe.
    n_gpus = 0 if user_forced_cpu else _detect_gpu_count()
    use_gpu = n_gpus >= n_gpu_min

    # 3. Config updates. Each is a best-effort — if JAX has already enumerated
    #    devices, further updates are silently ignored, and we log whatever
    #    JAX actually gave us.
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass
    if use_gpu:
        try:
            jax.config.update('jax_platform_name', 'gpu')
        except Exception:
            pass
    else:
        if not os.environ.get('JAX_NUM_CPU_DEVICES'):
            try:
                jax.config.update('jax_num_cpu_devices', n_cpu_devices)
            except Exception:
                pass
        try:
            jax.config.update('jax_platform_name', 'cpu')
        except Exception:
            pass

    USE_GPU             = use_gpu
    BACKEND             = 'GPU' if USE_GPU else 'CPU'
    LOCAL_DEVICE_COUNT  = jax.local_device_count()
    GLOBAL_DEVICE_COUNT = jax.device_count()

    if verbose:
        print(
            f"[device_config] Backend: {BACKEND}, "
            f"local devices: {LOCAL_DEVICE_COUNT}, "
            f"global devices: {GLOBAL_DEVICE_COUNT}",
            flush=True,
        )
        print(f"[device_config] devices: {jax.devices()}", flush=True)

    return {
        'USE_GPU':             USE_GPU,
        'BACKEND':             BACKEND,
        'LOCAL_DEVICE_COUNT':  LOCAL_DEVICE_COUNT,
        'GLOBAL_DEVICE_COUNT': GLOBAL_DEVICE_COUNT,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sharding helpers
#
# The "channel" pattern shards a (n_components, ...) array across n_components
# local devices. This is the same convention used by rotate_3D (3-component
# vector fields) and by the vector-case fast paths of
# coordinate_transformation (2- or 3-component vector fields).
#
# When we don't have enough local devices for a clean split (e.g. 3-way
# sharding on 2 GPUs), we fall back to a single-device replicated mesh —
# the code still runs correctly, just without the parallel speed-up.
# ─────────────────────────────────────────────────────────────────────────────

_channel_sharding_cache: dict = {}


def get_channel_sharding(n_components: int):
    """
    Return a `NamedSharding` that shards axis 0 of a `(n_components, …)`
    array across `n_components` local devices.

    Falls back to a 1-device, fully replicated mesh if there aren't enough
    local devices. Cached per `n_components`.
    """
    if n_components in _channel_sharding_cache:
        return _channel_sharding_cache[n_components]

    if USE_GPU is None:
        configure_jax_backend()

    import jax
    from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
    import numpy as np

    n_local = jax.local_device_count()
    if n_local >= n_components:
        devs  = jax.local_devices()[:n_components]
        mesh  = Mesh(np.array(devs), ('channel',))
        shard = NamedSharding(mesh, P('channel'))
    else:
        devs  = jax.local_devices()[:1]
        mesh  = Mesh(np.array(devs), ('channel',))
        shard = NamedSharding(mesh, P())           # fully replicated

    _channel_sharding_cache[n_components] = shard
    return shard


def get_replicated_on_channel_mesh(n_components: int):
    """
    Return a `NamedSharding` that replicates over the *same* mesh as
    `get_channel_sharding(n_components)`. Use this for small 1-D coordinate
    arrays fed alongside a sharded (n_components, …) field — the coordinates
    need to broadcast to every device that holds a shard of the field.
    """
    from jax.sharding import PartitionSpec as P, NamedSharding
    ch = get_channel_sharding(n_components)
    return NamedSharding(ch.mesh, P())
