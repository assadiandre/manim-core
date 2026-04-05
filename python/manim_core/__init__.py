"""
manim_core: Rust-accelerated hot paths for Manim.

Import this module and call activate() to monkey-patch manim with Rust-accelerated
versions of the 6 hot paths:
  - 3D projection (project_all_points)
  - Per-face shading (shade_all_objects)
  - Z-sort + tree walks (z_sort, get_family_order)
  - Scene hashing (hash_pool_state)
  - Deep copy + interpolation (clone_pool, interpolate_pools)
  - Render data preparation (prepare_render_data, compute_visibility)

Usage:
    import manim_core
    manim_core.activate()  # patches manim — call before scene.render()
"""

from manim_core._rust import (
    MeshPool,
    project_all_points,
    project_points_for_objects,
    shade_all_objects,
    recompute_family_order,
    get_family_order,
    z_sort,
    get_family_for,
    hash_pool_state,
    hash_objects,
    clone_pool,
    interpolate_pools,
    interpolate_objects,
    interpolate_object_attrs,
    prepare_render_data,
    compute_visibility,
)

from manim_core.pool_manager import PoolManager, get_scene_pool, set_scene_pool, get_pool_manager

_activated = False


def activate():
    """Apply all monkey-patches. Safe to call multiple times."""
    global _activated
    if _activated:
        return
    from manim_core.patches import apply_all_patches
    apply_all_patches()
    _activated = True


__all__ = [
    "MeshPool",
    "PoolManager",
    "activate",
    "get_scene_pool",
    "set_scene_pool",
    "get_pool_manager",
    "project_all_points",
    "project_points_for_objects",
    "shade_all_objects",
    "recompute_family_order",
    "get_family_order",
    "z_sort",
    "get_family_for",
    "hash_pool_state",
    "hash_objects",
    "clone_pool",
    "interpolate_pools",
    "interpolate_objects",
    "interpolate_object_attrs",
    "prepare_render_data",
    "compute_visibility",
]
