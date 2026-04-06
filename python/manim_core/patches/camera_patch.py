"""
Monkey-patches for ThreeDCamera.

Key optimizations:
  - transform_points_pre_display: uses batch-projected points from pool
  - get_mobjects_to_display: calls Camera base (skips ThreeDCamera Python z-sort),
    then applies Rust z_sort
"""
import numpy as np

from manim_core._rust import (
    project_all_points,
    shade_all_objects,
    z_sort,
)
from manim_core.pool_manager import get_scene_pool, get_pool_manager


def patch_three_d_camera():
    from manim.camera.three_d_camera import ThreeDCamera
    from manim.camera.camera import Camera as _CameraBase

    _orig_get_mobjects_to_display = ThreeDCamera.get_mobjects_to_display
    _orig_transform = ThreeDCamera.transform_points_pre_display
    # Grandparent method: does family extraction + z_index sort, but NOT 3D z-sort
    _camera_base_get_mobjects = _CameraBase.get_mobjects_to_display

    def _patched_get_mobjects_to_display(self, *args, **kwargs):
        """Skip ThreeDCamera's Python z-sort, use Rust z_sort instead."""
        pool = get_scene_pool()
        pm = get_pool_manager()

        if pool is None or pm is None or pool.len() == 0:
            return _orig_get_mobjects_to_display(self, *args, **kwargs)

        try:
            # Ensure any new mobjects from args are registered
            if args:
                for mob in args[0]:
                    pm.ensure_registered(mob)

            # Call Camera base directly — skips ThreeDCamera's expensive Python
            # sorted() + z_key which calls get_center() on every mobject
            base_result = _camera_base_get_mobjects(self, *args, **kwargs)

            rot_matrix = self._frame_rot_matrix

            # Separate pool-backed 3D objects and others
            pool_3d = []
            for mob in base_result:
                if hasattr(mob, '_pool_id') and getattr(mob, 'shade_in_3d', False):
                    pool_3d.append(mob)

            if not pool_3d:
                return base_result

            # Z-sort pool-backed objects using Rust
            sorted_ids = z_sort(pool, rot_matrix)
            id_to_pos = {int(sid): pos for pos, sid in enumerate(sorted_ids)}

            pool_3d.sort(key=lambda m: id_to_pos.get(m._pool_id, float('inf')))

            # Reconstruct: replace 3D objects in their original slots with sorted order
            result = []
            pool_iter = iter(pool_3d)
            for mob in base_result:
                if hasattr(mob, '_pool_id') and getattr(mob, 'shade_in_3d', False):
                    result.append(next(pool_iter))
                else:
                    result.append(mob)
            return result
        except Exception:
            return _orig_get_mobjects_to_display(self, *args, **kwargs)

    def _patched_transform_points_pre_display(self, mobject, points):
        """Use batch-projected points from pool cache if available."""
        pool = get_scene_pool()
        pm = get_pool_manager()

        if (pool is not None and pm is not None
                and hasattr(mobject, '_pool_id')
                and mobject not in getattr(self, 'fixed_in_frame_mobjects', {})
                and mobject not in getattr(self, 'fixed_orientation_mobjects', {})):

            cache = getattr(self, '_rust_proj_cache', None)
            if cache is not None:
                pid = mobject._pool_id
                try:
                    start, end = pm.pool.point_range(pid)
                    start, end = int(start), int(end)
                    npts = end - start
                    if npts > 0 and end <= len(cache) and npts == len(points):
                        return cache[start:end]
                except Exception:
                    pass

        return _orig_transform(self, mobject, points)

    def _batch_precompute(self):
        """Pre-compute projections for all pool objects this frame."""
        pool = get_scene_pool()
        if pool is None or pool.total_points() == 0:
            self._rust_proj_cache = None
            return
        try:
            fc = np.ascontiguousarray(self.frame_center, dtype=np.float64)
            rot = getattr(self, '_frame_rot_matrix', None)
            if rot is None:
                rot = np.ascontiguousarray(self.get_rotation_matrix(), dtype=np.float64)
            fd = float(self.get_focal_distance())
            zoom = float(self.get_zoom())
            exp_proj = bool(getattr(self, 'exponential_projection', True))
            self._rust_proj_cache = np.array(project_all_points(pool, fc, rot, fd, zoom, exp_proj))
        except Exception:
            self._rust_proj_cache = None

    def _batch_shade(self):
        """Pre-compute shading for all pool objects this frame."""
        pool = get_scene_pool()
        if pool is None or pool.len() == 0:
            self._rust_shaded_fills = None
            self._rust_shaded_strokes = None
            self._rust_shaded_fill_offsets = None
            self._rust_shaded_stroke_offsets = None
            return
        try:
            light_pos = np.ascontiguousarray(
                self.light_source.points[0], dtype=np.float64
            )
            fills, strokes, fill_offsets, stroke_offsets = shade_all_objects(pool, light_pos)
            self._rust_shaded_fills = np.array(fills)
            self._rust_shaded_strokes = np.array(strokes)
            self._rust_shaded_fill_offsets = np.array(fill_offsets)
            self._rust_shaded_stroke_offsets = np.array(stroke_offsets)
        except Exception:
            self._rust_shaded_fills = None
            self._rust_shaded_strokes = None
            self._rust_shaded_fill_offsets = None
            self._rust_shaded_stroke_offsets = None

    # Patch capture_mobjects to trigger batch precompute before each frame
    from manim.camera.camera import Camera
    _orig_capture = Camera.capture_mobjects

    def _patched_capture(self, mobjects, **kwargs):
        if not getattr(self, '_in_capture', False):
            self._in_capture = True
            try:
                # Always sync dirty mobjects to pool before rendering
                pm = get_pool_manager()
                if pm is not None:
                    pm.sync_all()

                # 3D-specific: projection and shading precomputation
                if isinstance(self, ThreeDCamera):
                    self._frame_rot_matrix = np.ascontiguousarray(
                        self.get_rotation_matrix(), dtype=np.float64
                    )
                    _batch_precompute(self)
                    _batch_shade(self)
            finally:
                self._in_capture = False
        return _orig_capture(self, mobjects, **kwargs)

    ThreeDCamera.get_mobjects_to_display = _patched_get_mobjects_to_display
    ThreeDCamera.transform_points_pre_display = _patched_transform_points_pre_display
    ThreeDCamera._batch_precompute = _batch_precompute
    ThreeDCamera._batch_shade = _batch_shade
    Camera.capture_mobjects = _patched_capture
