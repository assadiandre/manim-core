"""
Monkey-patch for rendering data preparation.

Patches:
  - Camera.display_multiple_non_background_colored_vmobjects -> uses Rust batch
    render via tiny-skia for pooled objects, with Python fallback for non-pooled,
    fixed-in-frame, fixed-orientation, and animation-intermediate mobjects.

Maintains correct z-order by flushing the Rust batch whenever a fallback object
is encountered, so pooled and non-pooled objects stay interleaved properly.
"""
import numpy as np

from manim_core._rust import compute_visibility, batch_render
from manim_core.pool_manager import get_scene_pool, get_pool_manager


def patch_rendering():
    from manim.camera.camera import Camera

    _orig_display_multi = Camera.display_multiple_non_background_colored_vmobjects

    def _flush_batch(self, pixel_array, pool, batch_ids):
        """Render a batch of pooled object IDs via Rust/tiny-skia."""
        if not batch_ids:
            return
        order = np.array(batch_ids, dtype=np.uint32)
        has_fill, has_stroke, has_bg_stroke = compute_visibility(pool, order)

        pw = self.pixel_width
        ph = self.pixel_height
        fw = self.frame_width
        fh = self.frame_height
        fc = self.frame_center

        tx = float(pw / 2 - fc[0] * pw / fw)
        ty = float(ph / 2 + fc[1] * ph / fh)
        sx = float(pw / fw)
        sy = float(-(ph / fh))

        batch_render(
            pixel_array,
            pw, ph,
            tx, ty, sx, sy,
            pool,
            order,
            has_fill,
            has_stroke,
            has_bg_stroke,
            getattr(self, '_rust_proj_cache', None),
            getattr(self, '_rust_shaded_fills', None),
            getattr(self, '_rust_shaded_strokes', None),
            getattr(self, '_rust_shaded_fill_offsets', None),
            getattr(self, '_rust_shaded_stroke_offsets', None),
            self.cairo_line_width_multiple,
        )

    def _is_batchable(vm, pm, pool, fixed_in_frame, fixed_orientation):
        """Check if a VMobject can be rendered via the Rust batch path.

        Returns False for:
        - Objects without a pool_id
        - Fixed-in-frame / fixed-orientation objects (need special transforms)
        - Animation intermediates (deepcopy clones whose id() isn't in the pool
          manager, meaning their points/colors may differ from the pool)
        - Objects whose point count differs from the pool (animations like
          Create use pointwise_become_partial which changes point count;
          sync_all silently fails for these, leaving stale data in the pool)
        """
        if not hasattr(vm, '_pool_id'):
            return False
        if vm in fixed_in_frame or vm in fixed_orientation:
            return False
        if id(vm) not in pm._obj_to_id:
            return False
        # Point count mismatch → pool has stale geometry (sync failed)
        pid = vm._pool_id
        pts = vm.points
        if pts is not None:
            start, end = pool.point_range(pid)
            if len(pts) != (end - start):
                return False
        return True

    def _patched_display_multi(self, vmobjects, pixel_array):
        pool = get_scene_pool()
        pm = get_pool_manager()
        vmobjects = list(vmobjects)

        if pool is None or pm is None or not vmobjects:
            return _orig_display_multi(self, vmobjects, pixel_array)

        try:
            fixed_in_frame = getattr(self, 'fixed_in_frame_mobjects', set())
            fixed_orientation = getattr(self, 'fixed_orientation_mobjects', {})

            # Walk objects in order, batching consecutive pool-eligible objects
            # and flushing when a fallback object is encountered.  This
            # preserves the painter's-algorithm z-order.
            current_batch = []
            ctx = None  # lazy-init skia canvas only if needed

            for vm in vmobjects:
                if _is_batchable(vm, pm, pool, fixed_in_frame, fixed_orientation):
                    current_batch.append(vm._pool_id)
                else:
                    # Flush any pending batch before drawing the fallback object
                    if current_batch:
                        _flush_batch(self, pixel_array, pool, current_batch)
                        current_batch = []
                    # Render fallback object via Python/skia-python
                    if ctx is None:
                        ctx = self.get_skia_canvas(pixel_array)
                    self.display_vectorized(vm, ctx)

            # Flush remaining batch
            if current_batch:
                _flush_batch(self, pixel_array, pool, current_batch)

        except Exception:
            return _orig_display_multi(self, vmobjects, pixel_array)

    Camera.display_multiple_non_background_colored_vmobjects = _patched_display_multi
