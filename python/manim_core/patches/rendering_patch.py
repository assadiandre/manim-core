"""
Monkey-patch for rendering data preparation.

Patches:
  - Camera.display_multiple_non_background_colored_vmobjects → uses Rust batch
    visibility checks and pre-computed subpath data
"""
import numpy as np

from manim_core._rust import prepare_render_data, compute_visibility
from manim_core.pool_manager import get_scene_pool, get_pool_manager


def patch_rendering():
    from manim.camera.camera import Camera

    _orig_display_multi = Camera.display_multiple_non_background_colored_vmobjects

    def _patched_display_multi(self, vmobjects, pixel_array):
        pool = get_scene_pool()
        pm = get_pool_manager()
        vmobjects = list(vmobjects)

        if pool is None or pm is None or not vmobjects:
            return _orig_display_multi(self, vmobjects, pixel_array)

        try:
            pool_ids = []
            pooled_mask = []
            for vm in vmobjects:
                if hasattr(vm, '_pool_id'):
                    pool_ids.append(vm._pool_id)
                    pooled_mask.append(True)
                else:
                    pooled_mask.append(False)

            if not pool_ids:
                return _orig_display_multi(self, vmobjects, pixel_array)

            order = np.array(pool_ids, dtype=np.uint32)

            # Batch visibility check
            has_fill, has_stroke, has_bg_stroke = compute_visibility(pool, order)

            # Batch subpath extraction
            render_data = prepare_render_data(pool, order)
            subpath_starts_list = render_data['subpath_starts']
            subpath_ends_list = render_data['subpath_ends']

            # Render each object, skipping invisible ones
            pool_idx = 0
            for i, vm in enumerate(vmobjects):
                if pooled_mask[i]:
                    if not has_fill[pool_idx] and not has_stroke[pool_idx] and not has_bg_stroke[pool_idx]:
                        pool_idx += 1
                        continue
                    # Attach pre-computed subpath indices for set_cairo_context_path
                    vm._precomputed_subpaths = (
                        subpath_starts_list[pool_idx],
                        subpath_ends_list[pool_idx],
                    )
                    pool_idx += 1
                self.display_vectorized(vm, self.get_skia_canvas(pixel_array))
                if hasattr(vm, '_precomputed_subpaths'):
                    del vm._precomputed_subpaths
        except Exception:
            return _orig_display_multi(self, vmobjects, pixel_array)

    Camera.display_multiple_non_background_colored_vmobjects = _patched_display_multi
