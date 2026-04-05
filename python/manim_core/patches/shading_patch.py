"""
Monkey-patch for ThreeDCamera shading.

Patches:
  - ThreeDCamera.modified_rgbas → uses batch-precomputed shaded colors from Rust
"""
import numpy as np

from manim_core.pool_manager import get_scene_pool, get_pool_manager


def patch_shading():
    from manim.camera.three_d_camera import ThreeDCamera

    _orig_modified_rgbas = ThreeDCamera.modified_rgbas

    def _patched_modified_rgbas(self, vmobject, rgbas):
        pool = get_scene_pool()
        pm = get_pool_manager()

        if (pool is not None and pm is not None
                and hasattr(vmobject, '_pool_id')
                and getattr(vmobject, 'shade_in_3d', False)
                and getattr(self, 'should_apply_shading', True)):

            shaded_fills = getattr(self, '_rust_shaded_fills', None)
            shaded_strokes = getattr(self, '_rust_shaded_strokes', None)
            fill_offsets = getattr(self, '_rust_shaded_fill_offsets', None)
            stroke_offsets = getattr(self, '_rust_shaded_stroke_offsets', None)

            if (shaded_fills is not None and shaded_strokes is not None
                    and fill_offsets is not None and stroke_offsets is not None):
                pid = vmobject._pool_id
                try:
                    if pid >= len(fill_offsets) - 1:
                        return _orig_modified_rgbas(self, vmobject, rgbas)

                    # Determine if this is fill or stroke
                    fill_rgbas = getattr(vmobject, 'fill_rgbas', None)
                    is_fill = (fill_rgbas is not None and rgbas is fill_rgbas)

                    if is_fill:
                        ostart = fill_offsets[pid]
                        oend = fill_offsets[pid + 1]
                        if oend > ostart and oend <= len(shaded_fills):
                            return shaded_fills[ostart:oend]
                    else:
                        ostart = stroke_offsets[pid]
                        oend = stroke_offsets[pid + 1]
                        if oend > ostart and oend <= len(shaded_strokes):
                            return shaded_strokes[ostart:oend]
                except Exception:
                    pass

        return _orig_modified_rgbas(self, vmobject, rgbas)

    ThreeDCamera.modified_rgbas = _patched_modified_rgbas
