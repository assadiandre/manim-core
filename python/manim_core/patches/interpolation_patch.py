"""
Monkey-patch for VMobject.interpolate_color.

Patches:
  - VMobject.interpolate_color → uses Rust intra-pool interpolation for all 7
    color/scalar attributes in one call instead of 7 separate Python lerps.
"""
import numpy as np

from manim_core._rust import interpolate_object_attrs
from manim_core.pool_manager import get_scene_pool, get_pool_manager


def patch_interpolation():
    from manim.mobject.types.vectorized_mobject import VMobject

    _orig_interpolate_color = VMobject.interpolate_color

    def _patched_interpolate_color(self, mobject1, mobject2, alpha):
        pool = get_scene_pool()
        pm = get_pool_manager()

        if (pool is not None and pm is not None
                and hasattr(self, '_pool_id')
                and hasattr(mobject1, '_pool_id')
                and hasattr(mobject2, '_pool_id')):
            try:
                interpolate_object_attrs(
                    pool,
                    self._pool_id,
                    mobject1._pool_id,
                    mobject2._pool_id,
                    alpha,
                )
                pid = self._pool_id

                # Read back interpolated values from pool to Python mobject
                self.fill_rgbas = np.array(pool.get_fill_rgbas(pid))
                self.stroke_rgbas = np.array(pool.get_stroke_rgbas(pid))
                self.background_stroke_rgbas = np.array(pool.get_bg_stroke_rgbas(pid))

                sw, bsw, sf, s3d = pool.get_scalars(pid)
                self.stroke_width = sw
                self.background_stroke_width = bsw
                self.sheen_factor = sf
                self.sheen_direction = np.array(pool.get_sheen_direction(pid))
                return
            except Exception:
                pass

        return _orig_interpolate_color(self, mobject1, mobject2, alpha)

    VMobject.interpolate_color = _patched_interpolate_color
