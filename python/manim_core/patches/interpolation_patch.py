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
            pid = self._pool_id
            pid1 = mobject1._pool_id
            pid2 = mobject2._pool_id

            # Verify all three objects are the actual registered objects
            # (not copies that inherited stale _pool_id)
            if (pm._id_to_obj.get(pid) is self
                    and pm._id_to_obj.get(pid1) is mobject1
                    and pm._id_to_obj.get(pid2) is mobject2):

                # Verify color array sizes match between all three objects.
                # Rust uses min(sizes) which skips entries; Python broadcasts.
                # Only use Rust path when sizes are identical.
                try:
                    tf = pool.fill_rgba_range(pid)
                    s1f = pool.fill_rgba_range(pid1)
                    s2f = pool.fill_rgba_range(pid2)
                    ts = pool.stroke_rgba_range(pid)
                    s1s = pool.stroke_rgba_range(pid1)
                    s2s = pool.stroke_rgba_range(pid2)
                    tb = pool.bg_stroke_rgba_range(pid)
                    s1b = pool.bg_stroke_rgba_range(pid1)
                    s2b = pool.bg_stroke_rgba_range(pid2)
                    fill_ok = (tf[1] - tf[0]) == (s1f[1] - s1f[0]) == (s2f[1] - s2f[0])
                    stroke_ok = (ts[1] - ts[0]) == (s1s[1] - s1s[0]) == (s2s[1] - s2s[0])
                    bg_ok = (tb[1] - tb[0]) == (s1b[1] - s1b[0]) == (s2b[1] - s2b[0])

                    if fill_ok and stroke_ok and bg_ok:
                        interpolate_object_attrs(pool, pid, pid1, pid2, alpha)

                        # Read back interpolated values from pool to Python mobject
                        self.fill_rgbas = np.array(pool.get_fill_rgbas(pid))
                        self.stroke_rgbas = np.array(pool.get_stroke_rgbas(pid))
                        self.background_stroke_rgbas = np.array(pool.get_bg_stroke_rgbas(pid))

                        sw, bsw, sf, s3d = pool.get_scalars(pid)
                        self.stroke_width = sw
                        self.background_stroke_width = bsw
                        self.sheen_factor = sf
                        self.sheen_direction = np.array(pool.get_sheen_direction(pid))
                        pm.mark_dirty(self)
                        return
                except Exception:
                    pass

        return _orig_interpolate_color(self, mobject1, mobject2, alpha)

    VMobject.interpolate_color = _patched_interpolate_color
