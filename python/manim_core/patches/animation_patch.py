"""
Monkey-patches for animation hot paths.

Patches:
  - Animation.begin → registers animation mobjects into pool
"""
from manim_core.pool_manager import get_pool_manager


def patch_animation():
    from manim.animation.animation import Animation

    _orig_begin = Animation.begin

    def _patched_begin(self):
        pm = get_pool_manager()
        if pm is not None:
            if hasattr(self, 'mobject') and self.mobject is not None:
                pm.ensure_registered(self.mobject)
        return _orig_begin(self)

    Animation.begin = _patched_begin
