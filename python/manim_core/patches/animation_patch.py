"""
Monkey-patches for animation hot paths.

Patches:
  - Animation.begin → registers animation mobjects into pool,
    strips stale _pool_id from deepcopy'd starting_mobject/target_mobject
"""
from manim_core.pool_manager import get_pool_manager


def _strip_pool_ids(mob):
    """Recursively strip _pool_id from a mobject and its submobjects."""
    if hasattr(mob, '_pool_id'):
        del mob._pool_id
    for sub in mob.submobjects:
        _strip_pool_ids(sub)


def patch_animation():
    from manim.animation.animation import Animation

    _orig_begin = Animation.begin

    def _patched_begin(self):
        pm = get_pool_manager()
        if pm is not None and hasattr(self, 'mobject') and self.mobject is not None:
            pm.ensure_registered(self.mobject)

        result = _orig_begin(self)

        # After begin(), deepcopy'd mobjects inherit stale _pool_id.
        # Strip them — they'll be registered lazily when needed.
        if pm is not None:
            for attr in ('starting_mobject', 'target_mobject', 'target_copy'):
                mob = getattr(self, attr, None)
                if mob is not None:
                    _strip_pool_ids(mob)

        return result

    Animation.begin = _patched_begin
