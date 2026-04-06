"""
Monkey-patches for animation hot paths.

Patches:
  - Animation.begin → registers animation mobjects into pool,
    strips stale _pool_id from deepcopy'd starting_mobject/target_mobject,
    then re-registers them with fresh pool slots for Rust interpolation.
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
    _orig_interpolate = Animation.interpolate

    def _patched_begin(self):
        pm = get_pool_manager()

        # Strip pool IDs from self.mobject BEFORE _orig_begin so that
        # during begin() (which calls align_data then interpolate(0)),
        # get_family falls through to the original Python implementation
        # instead of reading stale pool family trees.
        if pm is not None and hasattr(self, 'mobject') and self.mobject is not None:
            _strip_pool_ids(self.mobject)
            pm.unregister_family(self.mobject)
            # Also strip target_mobject if it was pre-registered
            target = getattr(self, 'target_mobject', None)
            if target is not None:
                _strip_pool_ids(target)
                pm.unregister_family(target)

        result = _orig_begin(self)

        # After begin(), register all mobjects (now with final
        # post-align_data family structure) into the pool.
        if pm is not None:
            for attr in ('mobject', 'starting_mobject', 'target_mobject', 'target_copy'):
                mob = getattr(self, attr, None)
                if mob is not None:
                    _strip_pool_ids(mob)

            for attr in ('mobject', 'starting_mobject', 'target_mobject', 'target_copy'):
                mob = getattr(self, attr, None)
                if mob is not None:
                    pm.register_mobject(mob)

            # Cache the animated family for per-frame dirty marking
            # (avoids calling get_family() every frame)
            if hasattr(self, 'mobject') and self.mobject is not None:
                self._pool_animated_family = self.mobject.get_family()
                for m in self._pool_animated_family:
                    pm.mark_dirty(m)
            else:
                self._pool_animated_family = None

        return result

    def _patched_interpolate(self, alpha):
        """After each frame's interpolation, mark animated mobjects dirty.

        Uses cached family from begin() to avoid repeated get_family() calls.
        """
        _orig_interpolate(self, alpha)
        family = getattr(self, '_pool_animated_family', None)
        if family is not None:
            pm = get_pool_manager()
            if pm is not None:
                dirty = pm._dirty
                obj_to_id = pm._obj_to_id
                for m in family:
                    key = id(m)
                    if key in obj_to_id:
                        dirty.add(key)

    Animation.begin = _patched_begin
    Animation.interpolate = _patched_interpolate
