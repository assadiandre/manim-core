"""
Monkey-patch for Scene to manage the MeshPool lifecycle.

Hooks:
  - Scene.begin_animations → registers all scene mobjects into the pool
"""
from manim_core.pool_manager import PoolManager, get_pool_manager


def patch_scene():
    from manim.scene.scene import Scene

    _orig_begin_animations = Scene.begin_animations

    def _patched_begin_animations(self):
        pm = get_pool_manager()
        if pm is None:
            pm = PoolManager()
            pm.activate()

        pm.register_scene_mobjects(self)

        if self.animations:
            for anim in self.animations:
                if hasattr(anim, 'mobject') and anim.mobject is not None:
                    pm.ensure_registered(anim.mobject)

        return _orig_begin_animations(self)

    Scene.begin_animations = _patched_begin_animations
