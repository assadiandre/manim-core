"""
Monkey-patches for manim hot paths.
Import apply_all_patches() to activate Rust acceleration.
"""


def apply_all_patches():
    """Apply all Rust-accelerated monkey-patches to manim."""
    from manim_core.patches.scene_patch import patch_scene
    from manim_core.patches.camera_patch import patch_three_d_camera
    from manim_core.patches.vmobject_patch import patch_vmobject
    from manim_core.patches.hashing_patch import patch_hashing
    from manim_core.patches.animation_patch import patch_animation

    patch_scene()
    patch_three_d_camera()
    patch_vmobject()
    patch_hashing()
    patch_animation()
