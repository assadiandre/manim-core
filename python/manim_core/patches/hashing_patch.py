"""
Monkey-patch for manim's scene hashing.

Patches:
  - get_hash_from_play_call → uses Rust hash_pool_state for mobject state
"""
import zlib

from manim_core._rust import hash_pool_state
from manim_core.pool_manager import get_scene_pool


def patch_hashing():
    from manim.utils import hashing as hashing_mod

    _orig_get_hash = hashing_mod.get_hash_from_play_call

    def _patched_get_hash_from_play_call(
        scene_object, camera_object, animations_list, current_mobjects_list
    ):
        pool = get_scene_pool()
        if pool is not None and pool.len() > 0:
            try:
                # Use Rust to hash the pool state (replaces JSON serialization of mobjects)
                pool_hash = hash_pool_state(pool)

                # Still use Python for camera and animations (small, fast enough)
                from manim.utils.hashing import get_json, _Memoizer
                from time import perf_counter
                import logging

                logger = logging.getLogger("manim")
                logger.debug("Hashing ...")
                t_start = perf_counter()

                _Memoizer.mark_as_processed(scene_object)
                camera_json = get_json(camera_object)
                animations_list_json = [
                    get_json(x) for x in sorted(animations_list, key=str)
                ]

                hash_camera = zlib.crc32(repr(camera_json).encode())
                hash_animations = zlib.crc32(repr(animations_list_json).encode())
                # Use the fast Rust hash for mobjects instead of JSON
                hash_current_mobjects = pool_hash & 0xFFFFFFFF  # truncate to u32 for compat

                hash_complete = (
                    f"{hash_camera}_{hash_animations}_{hash_current_mobjects}"
                )
                t_end = perf_counter()
                logger.debug(
                    "Hashing done in %(time)s s.", {"time": str(t_end - t_start)[:8]}
                )
                _Memoizer.reset_already_processed()
                return hash_complete
            except Exception:
                pass

        return _orig_get_hash(
            scene_object, camera_object, animations_list, current_mobjects_list
        )

    hashing_mod.get_hash_from_play_call = _patched_get_hash_from_play_call
