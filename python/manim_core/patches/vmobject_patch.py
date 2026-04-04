"""
Monkey-patches for VMobject and Mobject.

Patches:
  - Mobject.get_family → reads pre-computed family_order from pool
  - extract_mobject_family_members → uses pool tree for batch extraction
"""
import itertools as it

from manim_core._rust import get_family_for
from manim_core.pool_manager import get_scene_pool, get_pool_manager


def patch_vmobject():
    from manim.mobject.mobject import Mobject
    from manim.utils import family as family_mod
    from manim.utils.iterables import remove_list_redundancies

    _orig_get_family = Mobject.get_family
    _orig_extract = family_mod.extract_mobject_family_members

    def _patched_get_family(self, recurse=True):
        pool = get_scene_pool()
        pm = get_pool_manager()
        if pool is not None and pm is not None and hasattr(self, '_pool_id'):
            try:
                family_ids = get_family_for(pool, self._pool_id)
                result = []
                for pid in family_ids:
                    pid = int(pid)
                    obj = pm._id_to_obj.get(pid)
                    if obj is not None:
                        result.append(obj)
                if result:
                    return result
            except Exception:
                pass
        return _orig_get_family(self, recurse)

    def _patched_extract_family(mobjects, use_z_index=False, only_those_with_points=False):
        pool = get_scene_pool()
        pm = get_pool_manager()
        if pool is not None and pm is not None:
            try:
                all_pooled = all(hasattr(m, '_pool_id') for m in mobjects)
                if all_pooled and mobjects:
                    if only_those_with_points:
                        method = Mobject.family_members_with_points
                    else:
                        method = _patched_get_family
                    extracted = remove_list_redundancies(
                        list(it.chain(*(method(m) for m in mobjects))),
                    )
                    if use_z_index:
                        return sorted(extracted, key=lambda m: m.z_index)
                    return extracted
            except Exception:
                pass
        return _orig_extract(mobjects, use_z_index=use_z_index,
                             only_those_with_points=only_those_with_points)

    Mobject.get_family = _patched_get_family
    family_mod.extract_mobject_family_members = _patched_extract_family

    try:
        from manim.camera import camera as camera_mod
        camera_mod.extract_mobject_family_members = _patched_extract_family
    except Exception:
        pass
