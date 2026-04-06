"""
Monkey-patches for VMobject and Mobject.

Patches:
  - Mobject.get_family → reads pre-computed family_order from pool
  - extract_mobject_family_members → uses get_family_order for batch extraction
  - VMobject.set_points / set_fill / set_stroke → marks dirty for sync
"""
import itertools as it

from manim_core._rust import get_family_for, get_family_order
from manim_core.pool_manager import get_scene_pool, get_pool_manager


def patch_vmobject():
    from manim.mobject.mobject import Mobject
    from manim.mobject.types.vectorized_mobject import VMobject
    from manim.utils import family as family_mod
    from manim.utils.iterables import remove_list_redundancies

    _orig_get_family = Mobject.get_family
    _orig_extract = family_mod.extract_mobject_family_members

    def _patched_get_family(self, recurse=True):
        pool = get_scene_pool()
        pm = get_pool_manager()
        if pool is not None and pm is not None and hasattr(self, '_pool_id'):
            # Only use pool path if self is the actual registered object,
            # not a copy that inherited _pool_id
            if pm._id_to_obj.get(self._pool_id) is self:
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
                    # Use single Rust call for full family order instead of
                    # per-mobject get_family + remove_list_redundancies
                    family_ids = get_family_order(pool)
                    # Build lookup of which top-level mobjects' families we want
                    top_ids = set()
                    for m in mobjects:
                        top_ids.add(m._pool_id)
                    # Walk full family order, include objects belonging to requested trees
                    # First, get all IDs that belong to requested families
                    wanted_ids = set()
                    for m in mobjects:
                        for pid in get_family_for(pool, m._pool_id):
                            wanted_ids.add(int(pid))
                    # Walk in family order to preserve tree traversal order
                    seen = set()
                    extracted = []
                    for pid in family_ids:
                        pid = int(pid)
                        if pid not in wanted_ids or pid in seen:
                            continue
                        seen.add(pid)
                        obj = pm._id_to_obj.get(pid)
                        if obj is None:
                            continue
                        if only_those_with_points:
                            if not obj.has_points():
                                continue
                        extracted.append(obj)
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

    # Phase 2.2: Hook mutation paths to mark dirty
    _orig_set_points = VMobject.set_points
    _orig_set_fill = VMobject.set_fill
    _orig_set_stroke = VMobject.set_stroke

    def _dirty_set_points(self, points):
        result = _orig_set_points(self, points)
        pm = get_pool_manager()
        if pm is not None:
            pm.mark_dirty(self)
        return result

    def _dirty_set_fill(self, *args, **kwargs):
        result = _orig_set_fill(self, *args, **kwargs)
        pm = get_pool_manager()
        if pm is not None:
            pm.mark_dirty(self)
        return result

    def _dirty_set_stroke(self, *args, **kwargs):
        result = _orig_set_stroke(self, *args, **kwargs)
        pm = get_pool_manager()
        if pm is not None:
            pm.mark_dirty(self)
        return result

    VMobject.set_points = _dirty_set_points
    VMobject.set_fill = _dirty_set_fill
    VMobject.set_stroke = _dirty_set_stroke
