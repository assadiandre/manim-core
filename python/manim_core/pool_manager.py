"""
Thread-local pool lifecycle management.

Pool references are stored in thread-local globals, NOT on mobject instances,
to avoid breaking deepcopy (MeshPool is a Rust object that can't be pickled).

Each VMobject only gets a lightweight `_pool_id: int` attribute.
"""
import threading
import numpy as np

from manim_core._rust import MeshPool

_thread_local = threading.local()


def get_scene_pool():
    """Get the active MeshPool for the current thread/scene, or None."""
    return getattr(_thread_local, "pool", None)


def set_scene_pool(pool):
    """Set the active MeshPool for the current thread/scene."""
    _thread_local.pool = pool


def get_pool_manager():
    """Get the active PoolManager for the current thread, or None."""
    return getattr(_thread_local, "pool_manager", None)


def set_pool_manager(pm):
    """Set the active PoolManager for the current thread."""
    _thread_local.pool_manager = pm


class PoolManager:
    """Manages the lifecycle of a MeshPool for a scene."""

    def __init__(self):
        self.pool = MeshPool()
        self._obj_to_id = {}   # id(VMobject) -> pool_id
        self._id_to_obj = {}   # pool_id -> VMobject
        self._dirty = set()    # set of id(VMobject) that need sync

    def activate(self):
        set_scene_pool(self.pool)
        set_pool_manager(self)

    def deactivate(self):
        set_scene_pool(None)
        set_pool_manager(None)

    def is_registered(self, mobject):
        return id(mobject) in self._obj_to_id

    def mark_dirty(self, mobject):
        """Mark a mobject as needing sync to pool next frame."""
        obj_key = id(mobject)
        if obj_key in self._obj_to_id:
            self._dirty.add(obj_key)

    def unregister(self, mobject):
        """Remove a mobject from the pool manager tracking."""
        obj_key = id(mobject)
        pool_id = self._obj_to_id.pop(obj_key, None)
        if pool_id is not None:
            self._id_to_obj.pop(pool_id, None)
            self._dirty.discard(obj_key)
            if hasattr(mobject, '_pool_id'):
                del mobject._pool_id

    def unregister_family(self, mobject):
        """Remove a mobject and all its submobjects from pool manager tracking."""
        self.unregister(mobject)
        for sub in mobject.submobjects:
            self.unregister_family(sub)

    def register_mobject(self, mobject, parent_id=-1):
        """Register a VMobject and all its submobjects into the pool."""
        from manim.mobject.types.vectorized_mobject import VMobject

        if not isinstance(mobject, VMobject):
            return None

        obj_key = id(mobject)
        if obj_key in self._obj_to_id:
            return self._obj_to_id[obj_key]

        points = mobject.points if mobject.points is not None and len(mobject.points) > 0 else np.zeros((0, 3))

        try:
            fill_rgbas = mobject.get_fill_rgbas()
        except Exception:
            fill_rgbas = np.zeros((1, 4))

        try:
            stroke_rgbas = mobject.get_stroke_rgbas()
        except Exception:
            stroke_rgbas = np.zeros((1, 4))

        try:
            bg_stroke_rgbas = mobject.get_stroke_rgbas(background=True)
        except Exception:
            bg_stroke_rgbas = np.zeros((0, 4))

        if bg_stroke_rgbas is None or len(bg_stroke_rgbas) == 0:
            bg_stroke_rgbas = np.zeros((0, 4))

        if points.ndim == 1:
            points = points.reshape(-1, 3) if len(points) > 0 else np.zeros((0, 3))
        if fill_rgbas.ndim == 1:
            fill_rgbas = fill_rgbas.reshape(-1, 4)
        if stroke_rgbas.ndim == 1:
            stroke_rgbas = stroke_rgbas.reshape(-1, 4)
        if bg_stroke_rgbas.ndim == 1:
            bg_stroke_rgbas = bg_stroke_rgbas.reshape(-1, 4)

        try:
            stroke_width = float(mobject.get_stroke_width())
        except Exception:
            stroke_width = 0.0

        bg_stroke_width = float(getattr(mobject, "background_stroke_width", 0.0) or 0.0)
        sheen_factor = float(getattr(mobject, "sheen_factor", 0.0) or 0.0)
        sheen_direction = np.array(getattr(mobject, "sheen_direction", [1.0, 0.0, 0.0]), dtype=np.float64).flatten()[:3]
        if len(sheen_direction) < 3:
            sheen_direction = np.array([1.0, 0.0, 0.0])
        shade_in_3d = bool(getattr(mobject, "shade_in_3d", False))

        from manim.constants import LineJointType, CapStyleType
        joint_type = int(getattr(mobject, 'joint_type', LineJointType.AUTO).value)
        cap_style = int(getattr(mobject, 'cap_style', CapStyleType.AUTO).value)
        tolerance = float(getattr(mobject, 'tolerance_for_point_equality', 1e-6))

        pool_id = self.pool.register(
            np.ascontiguousarray(points, dtype=np.float64),
            np.ascontiguousarray(fill_rgbas, dtype=np.float64),
            np.ascontiguousarray(stroke_rgbas, dtype=np.float64),
            np.ascontiguousarray(bg_stroke_rgbas, dtype=np.float64),
            stroke_width,
            bg_stroke_width,
            sheen_factor,
            sheen_direction,
            shade_in_3d,
            joint_type,
            cap_style,
            tolerance,
            parent_id,
        )

        self._obj_to_id[obj_key] = pool_id
        self._id_to_obj[pool_id] = mobject
        # Only store a plain int on the mobject — no references to pool/manager
        mobject._pool_id = pool_id

        for sub in mobject.submobjects:
            self.register_mobject(sub, parent_id=pool_id)

        return pool_id

    def register_scene_mobjects(self, scene):
        for mob in scene.mobjects:
            self.register_mobject(mob)

    def ensure_registered(self, mobject):
        if self.is_registered(mobject):
            return self._obj_to_id[id(mobject)]
        return self.register_mobject(mobject)

    @staticmethod
    def _ensure_contiguous(arr):
        """Return a C-contiguous float64 array, avoiding copy if already correct."""
        if arr.flags['C_CONTIGUOUS'] and arr.dtype == np.float64:
            return arr
        return np.ascontiguousarray(arr, dtype=np.float64)

    def sync_mobject_to_pool(self, mobject):
        pool_id = self._obj_to_id.get(id(mobject))
        if pool_id is None:
            return
        # Points
        points = mobject.points
        if points is not None and len(points) > 0:
            try:
                self.pool.update_points(pool_id, self._ensure_contiguous(points))
            except (ValueError, Exception):
                pass
        # Colors
        for attr, updater in [
            ("fill_rgbas", self.pool.update_fill_rgbas),
            ("stroke_rgbas", self.pool.update_stroke_rgbas),
        ]:
            arr = getattr(mobject, attr, None)
            if arr is not None and len(arr) > 0:
                try:
                    updater(pool_id, self._ensure_contiguous(arr))
                except (ValueError, Exception):
                    pass
        # Scalars
        try:
            from manim.constants import LineJointType, CapStyleType
            self.pool.update_scalars(
                pool_id,
                float(getattr(mobject, "stroke_width", 0.0) or 0.0),
                float(getattr(mobject, "background_stroke_width", 0.0) or 0.0),
                float(getattr(mobject, "sheen_factor", 0.0) or 0.0),
                np.array(getattr(mobject, "sheen_direction", [1, 0, 0]), dtype=np.float64).flatten()[:3],
                bool(getattr(mobject, "shade_in_3d", False)),
                int(getattr(mobject, 'joint_type', LineJointType.AUTO).value),
                int(getattr(mobject, 'cap_style', CapStyleType.AUTO).value),
                float(getattr(mobject, 'tolerance_for_point_equality', 1e-6)),
            )
        except Exception:
            pass

    def sync_all(self):
        """Sync only dirty mobjects' current state into the pool."""
        if not self._dirty:
            return
        dirty = self._dirty
        self._dirty = set()
        for obj_pyid in dirty:
            pool_id = self._obj_to_id.get(obj_pyid)
            if pool_id is None:
                continue
            mob = self._id_to_obj.get(pool_id)
            if mob is not None:
                self.sync_mobject_to_pool(mob)

    def __enter__(self):
        self.activate()
        return self

    def __exit__(self, *args):
        self.deactivate()
