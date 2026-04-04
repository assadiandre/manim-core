"""Tests for Rust z-sort and family order."""
import numpy as np
import pytest

from manim_core._rust import MeshPool, z_sort, get_family_order, get_family_for, recompute_family_order


def _register_obj(pool, center_z, parent_id=-1):
    """Register a simple 3-point object centered at (0, 0, center_z)."""
    pts = np.array([
        [0, 0, center_z],
        [1, 0, center_z],
        [0, 1, center_z],
    ], dtype=np.float64)
    return pool.register(
        pts,
        np.zeros((1, 4), dtype=np.float64),
        np.zeros((1, 4), dtype=np.float64),
        np.zeros((0, 4), dtype=np.float64),
        0.0, 0.0, 0.0,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        True, parent_id,
    )


def test_z_sort_order():
    """Objects should be sorted by average z after rotation (back-to-front)."""
    pool = MeshPool()
    id_near = _register_obj(pool, 10.0)   # closer to camera
    id_far = _register_obj(pool, -10.0)   # further from camera
    id_mid = _register_obj(pool, 0.0)     # middle

    rot = np.eye(3, dtype=np.float64)
    sorted_ids = list(z_sort(pool, rot))

    # Back-to-front: far (-10) first, then mid (0), then near (10)
    assert sorted_ids == [id_far, id_mid, id_near]


def test_z_sort_with_rotation():
    """Z-sort with a rotation that swaps axes."""
    pool = MeshPool()
    # Object at x=10, z=0
    pts_a = np.array([[10, 0, 0], [11, 0, 0], [10, 1, 0]], dtype=np.float64)
    # Object at x=-10, z=0
    pts_b = np.array([[-10, 0, 0], [-9, 0, 0], [-10, 1, 0]], dtype=np.float64)

    id_a = pool.register(pts_a, np.zeros((1, 4), dtype=np.float64),
                         np.zeros((1, 4), dtype=np.float64),
                         np.zeros((0, 4), dtype=np.float64),
                         0.0, 0.0, 0.0, np.array([1, 0, 0], dtype=np.float64), True, -1)
    id_b = pool.register(pts_b, np.zeros((1, 4), dtype=np.float64),
                         np.zeros((1, 4), dtype=np.float64),
                         np.zeros((0, 4), dtype=np.float64),
                         0.0, 0.0, 0.0, np.array([1, 0, 0], dtype=np.float64), True, -1)

    # Rotation that maps x→z (90° about Y)
    rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
    sorted_ids = list(z_sort(pool, rot))

    # rot 3rd row is [-1, 0, 0], so z_rotated = -x
    # obj_a (avg x≈10.33) → z≈-10.33, obj_b (avg x≈-9.67) → z≈9.67
    # Back-to-front (ascending z): a first, then b
    assert sorted_ids == [id_a, id_b]


def test_family_order():
    """DFS family order for a simple tree."""
    pool = MeshPool()
    root = _register_obj(pool, 0.0)
    child1 = _register_obj(pool, 1.0, parent_id=root)
    child2 = _register_obj(pool, 2.0, parent_id=root)
    grandchild = _register_obj(pool, 3.0, parent_id=child1)

    order = list(get_family_order(pool))
    # DFS: root, child1, grandchild, child2
    assert order == [root, child1, grandchild, child2]


def test_get_family_for():
    """get_family_for returns subtree rooted at given object."""
    pool = MeshPool()
    root = _register_obj(pool, 0.0)
    child1 = _register_obj(pool, 1.0, parent_id=root)
    child2 = _register_obj(pool, 2.0, parent_id=root)
    grandchild = _register_obj(pool, 3.0, parent_id=child1)

    family = list(get_family_for(pool, child1))
    assert family == [child1, grandchild]

    family_root = list(get_family_for(pool, root))
    assert family_root == [root, child1, grandchild, child2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
