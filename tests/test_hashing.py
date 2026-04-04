"""Tests for Rust hashing."""
import numpy as np
import pytest

from manim_core._rust import MeshPool, hash_pool_state, hash_objects


def _make_pool():
    pool = MeshPool()
    pts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    pool.register(
        pts,
        np.array([[0.5, 0.3, 0.7, 1.0]], dtype=np.float64),
        np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float64),
        np.zeros((0, 4), dtype=np.float64),
        1.0, 0.0, 0.0,
        np.array([1, 0, 0], dtype=np.float64),
        True, -1,
    )
    return pool


def test_hash_deterministic():
    """Same pool state should produce same hash."""
    pool1 = _make_pool()
    pool2 = _make_pool()
    assert hash_pool_state(pool1) == hash_pool_state(pool2)


def test_hash_changes_with_points():
    """Changing a point should change the hash."""
    pool = _make_pool()
    h1 = hash_pool_state(pool)

    pool.update_points(0, np.array([[1, 2, 3.001], [4, 5, 6], [7, 8, 9]], dtype=np.float64))
    h2 = hash_pool_state(pool)

    assert h1 != h2


def test_hash_changes_with_colors():
    """Changing a fill color should change the hash."""
    pool = _make_pool()
    h1 = hash_pool_state(pool)

    pool.update_fill_rgbas(0, np.array([[0.5, 0.3, 0.8, 1.0]], dtype=np.float64))
    h2 = hash_pool_state(pool)

    assert h1 != h2


def test_hash_objects_subset():
    """hash_objects should work on a subset of objects."""
    pool = MeshPool()
    pts1 = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64)
    pts2 = np.array([[0, 1, 0], [0, 2, 0], [0, 3, 0]], dtype=np.float64)

    id1 = pool.register(pts1, np.zeros((1, 4), dtype=np.float64),
                        np.zeros((1, 4), dtype=np.float64),
                        np.zeros((0, 4), dtype=np.float64),
                        0.0, 0.0, 0.0, np.array([1, 0, 0], dtype=np.float64), False, -1)
    id2 = pool.register(pts2, np.zeros((1, 4), dtype=np.float64),
                        np.zeros((1, 4), dtype=np.float64),
                        np.zeros((0, 4), dtype=np.float64),
                        0.0, 0.0, 0.0, np.array([1, 0, 0], dtype=np.float64), False, -1)

    h_both = hash_objects(pool, [id1, id2])
    h_one = hash_objects(pool, [id1])
    h_two = hash_objects(pool, [id2])

    assert h_both != h_one
    assert h_both != h_two
    assert h_one != h_two


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
