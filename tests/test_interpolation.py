"""Tests for Rust pool cloning and interpolation."""
import numpy as np
import pytest

from manim_core._rust import MeshPool, clone_pool, interpolate_pools, hash_pool_state


def _make_pool(point_val=1.0, fill_val=0.5):
    pool = MeshPool()
    pts = np.array([[point_val, 0, 0], [0, point_val, 0], [0, 0, point_val]], dtype=np.float64)
    pool.register(
        pts,
        np.array([[fill_val, fill_val, fill_val, 1.0]], dtype=np.float64),
        np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float64),
        np.zeros((0, 4), dtype=np.float64),
        1.0, 0.0, 0.0,
        np.array([1, 0, 0], dtype=np.float64),
        False, -1,
    )
    return pool


def test_clone_produces_equal_hash():
    """Cloned pool should have identical hash."""
    pool = _make_pool()
    cloned = clone_pool(pool)
    assert hash_pool_state(pool) == hash_pool_state(cloned)


def test_clone_is_independent():
    """Mutating clone should not affect original."""
    pool = _make_pool()
    h_orig = hash_pool_state(pool)
    cloned = clone_pool(pool)
    cloned.update_points(0, np.array([[99, 0, 0], [0, 99, 0], [0, 0, 99]], dtype=np.float64))
    assert hash_pool_state(pool) == h_orig


def test_interpolation_at_zero():
    """alpha=0 should produce the start pool."""
    start = _make_pool(1.0, 0.2)
    end = _make_pool(10.0, 0.8)
    target = clone_pool(start)

    interpolate_pools(target, start, end, 0.0, 0)

    # Target should match start
    start_pts = np.array(start.get_points(0))
    target_pts = np.array(target.get_points(0))
    np.testing.assert_allclose(target_pts, start_pts, atol=1e-10)


def test_interpolation_at_one():
    """alpha=1 should produce the end pool."""
    start = _make_pool(1.0, 0.2)
    end = _make_pool(10.0, 0.8)
    target = clone_pool(start)

    interpolate_pools(target, start, end, 1.0, 0)

    end_pts = np.array(end.get_points(0))
    target_pts = np.array(target.get_points(0))
    np.testing.assert_allclose(target_pts, end_pts, atol=1e-10)


def test_interpolation_midpoint():
    """alpha=0.5 should produce midpoint."""
    start = _make_pool(0.0, 0.0)
    end = _make_pool(10.0, 1.0)
    target = clone_pool(start)

    interpolate_pools(target, start, end, 0.5, 0)

    target_pts = np.array(target.get_points(0))
    expected_pts = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]], dtype=np.float64)
    np.testing.assert_allclose(target_pts, expected_pts, atol=1e-10)


def test_interpolation_smooth():
    """path_func_type=1 should use smooth (3t^2 - 2t^3)."""
    start = _make_pool(0.0, 0.0)
    end = _make_pool(10.0, 1.0)
    target = clone_pool(start)

    alpha = 0.5
    smooth_alpha = 3 * alpha**2 - 2 * alpha**3  # = 0.5 for alpha=0.5

    interpolate_pools(target, start, end, alpha, 1)

    target_pts = np.array(target.get_points(0))
    expected = np.array([[10 * smooth_alpha, 0, 0],
                         [0, 10 * smooth_alpha, 0],
                         [0, 0, 10 * smooth_alpha]], dtype=np.float64)
    np.testing.assert_allclose(target_pts, expected, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
