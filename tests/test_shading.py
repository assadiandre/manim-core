"""Tests for Rust shading vs Python reference implementation."""
import numpy as np
import pytest

from manim_core._rust import MeshPool, shade_all_objects


def _make_shaded_pool(points, fill_rgbas, shade_in_3d=True):
    """Create a pool with one object for shading tests."""
    pool = MeshPool()
    pool.register(
        np.array(points, dtype=np.float64).reshape(-1, 3),
        np.array(fill_rgbas, dtype=np.float64).reshape(-1, 4),
        np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float64),  # stroke
        np.zeros((0, 4), dtype=np.float64),
        1.0, 0.0, 0.0,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        shade_in_3d, -1,
    )
    return pool


def test_no_shading_passthrough():
    """Objects with shade_in_3d=False should return original colors."""
    pool = _make_shaded_pool(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0.5, 0.3, 0.7, 1.0]],
        shade_in_3d=False,
    )
    light = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    fills, strokes = shade_all_objects(pool, light, 0.2, 0.7)
    fills = np.array(fills)
    np.testing.assert_allclose(fills[0], [0.5, 0.3, 0.7, 1.0], atol=1e-10)


def test_shading_brightens_facing_light():
    """Face pointing toward light should be brightened."""
    # Triangle in XY plane, normal points in +Z
    pool = _make_shaded_pool(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0.5, 0.5, 0.5, 1.0]],
        shade_in_3d=True,
    )
    # Light directly above → normal · light_dir ≈ 1
    light = np.array([0.0, 0.0, 100.0], dtype=np.float64)
    fills, _ = shade_all_objects(pool, light, 0.5, 0.7)
    fills = np.array(fills)
    # Should be brighter than original 0.5
    assert fills[0, 0] > 0.5
    assert fills[0, 1] > 0.5
    assert fills[0, 2] > 0.5


def test_shading_darkens_facing_away():
    """Face pointing away from light should be darkened."""
    pool = _make_shaded_pool(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0.5, 0.5, 0.5, 1.0]],
        shade_in_3d=True,
    )
    # Light below → normal · light_dir ≈ -1
    light = np.array([0.0, 0.0, -100.0], dtype=np.float64)
    fills, _ = shade_all_objects(pool, light, 0.5, 0.7)
    fills = np.array(fills)
    # Should be darker than original 0.5
    assert fills[0, 0] < 0.5
    assert fills[0, 1] < 0.5
    assert fills[0, 2] < 0.5


def test_shading_preserves_alpha():
    """Alpha channel should not be modified by shading."""
    pool = _make_shaded_pool(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0.5, 0.5, 0.5, 0.8]],
        shade_in_3d=True,
    )
    light = np.array([0.0, 0.0, 100.0], dtype=np.float64)
    fills, _ = shade_all_objects(pool, light, 0.5, 0.7)
    fills = np.array(fills)
    np.testing.assert_allclose(fills[0, 3], 0.8, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
