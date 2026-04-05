"""Tests for Rust shading vs manim's actual get_shaded_rgb / modified_rgbas."""
import numpy as np
import pytest

from manim_core._rust import MeshPool, shade_all_objects


def _make_shaded_pool(points, fill_rgbas, stroke_rgbas=None, shade_in_3d=True):
    """Create a pool with one object for shading tests."""
    pool = MeshPool()
    if stroke_rgbas is None:
        stroke_rgbas = [[1.0, 1.0, 1.0, 1.0]]
    pool.register(
        np.array(points, dtype=np.float64).reshape(-1, 3),
        np.array(fill_rgbas, dtype=np.float64).reshape(-1, 4),
        np.array(stroke_rgbas, dtype=np.float64).reshape(-1, 4),
        np.zeros((0, 4), dtype=np.float64),
        1.0, 0.0, 0.0,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        shade_in_3d, -1,
    )
    return pool


def _manim_get_shaded_rgb(rgb, point, unit_normal, light_source):
    """Reference: manim's get_shaded_rgb from color/core.py:1604-1636."""
    to_sun = light_source - point
    mag = np.linalg.norm(to_sun)
    if mag > 1e-10:
        to_sun = to_sun / mag
    dot = np.dot(unit_normal, to_sun)
    light = 0.5 * dot ** 3
    if light < 0:
        light *= 0.5
    return rgb + light


def _manim_get_unit_normal(v1, v2, tol=1e-6):
    """Reference: manim's get_unit_normal from space_ops.py."""
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    div1 = max(np.abs(v1))
    div2 = max(np.abs(v2))
    DOWN = np.array([0.0, 0.0, -1.0])

    if div1 == 0.0 and div2 == 0.0:
        return DOWN

    if div1 == 0.0:
        u = v2 / div2
    elif div2 == 0.0:
        u = v1 / div1
    else:
        u1, u2 = v1 / div1, v2 / div2
        cp = np.cross(u1, u2)
        cp_norm = np.linalg.norm(cp)
        if cp_norm > tol:
            return cp / cp_norm
        u = u1

    if abs(u[0]) < tol and abs(u[1]) < tol:
        return DOWN
    cp = np.array([-u[0]*u[2], -u[1]*u[2], u[0]**2 + u[1]**2])
    return cp / np.linalg.norm(cp)


def _compute_expected_shading(points, fill_rgba, light_source):
    """Compute what manim's modified_rgbas would produce for a single shaded object."""
    pts = np.array(points).reshape(-1, 3)

    # Start corner: first 3 points
    v1_s = pts[1] - pts[0]
    v2_s = pts[2] - pts[1]
    start_normal = _manim_get_unit_normal(v1_s, v2_s)
    start_point = pts[0]

    # End corner: last 3 points
    v1_e = pts[-2] - pts[-3]
    v2_e = pts[-1] - pts[-2]
    end_normal = _manim_get_unit_normal(v1_e, v2_e)
    end_point = pts[-1]

    fill = np.array(fill_rgba).reshape(-1, 4)
    row0 = fill[0].copy()
    row1 = fill[1].copy() if len(fill) >= 2 else fill[0].copy()

    shaded0 = _manim_get_shaded_rgb(row0[:3].copy(), start_point, start_normal, light_source)
    shaded1 = _manim_get_shaded_rgb(row1[:3].copy(), end_point, end_normal, light_source)

    return np.array([
        [shaded0[0], shaded0[1], shaded0[2], row0[3]],
        [shaded1[0], shaded1[1], shaded1[2], row1[3]],
    ])


# --- Passthrough tests ---

def test_no_shading_passthrough():
    """Objects with shade_in_3d=False should return original colors."""
    pool = _make_shaded_pool(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0.5, 0.3, 0.7, 1.0]],
        shade_in_3d=False,
    )
    light = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    fills, strokes = shade_all_objects(pool, light)
    fills = np.array(fills)
    np.testing.assert_allclose(fills[0], [0.5, 0.3, 0.7, 1.0], atol=1e-10)


# --- Formula correctness tests ---

def test_shading_matches_manim_facing_light():
    """Face pointing toward light: verify exact match with manim's formula."""
    # Triangle in XY plane, normal points in +Z
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    fill = [[0.5, 0.5, 0.5, 1.0]]
    light = np.array([0.0, 0.0, 100.0], dtype=np.float64)

    pool = _make_shaded_pool(points, fill, shade_in_3d=True)
    fills, _ = shade_all_objects(pool, light)
    fills = np.array(fills)

    expected = _compute_expected_shading(points, fill, light)
    np.testing.assert_allclose(fills[:2], expected, atol=1e-10)


def test_shading_matches_manim_facing_away():
    """Face pointing away from light: verify exact match."""
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    fill = [[0.5, 0.5, 0.5, 1.0]]
    light = np.array([0.0, 0.0, -100.0], dtype=np.float64)

    pool = _make_shaded_pool(points, fill, shade_in_3d=True)
    fills, _ = shade_all_objects(pool, light)
    fills = np.array(fills)

    expected = _compute_expected_shading(points, fill, light)
    np.testing.assert_allclose(fills[:2], expected, atol=1e-10)


def test_shading_matches_manim_oblique():
    """Light at oblique angle."""
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    fill = [[0.3, 0.6, 0.9, 0.8]]
    light = np.array([5.0, 3.0, 7.0], dtype=np.float64)

    pool = _make_shaded_pool(points, fill, shade_in_3d=True)
    fills, _ = shade_all_objects(pool, light)
    fills = np.array(fills)

    expected = _compute_expected_shading(points, fill, light)
    np.testing.assert_allclose(fills[:2], expected, atol=1e-10)


def test_shading_preserves_alpha():
    """Alpha channel should not be modified by shading."""
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    fill = [[0.5, 0.5, 0.5, 0.8]]
    light = np.array([0.0, 0.0, 100.0], dtype=np.float64)

    pool = _make_shaded_pool(points, fill, shade_in_3d=True)
    fills, _ = shade_all_objects(pool, light)
    fills = np.array(fills)

    # Both rows should preserve alpha
    assert fills[0, 3] == pytest.approx(0.8)
    assert fills[1, 3] == pytest.approx(0.8)


def test_shading_two_rows_output():
    """Shaded objects with 1 fill row should produce 2 output rows (manim behavior)."""
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    fill = [[0.5, 0.5, 0.5, 1.0]]  # 1 row in

    pool = _make_shaded_pool(points, fill, shade_in_3d=True)
    fills, strokes = shade_all_objects(pool, np.array([0.0, 0.0, 100.0], dtype=np.float64))
    fills = np.array(fills)
    strokes = np.array(strokes)

    # manim produces 2 rows minimum for shaded objects
    assert fills.shape[0] >= 2
    assert strokes.shape[0] >= 2


def test_shading_start_end_normals_differ():
    """When start and end corners have different normals, shading should differ."""
    # Curved path: start normal ≠ end normal
    points = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0],  # start: normal in +Z
        [1, 1, 0], [1, 1, 1], [0, 1, 1],  # end: normal tilted
    ]
    fill = [[0.5, 0.5, 0.5, 1.0]]
    light = np.array([5.0, 5.0, 5.0], dtype=np.float64)

    pool = _make_shaded_pool(points, fill, shade_in_3d=True)
    fills, _ = shade_all_objects(pool, light)
    fills = np.array(fills)

    expected = _compute_expected_shading(points, fill, light)
    np.testing.assert_allclose(fills[:2], expected, atol=1e-10)

    # Start and end should be shaded differently
    assert not np.allclose(fills[0, :3], fills[1, :3])


def test_shading_degenerate_collinear_points():
    """Collinear points should produce DOWN normal (matching manim)."""
    # All points along X axis → v1 and v2 are parallel
    points = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    fill = [[0.5, 0.5, 0.5, 1.0]]
    light = np.array([0.0, 0.0, 100.0], dtype=np.float64)

    pool = _make_shaded_pool(points, fill, shade_in_3d=True)
    fills, _ = shade_all_objects(pool, light)
    fills = np.array(fills)

    expected = _compute_expected_shading(points, fill, light)
    np.testing.assert_allclose(fills[:2], expected, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
