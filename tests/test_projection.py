"""Tests for Rust projection vs manim's actual ThreeDCamera.project_points."""
import numpy as np
import pytest

from manim_core._rust import MeshPool, project_all_points


def _make_pool_with_points(points_list):
    """Helper: create a MeshPool and register objects with given point arrays."""
    pool = MeshPool()
    for pts in points_list:
        pts = np.array(pts, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        pool.register(
            pts,
            np.zeros((1, 4), dtype=np.float64),  # fill
            np.zeros((1, 4), dtype=np.float64),  # stroke
            np.zeros((0, 4), dtype=np.float64),  # bg stroke
            0.0, 0.0, 0.0,
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            False, -1,
        )
    return pool


def _manim_project_points(points, frame_center, rot_matrix, focal_distance, zoom, exponential=True):
    """Reference implementation matching manim's ThreeDCamera.project_points exactly."""
    pts = points - frame_center
    pts = np.dot(pts, rot_matrix.T)
    zs = pts[:, 2]
    for i in (0, 1):
        if exponential:
            # manim's exponential_projection path
            factor = np.exp(zs / focal_distance)
            lt0 = zs < 0
            factor[lt0] = focal_distance / (focal_distance - zs[lt0])
        else:
            factor = focal_distance / (focal_distance - zs)
            factor[(focal_distance - zs) < 0] = 1e6
        pts[:, i] *= factor * zoom
    return pts


# --- Exponential projection tests (ThreeDScene default) ---

def test_projection_exponential_identity():
    """Identity rotation, zero center, exponential mode."""
    points = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [-1.0, -2.0, 5.0]])
    pool = _make_pool_with_points([points])

    fc = np.zeros(3, dtype=np.float64)
    rot = np.eye(3, dtype=np.float64)
    fd = 20.0
    zoom = 1.0

    rust_result = np.array(project_all_points(pool, fc, rot, fd, zoom, True))
    py_result = _manim_project_points(points.copy(), fc, rot, fd, zoom, True)

    np.testing.assert_allclose(rust_result, py_result, atol=1e-10)


def test_projection_exponential_positive_z():
    """Points with z > 0 use exp(z/fd) in exponential mode — this is where the old code was wrong."""
    points = np.array([[1.0, 1.0, 15.0], [2.0, 3.0, 19.0]])
    pool = _make_pool_with_points([points])

    fc = np.zeros(3, dtype=np.float64)
    rot = np.eye(3, dtype=np.float64)
    fd = 20.0
    zoom = 1.0

    rust_result = np.array(project_all_points(pool, fc, rot, fd, zoom, True))
    py_result = _manim_project_points(points.copy(), fc, rot, fd, zoom, True)

    # exp(15/20) ≈ 2.117, NOT fd/(fd-15) = 4.0
    np.testing.assert_allclose(rust_result[:, :2], py_result[:, :2], atol=1e-10)


def test_projection_exponential_negative_z():
    """Points with z < 0 use standard fd/(fd-z) even in exponential mode."""
    points = np.array([[1.0, 1.0, -5.0], [2.0, 3.0, -15.0]])
    pool = _make_pool_with_points([points])

    fc = np.zeros(3, dtype=np.float64)
    rot = np.eye(3, dtype=np.float64)
    fd = 20.0
    zoom = 1.0

    rust_result = np.array(project_all_points(pool, fc, rot, fd, zoom, True))
    py_result = _manim_project_points(points.copy(), fc, rot, fd, zoom, True)

    np.testing.assert_allclose(rust_result, py_result, atol=1e-10)


def test_projection_exponential_mixed_z():
    """Mix of positive and negative z values in exponential mode."""
    points = np.array([[1.0, 2.0, 10.0], [3.0, 4.0, -10.0], [5.0, 6.0, 0.0]])
    pool = _make_pool_with_points([points])

    fc = np.zeros(3, dtype=np.float64)
    rot = np.eye(3, dtype=np.float64)
    fd = 20.0
    zoom = 1.5

    rust_result = np.array(project_all_points(pool, fc, rot, fd, zoom, True))
    py_result = _manim_project_points(points.copy(), fc, rot, fd, zoom, True)

    np.testing.assert_allclose(rust_result, py_result, atol=1e-10)


# --- Standard projection tests (exponential_projection=False) ---

def test_projection_standard_identity():
    """Identity rotation, zero center, standard mode."""
    points = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [-1.0, -2.0, 5.0]])
    pool = _make_pool_with_points([points])

    fc = np.zeros(3, dtype=np.float64)
    rot = np.eye(3, dtype=np.float64)
    fd = 20.0
    zoom = 1.0

    rust_result = np.array(project_all_points(pool, fc, rot, fd, zoom, False))
    py_result = _manim_project_points(points.copy(), fc, rot, fd, zoom, False)

    np.testing.assert_allclose(rust_result, py_result, atol=1e-10)


def test_projection_with_rotation():
    """45-degree rotation about Z axis."""
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pool = _make_pool_with_points([points])

    fc = np.zeros(3, dtype=np.float64)
    angle = np.pi / 4
    rot = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1],
    ], dtype=np.float64)
    fd = 20.0
    zoom = 1.0

    # Both modes should agree when z=0
    for exp in [True, False]:
        rust_result = np.array(project_all_points(pool, fc, rot, fd, zoom, exp))
        py_result = _manim_project_points(points.copy(), fc, rot, fd, zoom, exp)
        np.testing.assert_allclose(rust_result, py_result, atol=1e-10)


def test_projection_with_offset():
    """Non-zero frame center."""
    points = np.array([[5.0, 5.0, 5.0]])
    pool = _make_pool_with_points([points])

    fc = np.array([2.0, 3.0, 1.0], dtype=np.float64)
    rot = np.eye(3, dtype=np.float64)
    fd = 20.0
    zoom = 2.0

    for exp in [True, False]:
        rust_result = np.array(project_all_points(pool, fc, rot, fd, zoom, exp))
        py_result = _manim_project_points(points.copy(), fc, rot, fd, zoom, exp)
        np.testing.assert_allclose(rust_result, py_result, atol=1e-10)


def test_projection_multiple_objects():
    """Multiple objects registered in pool, all projected correctly."""
    obj1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    obj2 = np.array([[-1, -2, -3]], dtype=np.float64)
    all_pts = np.vstack([obj1, obj2])
    pool = _make_pool_with_points([obj1, obj2])

    fc = np.zeros(3, dtype=np.float64)
    rot = np.eye(3, dtype=np.float64)
    fd = 20.0
    zoom = 1.0

    for exp in [True, False]:
        rust_result = np.array(project_all_points(pool, fc, rot, fd, zoom, exp))
        py_result = _manim_project_points(all_pts.copy(), fc, rot, fd, zoom, exp)
        np.testing.assert_allclose(rust_result, py_result, atol=1e-10)


def test_projection_default_is_exponential():
    """Default argument should be exponential=True (matching ThreeDScene)."""
    points = np.array([[1.0, 1.0, 15.0]])
    pool = _make_pool_with_points([points])

    fc = np.zeros(3, dtype=np.float64)
    rot = np.eye(3, dtype=np.float64)
    fd = 20.0
    zoom = 1.0

    # Call without specifying exponential_projection
    rust_default = np.array(project_all_points(pool, fc, rot, fd, zoom))
    rust_explicit = np.array(project_all_points(pool, fc, rot, fd, zoom, True))

    np.testing.assert_allclose(rust_default, rust_explicit, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
