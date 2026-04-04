"""Tests for Rust projection vs Python/numpy reference implementation."""
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


def _python_project_points(points, frame_center, rot_matrix, focal_distance, zoom):
    """Reference Python implementation matching ThreeDCamera.project_points."""
    pts = points - frame_center
    pts = np.dot(pts, rot_matrix.T)
    zs = pts[:, 2]
    for i in (0, 1):
        factor = focal_distance / (focal_distance - zs)
        factor[(focal_distance - zs) < 0] = 1e6
        pts[:, i] *= factor * zoom
    return pts


def test_projection_identity():
    """Identity rotation, zero center → points scaled by perspective."""
    points = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [-1.0, -2.0, 5.0]])
    pool = _make_pool_with_points([points])

    fc = np.zeros(3, dtype=np.float64)
    rot = np.eye(3, dtype=np.float64)
    fd = 20.0
    zoom = 1.0

    rust_result = np.array(project_all_points(pool, fc, rot, fd, zoom))
    py_result = _python_project_points(points.copy(), fc, rot, fd, zoom)

    # Z values: Rust keeps raw z, Python doesn't zero it in our ref impl
    np.testing.assert_allclose(rust_result[:, :2], py_result[:, :2], atol=1e-10)
    np.testing.assert_allclose(rust_result[:, 2], py_result[:, 2], atol=1e-10)


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

    rust_result = np.array(project_all_points(pool, fc, rot, fd, zoom))
    py_result = _python_project_points(points.copy(), fc, rot, fd, zoom)

    np.testing.assert_allclose(rust_result, py_result, atol=1e-10)


def test_projection_with_offset():
    """Non-zero frame center."""
    points = np.array([[5.0, 5.0, 5.0]])
    pool = _make_pool_with_points([points])

    fc = np.array([2.0, 3.0, 1.0], dtype=np.float64)
    rot = np.eye(3, dtype=np.float64)
    fd = 20.0
    zoom = 2.0

    rust_result = np.array(project_all_points(pool, fc, rot, fd, zoom))
    py_result = _python_project_points(points.copy(), fc, rot, fd, zoom)

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

    rust_result = np.array(project_all_points(pool, fc, rot, fd, zoom))
    py_result = _python_project_points(all_pts.copy(), fc, rot, fd, zoom)

    np.testing.assert_allclose(rust_result, py_result, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
