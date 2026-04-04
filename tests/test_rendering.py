"""Tests for Rust render data preparation."""
import numpy as np
import pytest

from manim_core._rust import MeshPool, prepare_render_data, compute_visibility


def _register_obj(pool, pts, fill_alpha=1.0, stroke_width=1.0, stroke_alpha=1.0):
    return pool.register(
        np.array(pts, dtype=np.float64).reshape(-1, 3),
        np.array([[0.5, 0.5, 0.5, fill_alpha]], dtype=np.float64),
        np.array([[1.0, 1.0, 1.0, stroke_alpha]], dtype=np.float64),
        np.zeros((0, 4), dtype=np.float64),
        stroke_width, 0.0, 0.0,
        np.array([1, 0, 0], dtype=np.float64),
        False, -1,
    )


def test_compute_visibility():
    """Test visibility detection based on alpha and stroke width."""
    pool = MeshPool()
    # Visible fill + stroke
    id1 = _register_obj(pool, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                         fill_alpha=1.0, stroke_width=2.0, stroke_alpha=1.0)
    # No fill (alpha=0), has stroke
    id2 = _register_obj(pool, [[2, 0, 0], [3, 0, 0], [2, 1, 0], [3, 1, 0]],
                         fill_alpha=0.0, stroke_width=1.0, stroke_alpha=1.0)
    # Has fill, no stroke (width=0)
    id3 = _register_obj(pool, [[4, 0, 0], [5, 0, 0], [4, 1, 0], [5, 1, 0]],
                         fill_alpha=0.5, stroke_width=0.0, stroke_alpha=1.0)

    ids = np.array([id1, id2, id3], dtype=np.uint32)
    has_fill, has_stroke, has_bg = compute_visibility(pool, ids)

    has_fill = np.array(has_fill)
    has_stroke = np.array(has_stroke)

    assert has_fill[0] == True
    assert has_fill[1] == False
    assert has_fill[2] == True

    assert has_stroke[0] == True
    assert has_stroke[1] == True
    assert has_stroke[2] == False


def test_prepare_render_data_basic():
    """Test that prepare_render_data returns subpath info."""
    pool = MeshPool()
    # 8 points = 2 cubic bezier segments
    pts = [[i, 0, 0] for i in range(8)]
    _register_obj(pool, pts)

    order = np.array([0], dtype=np.uint32)
    data = prepare_render_data(pool, order)

    assert "subpath_starts" in data
    assert "subpath_ends" in data
    assert len(data["subpath_starts"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
