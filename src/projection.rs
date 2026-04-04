use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::mesh_pool::MeshPool;

/// Project all points in the pool through a 3D camera transformation.
///
/// Replaces ThreeDCamera.project_points() — called per-object 586K times in Python.
/// This does it in one batched pass over all points.
///
/// Algorithm (from three_d_camera.py):
///   1. Subtract frame_center from all points
///   2. Multiply by rotation_matrix (3x3)
///   3. Apply perspective: x' = x * fd/(fd - z), y' = y * fd/(fd - z)
///   4. Scale by zoom factor
///   5. Keep z for z-sorting
#[pyfunction]
pub fn project_all_points<'py>(
    py: Python<'py>,
    pool: &MeshPool,
    frame_center: PyReadonlyArray1<f64>,
    rotation_matrix: PyReadonlyArray2<f64>,
    focal_distance: f64,
    zoom: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let fc = frame_center.as_array();
    let rot = rotation_matrix.as_array();
    let n = pool.points.len();

    let fc0 = fc[0];
    let fc1 = fc[1];
    let fc2 = fc[2];

    let r00 = rot[[0, 0]];
    let r01 = rot[[0, 1]];
    let r02 = rot[[0, 2]];
    let r10 = rot[[1, 0]];
    let r11 = rot[[1, 1]];
    let r12 = rot[[1, 2]];
    let r20 = rot[[2, 0]];
    let r21 = rot[[2, 1]];
    let r22 = rot[[2, 2]];

    let use_perspective = focal_distance.is_finite() && focal_distance > 0.0;

    let mut result = vec![[0.0f64; 3]; n];

    result.par_iter_mut().enumerate().for_each(|(i, out)| {
        let p = &pool.points[i];
        let dx = p[0] - fc0;
        let dy = p[1] - fc1;
        let dz = p[2] - fc2;

        let rx = r00 * dx + r01 * dy + r02 * dz;
        let ry = r10 * dx + r11 * dy + r12 * dz;
        let rz = r20 * dx + r21 * dy + r22 * dz;

        if use_perspective {
            let factor = if (focal_distance - rz).abs() > 1e-10 {
                focal_distance / (focal_distance - rz)
            } else {
                focal_distance / 1e-10
            };
            out[0] = rx * factor * zoom;
            out[1] = ry * factor * zoom;
            out[2] = rz;
        } else {
            out[0] = rx * zoom;
            out[1] = ry * zoom;
            out[2] = rz;
        }
    });

    let mut arr = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        arr[[i, 0]] = result[i][0];
        arr[[i, 1]] = result[i][1];
        arr[[i, 2]] = result[i][2];
    }
    Ok(arr.into_pyarray_bound(py))
}

/// Project points for a specific set of pool objects (by their IDs).
#[pyfunction]
pub fn project_points_for_objects<'py>(
    py: Python<'py>,
    pool: &MeshPool,
    object_ids: PyReadonlyArray1<u32>,
    frame_center: PyReadonlyArray1<f64>,
    rotation_matrix: PyReadonlyArray2<f64>,
    focal_distance: f64,
    zoom: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let fc = frame_center.as_array();
    let rot = rotation_matrix.as_array();
    let ids = object_ids.as_array();

    let fc0 = fc[0];
    let fc1 = fc[1];
    let fc2 = fc[2];
    let r00 = rot[[0, 0]];
    let r01 = rot[[0, 1]];
    let r02 = rot[[0, 2]];
    let r10 = rot[[1, 0]];
    let r11 = rot[[1, 1]];
    let r12 = rot[[1, 2]];
    let r20 = rot[[2, 0]];
    let r21 = rot[[2, 1]];
    let r22 = rot[[2, 2]];
    let use_perspective = focal_distance.is_finite() && focal_distance > 0.0;

    let mut total = 0usize;
    for &id in ids.iter() {
        let idx = id as usize;
        total += (pool.point_offsets[idx + 1] - pool.point_offsets[idx]) as usize;
    }

    let mut arr = Array2::<f64>::zeros((total, 3));
    let mut write_idx = 0usize;

    for &id in ids.iter() {
        let idx = id as usize;
        let start = pool.point_offsets[idx] as usize;
        let end = pool.point_offsets[idx + 1] as usize;

        for i in start..end {
            let p = &pool.points[i];
            let dx = p[0] - fc0;
            let dy = p[1] - fc1;
            let dz = p[2] - fc2;
            let rx = r00 * dx + r01 * dy + r02 * dz;
            let ry = r10 * dx + r11 * dy + r12 * dz;
            let rz = r20 * dx + r21 * dy + r22 * dz;

            if use_perspective {
                let denom = focal_distance - rz;
                let factor = if denom.abs() > 1e-10 {
                    focal_distance / denom
                } else {
                    focal_distance / 1e-10
                };
                arr[[write_idx, 0]] = rx * factor * zoom;
                arr[[write_idx, 1]] = ry * factor * zoom;
            } else {
                arr[[write_idx, 0]] = rx * zoom;
                arr[[write_idx, 1]] = ry * zoom;
            }
            arr[[write_idx, 2]] = rz;
            write_idx += 1;
        }
    }

    Ok(arr.into_pyarray_bound(py))
}
