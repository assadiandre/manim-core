use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::mesh_pool::MeshPool;

/// Recompute the flat DFS family_order traversal.
/// Replaces recursive get_family() calls — 3.2M calls in the profiled scene.
#[pyfunction]
pub fn recompute_family_order(pool: &mut MeshPool) {
    if !pool.family_order_dirty {
        return;
    }

    let n = pool.num_objects as usize;
    pool.family_order.clear();
    pool.family_order.reserve(n);

    let mut roots: Vec<u32> = Vec::new();
    for i in 0..n {
        if pool.parent_ids[i] < 0 {
            roots.push(i as u32);
        }
    }

    let mut stack: Vec<u32> = Vec::with_capacity(n);
    for &root in roots.iter().rev() {
        stack.push(root);
    }

    while let Some(node) = stack.pop() {
        pool.family_order.push(node);
        let children = &pool.children[node as usize];
        for &child in children.iter().rev() {
            stack.push(child);
        }
    }

    pool.family_order_dirty = false;
}

/// Get the pre-computed family order as a numpy array.
#[pyfunction]
pub fn get_family_order<'py>(py: Python<'py>, pool: &mut MeshPool) -> Bound<'py, PyArray1<u32>> {
    if pool.family_order_dirty {
        recompute_family_order(pool);
    }
    Array1::from_vec(pool.family_order.clone()).into_pyarray_bound(py)
}

/// Z-sort objects by average z-coordinate after rotation.
/// Returns sorted object IDs (back-to-front for painter's algorithm).
#[pyfunction]
pub fn z_sort<'py>(
    py: Python<'py>,
    pool: &mut MeshPool,
    rotation_matrix: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray1<u32>> {
    if pool.family_order_dirty {
        recompute_family_order(pool);
    }

    let rot = rotation_matrix.as_array();
    let r20 = rot[[2, 0]];
    let r21 = rot[[2, 1]];
    let r22 = rot[[2, 2]];

    let n = pool.family_order.len();
    let mut z_values: Vec<(u32, f64)> = Vec::with_capacity(n);

    for &obj_id in &pool.family_order {
        let idx = obj_id as usize;
        let start = pool.point_offsets[idx] as usize;
        let end = pool.point_offsets[idx + 1] as usize;

        if end <= start {
            z_values.push((obj_id, 0.0));
            continue;
        }

        // Use bounding-box center to match Python's get_z_index_reference_point()
        // which calls get_center() = (min + max) / 2
        let mut min_xyz = pool.points[start];
        let mut max_xyz = pool.points[start];
        for i in (start + 1)..end {
            let p = &pool.points[i];
            for j in 0..3 {
                if p[j] < min_xyz[j] { min_xyz[j] = p[j]; }
                if p[j] > max_xyz[j] { max_xyz[j] = p[j]; }
            }
        }
        let cx = (min_xyz[0] + max_xyz[0]) * 0.5;
        let cy = (min_xyz[1] + max_xyz[1]) * 0.5;
        let cz = (min_xyz[2] + max_xyz[2]) * 0.5;
        let z = r20 * cx + r21 * cy + r22 * cz;
        z_values.push((obj_id, z));
    }

    z_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_ids: Vec<u32> = z_values.iter().map(|(id, _)| *id).collect();
    Array1::from_vec(sorted_ids).into_pyarray_bound(py)
}

/// Get family for a specific object (subtree rooted at obj_id), DFS order.
#[pyfunction]
pub fn get_family_for<'py>(
    py: Python<'py>,
    pool: &MeshPool,
    obj_id: u32,
) -> Bound<'py, PyArray1<u32>> {
    let mut result = Vec::new();
    let mut stack = vec![obj_id];

    while let Some(node) = stack.pop() {
        result.push(node);
        let children = &pool.children[node as usize];
        for &child in children.iter().rev() {
            stack.push(child);
        }
    }

    Array1::from_vec(result).into_pyarray_bound(py)
}
