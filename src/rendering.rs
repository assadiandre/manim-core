use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::mesh_pool::MeshPool;

/// Extract cubic bezier subpaths from VMobject points and return render metadata.
///
/// Phase 4 Option B (data prep only): Rust prepares all the data,
/// Python loops over objects but each iteration is 4-5 cairo calls with zero computation.
#[pyfunction]
pub fn prepare_render_data<'py>(
    py: Python<'py>,
    pool: &MeshPool,
    render_order: PyReadonlyArray1<u32>,
) -> PyResult<PyObject> {
    let order = render_order.as_array();
    let dict = pyo3::types::PyDict::new_bound(py);

    let mut all_subpath_starts: Vec<Vec<u32>> = Vec::with_capacity(order.len());
    let mut all_subpath_ends: Vec<Vec<u32>> = Vec::with_capacity(order.len());

    for &obj_id in order.iter() {
        let idx = obj_id as usize;
        let start = pool.point_offsets[idx] as usize;
        let end = pool.point_offsets[idx + 1] as usize;
        let npts = end - start;

        let mut subpath_starts = Vec::new();
        let mut subpath_ends = Vec::new();

        if npts >= 4 {
            let mut sp_start = 0u32;
            let mut i = 0;

            while i + 3 < npts {
                let p0 = &pool.points[start + i];
                let p1 = &pool.points[start + i + 1];
                let p2 = &pool.points[start + i + 2];
                let p3 = &pool.points[start + i + 3];

                let is_null = (p0[0] == p1[0] && p0[1] == p1[1] && p0[2] == p1[2])
                    && (p0[0] == p2[0] && p0[1] == p2[1] && p0[2] == p2[2])
                    && (p0[0] == p3[0] && p0[1] == p3[1] && p0[2] == p3[2]);

                if is_null {
                    if i as u32 > sp_start {
                        subpath_starts.push(sp_start);
                        subpath_ends.push(i as u32);
                    }
                    sp_start = (i + 4) as u32;
                }
                i += 4;
            }

            if (npts as u32) > sp_start && sp_start < npts as u32 {
                subpath_starts.push(sp_start);
                subpath_ends.push(npts as u32);
            }
        }

        all_subpath_starts.push(subpath_starts);
        all_subpath_ends.push(subpath_ends);
    }

    let py_starts = PyList::empty_bound(py);
    let py_ends = PyList::empty_bound(py);

    for (starts, ends) in all_subpath_starts.iter().zip(all_subpath_ends.iter()) {
        let s = Array1::from_vec(starts.clone()).into_pyarray_bound(py);
        let e = Array1::from_vec(ends.clone()).into_pyarray_bound(py);
        py_starts.append(s)?;
        py_ends.append(e)?;
    }

    dict.set_item("subpath_starts", py_starts)?;
    dict.set_item("subpath_ends", py_ends)?;

    Ok(dict.into())
}

/// Batch-compute which objects have visible fill or stroke.
#[pyfunction]
pub fn compute_visibility<'py>(
    py: Python<'py>,
    pool: &MeshPool,
    object_ids: PyReadonlyArray1<u32>,
) -> PyResult<(
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<bool>>,
)> {
    let ids = object_ids.as_array();
    let n = ids.len();
    let mut has_fill = vec![false; n];
    let mut has_stroke = vec![false; n];
    let mut has_bg_stroke = vec![false; n];

    for (i, &obj_id) in ids.iter().enumerate() {
        let idx = obj_id as usize;

        let fstart = pool.fill_rgba_offsets[idx] as usize;
        let fend = pool.fill_rgba_offsets[idx + 1] as usize;
        for j in fstart..fend {
            if pool.fill_rgbas[j][3] > 0.0 {
                has_fill[i] = true;
                break;
            }
        }

        if pool.stroke_widths[idx] > 0.0 {
            let sstart = pool.stroke_rgba_offsets[idx] as usize;
            let send = pool.stroke_rgba_offsets[idx + 1] as usize;
            for j in sstart..send {
                if pool.stroke_rgbas[j][3] > 0.0 {
                    has_stroke[i] = true;
                    break;
                }
            }
        }

        if pool.bg_stroke_widths[idx] > 0.0 {
            let bstart = pool.bg_stroke_rgba_offsets[idx] as usize;
            let bend = pool.bg_stroke_rgba_offsets[idx + 1] as usize;
            for j in bstart..bend {
                if pool.bg_stroke_rgbas[j][3] > 0.0 {
                    has_bg_stroke[i] = true;
                    break;
                }
            }
        }
    }

    Ok((
        Array1::from_vec(has_fill).into_pyarray_bound(py),
        Array1::from_vec(has_stroke).into_pyarray_bound(py),
        Array1::from_vec(has_bg_stroke).into_pyarray_bound(py),
    ))
}
