use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::mesh_pool::MeshPool;

#[inline]
fn compute_unit_normal(points: &[[f64; 3]]) -> [f64; 3] {
    if points.len() < 3 {
        return [0.0, 0.0, 1.0];
    }

    let p0 = &points[0];
    let p1 = &points[1];
    let p2 = &points[2];

    let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v2 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];

    let cx = v1[1] * v2[2] - v1[2] * v2[1];
    let cy = v1[2] * v2[0] - v1[0] * v2[2];
    let cz = v1[0] * v2[1] - v1[1] * v2[0];

    let mag = (cx * cx + cy * cy + cz * cz).sqrt();
    if mag < 1e-10 {
        for i in 0..points.len().saturating_sub(2) {
            let a = &points[i];
            let mid = points.len() / 2;
            let b = &points[mid.min(i + 1)];
            let c = &points[(points.len() - 1).max(i + 2)];

            let u1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let u2 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];

            let nx = u1[1] * u2[2] - u1[2] * u2[1];
            let ny = u1[2] * u2[0] - u1[0] * u2[2];
            let nz = u1[0] * u2[1] - u1[1] * u2[0];

            let m = (nx * nx + ny * ny + nz * nz).sqrt();
            if m > 1e-10 {
                return [nx / m, ny / m, nz / m];
            }
        }
        [0.0, 0.0, 1.0]
    } else {
        [cx / mag, cy / mag, cz / mag]
    }
}

#[inline]
fn shade_rgba(
    rgba: &[f64; 4],
    normal: &[f64; 3],
    light_dir: &[f64; 3],
    reflectiveness: f64,
    shadow: f64,
) -> [f64; 4] {
    let dot = (normal[0] * light_dir[0] + normal[1] * light_dir[1] + normal[2] * light_dir[2])
        .clamp(-1.0, 1.0);

    let mut r = rgba[0];
    let mut g = rgba[1];
    let mut b = rgba[2];

    if dot >= 0.0 {
        let t = dot * reflectiveness;
        r += (1.0 - r) * t;
        g += (1.0 - g) * t;
        b += (1.0 - b) * t;
    } else {
        let t = -dot * shadow;
        r *= 1.0 - t;
        g *= 1.0 - t;
        b *= 1.0 - t;
    }

    [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0), rgba[3]]
}

/// Shade all objects in the pool, returning shaded fill and stroke colors.
/// Replaces ThreeDCamera.modified_rgbas() + get_shaded_rgb() + get_unit_normal().
#[pyfunction]
pub fn shade_all_objects<'py>(
    py: Python<'py>,
    pool: &MeshPool,
    light_source_position: PyReadonlyArray1<f64>,
    reflectiveness: f64,
    shadow: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let ls = light_source_position.as_array();
    let light_pos = [ls[0], ls[1], ls[2]];
    let num_obj = pool.num_objects as usize;

    let mut normals = vec![[0.0f64; 3]; num_obj];
    let mut light_dirs = vec![[0.0f64; 3]; num_obj];

    for obj_id in 0..num_obj {
        if !pool.shade_in_3d[obj_id] {
            continue;
        }
        let start = pool.point_offsets[obj_id] as usize;
        let end = pool.point_offsets[obj_id + 1] as usize;
        if end <= start {
            normals[obj_id] = [0.0, 0.0, 1.0];
            light_dirs[obj_id] = [0.0, 0.0, 1.0];
            continue;
        }

        let pts = &pool.points[start..end];
        normals[obj_id] = compute_unit_normal(pts);

        let mut cx = 0.0f64;
        let mut cy = 0.0f64;
        let mut cz = 0.0f64;
        let npts = (end - start) as f64;
        for p in pts {
            cx += p[0];
            cy += p[1];
            cz += p[2];
        }
        cx /= npts;
        cy /= npts;
        cz /= npts;

        let dx = light_pos[0] - cx;
        let dy = light_pos[1] - cy;
        let dz = light_pos[2] - cz;
        let mag = (dx * dx + dy * dy + dz * dz).sqrt();
        if mag > 1e-10 {
            light_dirs[obj_id] = [dx / mag, dy / mag, dz / mag];
        } else {
            light_dirs[obj_id] = [0.0, 0.0, 1.0];
        }
    }

    let nfill = pool.fill_rgbas.len();
    let mut shaded_fills = Array2::<f64>::zeros((nfill, 4));
    for obj_id in 0..num_obj {
        let fstart = pool.fill_rgba_offsets[obj_id] as usize;
        let fend = pool.fill_rgba_offsets[obj_id + 1] as usize;
        for i in fstart..fend {
            let shaded = if pool.shade_in_3d[obj_id] {
                shade_rgba(&pool.fill_rgbas[i], &normals[obj_id], &light_dirs[obj_id], reflectiveness, shadow)
            } else {
                pool.fill_rgbas[i]
            };
            shaded_fills[[i, 0]] = shaded[0];
            shaded_fills[[i, 1]] = shaded[1];
            shaded_fills[[i, 2]] = shaded[2];
            shaded_fills[[i, 3]] = shaded[3];
        }
    }

    let nstroke = pool.stroke_rgbas.len();
    let mut shaded_strokes = Array2::<f64>::zeros((nstroke, 4));
    for obj_id in 0..num_obj {
        let sstart = pool.stroke_rgba_offsets[obj_id] as usize;
        let send = pool.stroke_rgba_offsets[obj_id + 1] as usize;
        for i in sstart..send {
            let shaded = if pool.shade_in_3d[obj_id] {
                shade_rgba(&pool.stroke_rgbas[i], &normals[obj_id], &light_dirs[obj_id], reflectiveness, shadow)
            } else {
                pool.stroke_rgbas[i]
            };
            shaded_strokes[[i, 0]] = shaded[0];
            shaded_strokes[[i, 1]] = shaded[1];
            shaded_strokes[[i, 2]] = shaded[2];
            shaded_strokes[[i, 3]] = shaded[3];
        }
    }

    Ok((
        shaded_fills.into_pyarray_bound(py),
        shaded_strokes.into_pyarray_bound(py),
    ))
}
