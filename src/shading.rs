use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::mesh_pool::MeshPool;

/// Compute unit normal from two vectors, matching manim's get_unit_normal(v1, v2).
///
/// manim algorithm (space_ops.py:392-442):
///   1. Scale v1, v2 by max(abs(components)) to avoid overflow
///   2. Cross product; if non-zero, normalize and return
///   3. If collinear, pick the non-zero vector and rotate toward Z axis
///   4. If aligned with Z, return DOWN = [0, 0, -1]
#[inline]
fn get_unit_normal(v1: &[f64; 3], v2: &[f64; 3]) -> [f64; 3] {
    let div1 = v1[0].abs().max(v1[1].abs()).max(v1[2].abs());
    let div2 = v2[0].abs().max(v2[1].abs()).max(v2[2].abs());

    const TOL: f64 = 1e-6;
    const DOWN: [f64; 3] = [0.0, 0.0, -1.0];

    if div1 == 0.0 && div2 == 0.0 {
        return DOWN;
    }

    if div1 == 0.0 {
        // v1 is zero, use v2
        let u = [v2[0] / div2, v2[1] / div2, v2[2] / div2];
        return rotate_toward_z_or_down(&u);
    }

    if div2 == 0.0 {
        // v2 is zero, use v1
        let u = [v1[0] / div1, v1[1] / div1, v1[2] / div1];
        return rotate_toward_z_or_down(&u);
    }

    // Normal case: both non-zero
    let u1 = [v1[0] / div1, v1[1] / div1, v1[2] / div1];
    let u2 = [v2[0] / div2, v2[1] / div2, v2[2] / div2];

    let cx = u1[1] * u2[2] - u1[2] * u2[1];
    let cy = u1[2] * u2[0] - u1[0] * u2[2];
    let cz = u1[0] * u2[1] - u1[1] * u2[0];

    let cp_norm = (cx * cx + cy * cy + cz * cz).sqrt();
    if cp_norm > TOL {
        return [cx / cp_norm, cy / cp_norm, cz / cp_norm];
    }

    // Collinear: use u1 to derive a normal
    rotate_toward_z_or_down(&u1)
}

/// For a single non-zero unit-ish vector u, rotate it 90° toward Z axis.
/// If too aligned with Z, return DOWN.
#[inline]
fn rotate_toward_z_or_down(u: &[f64; 3]) -> [f64; 3] {
    const TOL: f64 = 1e-6;
    const DOWN: [f64; 3] = [0.0, 0.0, -1.0];

    if u[0].abs() < TOL && u[1].abs() < TOL {
        return DOWN;
    }
    // (u x [0,0,1]) x u = [-xz, -yz, x²+y²]
    let cpx = -u[0] * u[2];
    let cpy = -u[1] * u[2];
    let cpz = u[0] * u[0] + u[1] * u[1];
    let cp_norm = (cpx * cpx + cpy * cpy + cpz * cpz).sqrt();
    [cpx / cp_norm, cpy / cp_norm, cpz / cp_norm]
}

/// Compute the unit normal for a VMobject at a given point_index.
/// Matches manim's get_3d_vmob_unit_normal (three_d_utils.py:59-72).
///
/// Algorithm:
///   i = point_index
///   im3 = i - 3 if i > 2 else (n_points - 4)
///   ip3 = i + 3 if i < (n_points - 3) else 3
///   v1 = points[ip3] - points[i]
///   v2 = points[im3] - points[i]
///   normal = get_unit_normal(v1, v2)
///   if norm == 0: return UP
#[inline]
fn vmob_unit_normal(points: &[[f64; 3]], point_index: usize) -> [f64; 3] {
    let n = points.len();
    const UP: [f64; 3] = [0.0, 1.0, 0.0];

    // manim: if len(vmob.get_anchors()) <= 2: return UP
    // anchors are at stride n_points_per_curve-1 = 3 for cubic beziers
    // so anchors count ≈ (n + 2) / 3; <= 2 means n < 7
    if n < 7 {
        return UP;
    }

    let i = point_index;
    let im3 = if i > 2 { i - 3 } else { n - 4 };
    let ip3 = if i < n - 3 { i + 3 } else { 3 };

    let v1 = [
        points[ip3][0] - points[i][0],
        points[ip3][1] - points[i][1],
        points[ip3][2] - points[i][2],
    ];
    let v2 = [
        points[im3][0] - points[i][0],
        points[im3][1] - points[i][1],
        points[im3][2] - points[i][2],
    ];

    let result = get_unit_normal(&v1, &v2);
    let norm = (result[0] * result[0] + result[1] * result[1] + result[2] * result[2]).sqrt();
    if norm < 1e-10 {
        return UP;
    }
    result
}

/// Start corner index: always 0 (matches get_3d_vmob_start_corner_index).
#[inline]
fn start_corner_index(_n_points: usize) -> usize {
    0
}

/// End corner index: ((n_points - 1) / 6) * 3 (matches get_3d_vmob_end_corner_index).
#[inline]
fn end_corner_index(n_points: usize) -> usize {
    ((n_points - 1) / 6) * 3
}

/// Compute the unit normal at the start corner.
#[inline]
fn start_corner_unit_normal(points: &[[f64; 3]]) -> [f64; 3] {
    vmob_unit_normal(points, start_corner_index(points.len()))
}

/// Compute the unit normal at the end corner.
#[inline]
fn end_corner_unit_normal(points: &[[f64; 3]]) -> [f64; 3] {
    vmob_unit_normal(points, end_corner_index(points.len()))
}

/// Apply manim's shading formula: get_shaded_rgb (color/core.py:1604-1636)
///
///   to_sun = normalize(light_source - point)
///   light = 0.5 * dot(unit_normal, to_sun)^3
///   if light < 0: light *= 0.5
///   shaded_rgb = rgb + light
#[inline]
fn shade_rgb(rgb: &[f64], normal: &[f64; 3], point: &[f64; 3], light_source: &[f64; 3]) -> [f64; 3] {
    let dx = light_source[0] - point[0];
    let dy = light_source[1] - point[1];
    let dz = light_source[2] - point[2];
    let mag = (dx * dx + dy * dy + dz * dz).sqrt();

    let to_sun = if mag > 1e-10 {
        [dx / mag, dy / mag, dz / mag]
    } else {
        [0.0, 0.0, 0.0]
    };

    let dot = normal[0] * to_sun[0] + normal[1] * to_sun[1] + normal[2] * to_sun[2];
    let mut light = 0.5 * dot * dot * dot;
    if light < 0.0 {
        light *= 0.5;
    }

    [rgb[0] + light, rgb[1] + light, rgb[2] + light]
}

/// Shade all objects in the pool, returning per-object shaded fill and stroke colors
/// plus per-object offset tables into the output arrays.
///
/// Matches manim's ThreeDCamera.modified_rgbas():
///   - For each object with shade_in_3d=True and points > 0:
///     - Takes first 2 rgba rows (repeating if only 1 row), shades them
///     - Always returns exactly 2 rows per shaded object
///   - For non-shaded objects: returns original colors unchanged
///
/// Returns: (fills, strokes, fill_offsets, stroke_offsets)
///   fills:          (total_fill_rows, 4) shaded fill colors
///   strokes:        (total_stroke_rows, 4) shaded stroke colors
///   fill_offsets:   (num_objects + 1,) per-object start offsets into fills
///   stroke_offsets: (num_objects + 1,) per-object start offsets into strokes
#[pyfunction]
pub fn shade_all_objects<'py>(
    py: Python<'py>,
    pool: &MeshPool,
    light_source_position: PyReadonlyArray1<f64>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, numpy::PyArray1<u32>>,
    Bound<'py, numpy::PyArray1<u32>>,
)> {
    let ls = light_source_position.as_array();
    let light_pos = [ls[0], ls[1], ls[2]];
    let num_obj = pool.num_objects as usize;

    // Pre-compute output offsets per object.
    // Shaded objects with points always produce exactly 2 rows.
    // Unshaded objects pass through their original row count.
    let mut fill_offsets = vec![0u32; num_obj + 1];
    let mut stroke_offsets = vec![0u32; num_obj + 1];
    let mut fill_out_rows = 0u32;
    let mut stroke_out_rows = 0u32;

    for obj_id in 0..num_obj {
        fill_offsets[obj_id] = fill_out_rows;
        stroke_offsets[obj_id] = stroke_out_rows;

        let fstart = pool.fill_rgba_offsets[obj_id] as usize;
        let fend = pool.fill_rgba_offsets[obj_id + 1] as usize;
        let sstart = pool.stroke_rgba_offsets[obj_id] as usize;
        let send = pool.stroke_rgba_offsets[obj_id + 1] as usize;
        let pstart = pool.point_offsets[obj_id] as usize;
        let pend = pool.point_offsets[obj_id + 1] as usize;

        if pool.shade_in_3d[obj_id] && pend > pstart {
            fill_out_rows += 2;
            stroke_out_rows += 2;
        } else {
            fill_out_rows += (fend - fstart) as u32;
            stroke_out_rows += (send - sstart) as u32;
        }
    }
    fill_offsets[num_obj] = fill_out_rows;
    stroke_offsets[num_obj] = stroke_out_rows;

    let mut shaded_fills = Array2::<f64>::zeros((fill_out_rows as usize, 4));
    let mut shaded_strokes = Array2::<f64>::zeros((stroke_out_rows as usize, 4));

    let mut fill_write = 0usize;
    let mut stroke_write = 0usize;

    for obj_id in 0..num_obj {
        let fstart = pool.fill_rgba_offsets[obj_id] as usize;
        let fend = pool.fill_rgba_offsets[obj_id + 1] as usize;
        let sstart = pool.stroke_rgba_offsets[obj_id] as usize;
        let send = pool.stroke_rgba_offsets[obj_id + 1] as usize;
        let pstart = pool.point_offsets[obj_id] as usize;
        let pend = pool.point_offsets[obj_id + 1] as usize;

        if pool.shade_in_3d[obj_id] && pend > pstart {
            let pts = &pool.points[pstart..pend];

            let start_normal = start_corner_unit_normal(pts);
            let end_normal = end_corner_unit_normal(pts);
            let start_point = pts[start_corner_index(pts.len())];
            let end_point = pts[end_corner_index(pts.len())];

            // Shade fills: always exactly 2 rows
            let fcount = fend - fstart;
            if fcount == 0 {
                fill_write += 2;
            } else {
                // Python: if len(rgbas) < 2: shaded_rgbas = rgbas.repeat(2, axis=0)
                //         else: shaded_rgbas = np.array(rgbas[:2])
                let row0 = &pool.fill_rgbas[fstart];
                let row1 = if fcount >= 2 { &pool.fill_rgbas[fstart + 1] } else { row0 };

                let shaded0 = shade_rgb(&[row0[0], row0[1], row0[2]], &start_normal, &start_point, &light_pos);
                shaded_fills[[fill_write, 0]] = shaded0[0];
                shaded_fills[[fill_write, 1]] = shaded0[1];
                shaded_fills[[fill_write, 2]] = shaded0[2];
                shaded_fills[[fill_write, 3]] = row0[3];
                fill_write += 1;

                let shaded1 = shade_rgb(&[row1[0], row1[1], row1[2]], &end_normal, &end_point, &light_pos);
                shaded_fills[[fill_write, 0]] = shaded1[0];
                shaded_fills[[fill_write, 1]] = shaded1[1];
                shaded_fills[[fill_write, 2]] = shaded1[2];
                shaded_fills[[fill_write, 3]] = row1[3];
                fill_write += 1;
            }

            // Shade strokes: always exactly 2 rows
            let scount = send - sstart;
            if scount == 0 {
                stroke_write += 2;
            } else {
                let row0 = &pool.stroke_rgbas[sstart];
                let row1 = if scount >= 2 { &pool.stroke_rgbas[sstart + 1] } else { row0 };

                let shaded0 = shade_rgb(&[row0[0], row0[1], row0[2]], &start_normal, &start_point, &light_pos);
                shaded_strokes[[stroke_write, 0]] = shaded0[0];
                shaded_strokes[[stroke_write, 1]] = shaded0[1];
                shaded_strokes[[stroke_write, 2]] = shaded0[2];
                shaded_strokes[[stroke_write, 3]] = row0[3];
                stroke_write += 1;

                let shaded1 = shade_rgb(&[row1[0], row1[1], row1[2]], &end_normal, &end_point, &light_pos);
                shaded_strokes[[stroke_write, 0]] = shaded1[0];
                shaded_strokes[[stroke_write, 1]] = shaded1[1];
                shaded_strokes[[stroke_write, 2]] = shaded1[2];
                shaded_strokes[[stroke_write, 3]] = row1[3];
                stroke_write += 1;
            }
        } else {
            // Not shaded: copy through unchanged
            for i in fstart..fend {
                let r = &pool.fill_rgbas[i];
                shaded_fills[[fill_write, 0]] = r[0];
                shaded_fills[[fill_write, 1]] = r[1];
                shaded_fills[[fill_write, 2]] = r[2];
                shaded_fills[[fill_write, 3]] = r[3];
                fill_write += 1;
            }
            for i in sstart..send {
                let r = &pool.stroke_rgbas[i];
                shaded_strokes[[stroke_write, 0]] = r[0];
                shaded_strokes[[stroke_write, 1]] = r[1];
                shaded_strokes[[stroke_write, 2]] = r[2];
                shaded_strokes[[stroke_write, 3]] = r[3];
                stroke_write += 1;
            }
        }
    }

    Ok((
        shaded_fills.into_pyarray_bound(py),
        shaded_strokes.into_pyarray_bound(py),
        numpy::ndarray::Array1::from_vec(fill_offsets).into_pyarray_bound(py),
        numpy::ndarray::Array1::from_vec(stroke_offsets).into_pyarray_bound(py),
    ))
}
