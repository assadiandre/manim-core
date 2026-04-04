use pyo3::prelude::*;

use crate::mesh_pool::MeshPool;

/// Clone the entire pool (deep copy of all data).
/// Replaces copy.deepcopy — 835K recursive calls → one ~250KB memcpy.
#[pyfunction]
pub fn clone_pool(pool: &MeshPool) -> MeshPool {
    MeshPool {
        points: pool.points.clone(),
        point_offsets: pool.point_offsets.clone(),
        fill_rgbas: pool.fill_rgbas.clone(),
        fill_rgba_offsets: pool.fill_rgba_offsets.clone(),
        stroke_rgbas: pool.stroke_rgbas.clone(),
        stroke_rgba_offsets: pool.stroke_rgba_offsets.clone(),
        bg_stroke_rgbas: pool.bg_stroke_rgbas.clone(),
        bg_stroke_rgba_offsets: pool.bg_stroke_rgba_offsets.clone(),
        stroke_widths: pool.stroke_widths.clone(),
        bg_stroke_widths: pool.bg_stroke_widths.clone(),
        sheen_factors: pool.sheen_factors.clone(),
        sheen_directions: pool.sheen_directions.clone(),
        shade_in_3d: pool.shade_in_3d.clone(),
        parent_ids: pool.parent_ids.clone(),
        children: pool.children.clone(),
        family_order: pool.family_order.clone(),
        family_order_dirty: pool.family_order_dirty,
        num_objects: pool.num_objects,
        free_ids: pool.free_ids.clone(),
    }
}

/// Interpolate all data in `target` between `start` and `end` pools at parameter `alpha`.
///
/// path_func_type:
///   0 = linear (alpha used directly)
///   1 = smooth (built-in smooth function: 3t^2 - 2t^3)
///   2 = rate_func already applied (alpha is the final parameter)
#[pyfunction]
pub fn interpolate_pools(
    target: &mut MeshPool,
    start: &MeshPool,
    end: &MeshPool,
    alpha: f64,
    path_func_type: u8,
) -> PyResult<()> {
    let t = match path_func_type {
        1 => smooth(alpha),
        _ => alpha,
    };

    let n_pts = target.points.len().min(start.points.len()).min(end.points.len());
    for i in 0..n_pts {
        for j in 0..3 {
            target.points[i][j] = start.points[i][j] + t * (end.points[i][j] - start.points[i][j]);
        }
    }

    let n_fill = target.fill_rgbas.len().min(start.fill_rgbas.len()).min(end.fill_rgbas.len());
    for i in 0..n_fill {
        for j in 0..4 {
            target.fill_rgbas[i][j] = start.fill_rgbas[i][j] + t * (end.fill_rgbas[i][j] - start.fill_rgbas[i][j]);
        }
    }

    let n_stroke = target.stroke_rgbas.len().min(start.stroke_rgbas.len()).min(end.stroke_rgbas.len());
    for i in 0..n_stroke {
        for j in 0..4 {
            target.stroke_rgbas[i][j] = start.stroke_rgbas[i][j] + t * (end.stroke_rgbas[i][j] - start.stroke_rgbas[i][j]);
        }
    }

    let n_bg = target.bg_stroke_rgbas.len().min(start.bg_stroke_rgbas.len()).min(end.bg_stroke_rgbas.len());
    for i in 0..n_bg {
        for j in 0..4 {
            target.bg_stroke_rgbas[i][j] = start.bg_stroke_rgbas[i][j] + t * (end.bg_stroke_rgbas[i][j] - start.bg_stroke_rgbas[i][j]);
        }
    }

    let n_obj = target.stroke_widths.len().min(start.stroke_widths.len()).min(end.stroke_widths.len());
    for i in 0..n_obj {
        target.stroke_widths[i] = start.stroke_widths[i] + t * (end.stroke_widths[i] - start.stroke_widths[i]);
        target.bg_stroke_widths[i] = start.bg_stroke_widths[i] + t * (end.bg_stroke_widths[i] - start.bg_stroke_widths[i]);
        target.sheen_factors[i] = start.sheen_factors[i] + t * (end.sheen_factors[i] - start.sheen_factors[i]);
        for j in 0..3 {
            target.sheen_directions[i][j] = start.sheen_directions[i][j] + t * (end.sheen_directions[i][j] - start.sheen_directions[i][j]);
        }
    }

    Ok(())
}

/// Interpolate only specific objects (by ID) between start and end pools.
#[pyfunction]
pub fn interpolate_objects(
    target: &mut MeshPool,
    start: &MeshPool,
    end: &MeshPool,
    object_ids: Vec<u32>,
    alpha: f64,
    path_func_type: u8,
) -> PyResult<()> {
    let t = match path_func_type {
        1 => smooth(alpha),
        _ => alpha,
    };

    for &obj_id in &object_ids {
        let idx = obj_id as usize;

        let pstart = target.point_offsets[idx] as usize;
        let pend = target.point_offsets[idx + 1] as usize;
        for i in pstart..pend {
            for j in 0..3 {
                target.points[i][j] = start.points[i][j] + t * (end.points[i][j] - start.points[i][j]);
            }
        }

        let fstart = target.fill_rgba_offsets[idx] as usize;
        let fend = target.fill_rgba_offsets[idx + 1] as usize;
        for i in fstart..fend {
            for j in 0..4 {
                target.fill_rgbas[i][j] = start.fill_rgbas[i][j] + t * (end.fill_rgbas[i][j] - start.fill_rgbas[i][j]);
            }
        }

        let sstart = target.stroke_rgba_offsets[idx] as usize;
        let send = target.stroke_rgba_offsets[idx + 1] as usize;
        for i in sstart..send {
            for j in 0..4 {
                target.stroke_rgbas[i][j] = start.stroke_rgbas[i][j] + t * (end.stroke_rgbas[i][j] - start.stroke_rgbas[i][j]);
            }
        }

        target.stroke_widths[idx] = start.stroke_widths[idx] + t * (end.stroke_widths[idx] - start.stroke_widths[idx]);
        target.bg_stroke_widths[idx] = start.bg_stroke_widths[idx] + t * (end.bg_stroke_widths[idx] - start.bg_stroke_widths[idx]);
        target.sheen_factors[idx] = start.sheen_factors[idx] + t * (end.sheen_factors[idx] - start.sheen_factors[idx]);
        for j in 0..3 {
            target.sheen_directions[idx][j] = start.sheen_directions[idx][j] + t * (end.sheen_directions[idx][j] - start.sheen_directions[idx][j]);
        }
    }

    Ok(())
}

#[inline]
fn smooth(t: f64) -> f64 {
    let s = t.clamp(0.0, 1.0);
    3.0 * s * s - 2.0 * s * s * s
}
