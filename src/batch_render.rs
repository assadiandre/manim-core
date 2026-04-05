use numpy::{PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use tiny_skia::{
    Color, FillRule, LineCap, LineJoin, Paint, PathBuilder, PixmapMut,
    Stroke, Transform,
};

use crate::mesh_pool::MeshPool;

fn premultiply_in_place(data: &mut [u8]) {
    for px in data.chunks_exact_mut(4) {
        let a = px[3] as u16;
        if a == 0 {
            px[0] = 0;
            px[1] = 0;
            px[2] = 0;
        } else if a < 255 {
            px[0] = ((px[0] as u16 * a + 127) / 255) as u8;
            px[1] = ((px[1] as u16 * a + 127) / 255) as u8;
            px[2] = ((px[2] as u16 * a + 127) / 255) as u8;
        }
    }
}

fn unpremultiply_in_place(data: &mut [u8]) {
    for px in data.chunks_exact_mut(4) {
        let a = px[3] as u16;
        if a > 0 && a < 255 {
            px[0] = ((px[0] as u16 * 255 + a / 2) / a).min(255) as u8;
            px[1] = ((px[1] as u16 * 255 + a / 2) / a).min(255) as u8;
            px[2] = ((px[2] as u16 * 255 + a / 2) / a).min(255) as u8;
        }
    }
}

/// Check if two 2D points are approximately equal using the same algorithm as
/// VMobject.consider_points_equals_2d (np.isclose defaults).
#[inline]
fn points_close_2d(p0: &[f64; 3], p1: &[f64; 3], tolerance: f64) -> bool {
    let rtol = 1.0e-5;
    if (p0[0] - p1[0]).abs() > tolerance + rtol * p1[0].abs() {
        return false;
    }
    (p0[1] - p1[1]).abs() <= tolerance + rtol * p1[1].abs()
}

/// Build a tiny_skia path from object points using the tolerance-based
/// subpath splitting algorithm that matches Python's gen_subpaths_from_points_2d.
fn build_path(points: &[[f64; 3]], tolerance: f64) -> Option<tiny_skia::Path> {
    let npts = points.len();
    if npts < 4 {
        return None;
    }

    let nppcc = 4usize; // n_points_per_cubic_curve

    // Compute split indices: where points[n-1] != points[n] at nppcc boundaries
    // This mirrors _gen_subpaths_from_points with the 2D filter
    let mut split_indices = vec![0usize];
    let mut i = nppcc;
    while i < npts {
        if !points_close_2d(&points[i - 1], &points[i], tolerance) {
            split_indices.push(i);
        }
        i += nppcc;
    }
    split_indices.push(npts);

    let mut pb = PathBuilder::new();
    let mut has_geometry = false;

    for w in split_indices.windows(2) {
        let sp_start = w[0];
        let sp_end = w[1];
        let sp_len = sp_end - sp_start;
        if sp_len < nppcc {
            continue;
        }

        let subpath = &points[sp_start..sp_end];
        pb.move_to(subpath[0][0] as f32, subpath[0][1] as f32);
        has_geometry = true;

        // Generate cubic bezier tuples: every nppcc points
        let usable = sp_len - (sp_len % nppcc);
        let mut j = 0;
        while j + nppcc <= usable {
            let p1 = &subpath[j + 1];
            let p2 = &subpath[j + 2];
            let p3 = &subpath[j + 3];
            pb.cubic_to(
                p1[0] as f32, p1[1] as f32,
                p2[0] as f32, p2[1] as f32,
                p3[0] as f32, p3[1] as f32,
            );
            j += nppcc;
        }

        // Close if first ~ last
        if points_close_2d(&subpath[0], &subpath[sp_len - 1], tolerance) {
            pb.close();
        }
    }

    if !has_geometry {
        return None;
    }
    pb.finish()
}

/// Clamp a color component to the valid 0.0..=1.0 range for tiny-skia.
/// Shading can produce values > 1.0 for bright-lit surfaces.
#[inline]
fn clamp01(v: f64) -> f32 {
    (v as f32).clamp(0.0, 1.0)
}

/// Create a solid-color Paint. tiny-skia Color::from_rgba takes f32 in 0.0..=1.0.
fn make_solid_paint(r: f64, g: f64, b: f64, a: f64) -> Paint<'static> {
    let mut paint = Paint::default();
    paint.set_color(
        Color::from_rgba(clamp01(r), clamp01(g), clamp01(b), clamp01(a))
            .unwrap_or(Color::TRANSPARENT),
    );
    paint.anti_alias = true;
    paint
}

/// Create a gradient paint for multiple colors.
/// Gradient endpoints are in world coordinates; transform them to pixel space
/// since tiny-skia gradient coordinates are in device (pixel) space, unlike
/// Skia where the canvas transform applies to the shader automatically.
fn make_gradient_paint(
    rgbas: &[[f64; 4]],
    start: (f32, f32),
    end: (f32, f32),
    transform: Transform,
) -> Paint<'static> {
    let mut paint = Paint::default();
    paint.anti_alias = true;

    let n = rgbas.len();
    let mut stops = Vec::with_capacity(n);
    for (i, c) in rgbas.iter().enumerate() {
        let color = Color::from_rgba(clamp01(c[0]), clamp01(c[1]), clamp01(c[2]), clamp01(c[3]))
            .unwrap_or(Color::TRANSPARENT);
        let pos = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.0 };
        stops.push(tiny_skia::GradientStop::new(pos, color));
    }

    // Transform gradient endpoints from world to pixel space manually.
    // tiny-skia shaders operate in device (pixel) coords by default.
    let mut p0 = tiny_skia::Point { x: start.0, y: start.1 };
    let mut p1 = tiny_skia::Point { x: end.0, y: end.1 };
    transform.map_point(&mut p0);
    transform.map_point(&mut p1);

    let shader = tiny_skia::LinearGradient::new(
        p0, p1,
        stops,
        tiny_skia::SpreadMode::Pad,
        Transform::identity(),
    );
    if let Some(s) = shader {
        paint.shader = s;
    } else {
        // Gradient creation failed (e.g. start ≈ end) — fall back to first color
        paint.set_color(
            Color::from_rgba(
                rgbas[0][0] as f32, rgbas[0][1] as f32,
                rgbas[0][2] as f32, rgbas[0][3] as f32,
            ).unwrap_or(Color::TRANSPARENT),
        );
    }
    paint
}

/// Compute 2D gradient endpoints for an object.
/// For 2D: bounding box center +/- dot(bases, sheen_direction)
fn compute_2d_gradient_endpoints(
    points: &[[f64; 3]],
    sheen_direction: &[f64; 3],
) -> ((f32, f32), (f32, f32)) {
    if points.is_empty() {
        return ((0.0, 0.0), (0.0, 0.0));
    }
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;
    for p in points {
        if p[0] < min_x { min_x = p[0]; }
        if p[0] > max_x { max_x = p[0]; }
        if p[1] < min_y { min_y = p[1]; }
        if p[1] > max_y { max_y = p[1]; }
    }
    let cx = (min_x + max_x) * 0.5;
    let cy = (min_y + max_y) * 0.5;
    // bases = half-extents along x, y (matching get_edge_center - center)
    let hx = (max_x - min_x) * 0.5;
    let hy = (max_y - min_y) * 0.5;
    // offset = bases.T @ sheen_direction = [hx*sd[0], hy*sd[1], 0]
    let off_x = hx * sheen_direction[0];
    let off_y = hy * sheen_direction[1];
    (
        ((cx - off_x) as f32, (cy - off_y) as f32),
        ((cx + off_x) as f32, (cy + off_y) as f32),
    )
}

/// Compute 3D gradient endpoints.
/// start = pts[0][:2], end = pts[((npts-1)/6)*3][:2]
fn compute_3d_gradient_endpoints(
    points: &[[f64; 3]],
) -> ((f32, f32), (f32, f32)) {
    if points.is_empty() {
        return ((0.0, 0.0), (0.0, 0.0));
    }
    let start = (points[0][0] as f32, points[0][1] as f32);
    let end_idx = ((points.len() - 1) / 6) * 3;
    let end_idx = end_idx.min(points.len() - 1);
    let end = (points[end_idx][0] as f32, points[end_idx][1] as f32);
    (start, end)
}

fn map_line_join(joint_type: u8) -> LineJoin {
    match joint_type {
        1 => LineJoin::Round,
        2 => LineJoin::Bevel,
        3 => LineJoin::Miter,
        _ => LineJoin::Miter, // AUTO default
    }
}

fn map_line_cap(cap_style: u8) -> LineCap {
    match cap_style {
        1 => LineCap::Round,
        2 => LineCap::Butt,
        3 => LineCap::Square,
        _ => LineCap::Butt, // AUTO default
    }
}

/// Read color slice from pool arrays.
fn read_colors_from_pool(
    pool_colors: &[[f64; 4]],
    offsets: &[u32],
    idx: usize,
) -> Vec<[f64; 4]> {
    let start = offsets[idx] as usize;
    let end = offsets[idx + 1] as usize;
    pool_colors[start..end].to_vec()
}

/// Read colors from shaded numpy arrays.
fn read_shaded_colors(
    shaded: &numpy::ndarray::ArrayView2<f64>,
    offsets: &[u32],
    obj_index: usize,
) -> Vec<[f64; 4]> {
    let start = offsets[obj_index] as usize;
    let end = offsets[obj_index + 1] as usize;
    let mut result = Vec::with_capacity(end - start);
    for i in start..end {
        result.push([shaded[[i, 0]], shaded[[i, 1]], shaded[[i, 2]], shaded[[i, 3]]]);
    }
    result
}

#[pyfunction]
#[pyo3(signature = (
    pixel_data,
    width, height,
    tx, ty, sx, sy,
    pool,
    render_order,
    has_fill,
    has_stroke,
    has_bg_stroke,
    projected_points=None,
    shaded_fills=None,
    shaded_strokes=None,
    shaded_fill_offsets=None,
    shaded_stroke_offsets=None,
    line_width_multiple=0.01,
))]
pub fn batch_render(
    _py: Python<'_>,
    pixel_data: &Bound<'_, PyArray3<u8>>,
    width: u32,
    height: u32,
    tx: f32,
    ty: f32,
    sx: f32,
    sy: f32,
    pool: &MeshPool,
    render_order: PyReadonlyArray1<u32>,
    has_fill: PyReadonlyArray1<bool>,
    has_stroke: PyReadonlyArray1<bool>,
    has_bg_stroke: PyReadonlyArray1<bool>,
    projected_points: Option<PyReadonlyArray2<f64>>,
    shaded_fills: Option<PyReadonlyArray2<f64>>,
    shaded_strokes: Option<PyReadonlyArray2<f64>>,
    shaded_fill_offsets: Option<PyReadonlyArray1<u32>>,
    shaded_stroke_offsets: Option<PyReadonlyArray1<u32>>,
    line_width_multiple: f64,
) -> PyResult<()> {
    let order = render_order.as_array();
    let fill_vis = has_fill.as_array();
    let stroke_vis = has_stroke.as_array();
    let bg_stroke_vis = has_bg_stroke.as_array();

    let proj_view = projected_points.as_ref().map(|p| p.as_array());
    let shaded_fill_view = shaded_fills.as_ref().map(|s| s.as_array());
    let shaded_stroke_view = shaded_strokes.as_ref().map(|s| s.as_array());
    let shaded_fill_off = shaded_fill_offsets.as_ref().map(|o| o.as_array());
    let shaded_stroke_off = shaded_stroke_offsets.as_ref().map(|o| o.as_array());

    // Get mutable pixel slice
    let pixel_slice = unsafe {
        let ptr = pixel_data.data();
        let len = (width as usize) * (height as usize) * 4;
        std::slice::from_raw_parts_mut(ptr as *mut u8, len)
    };

    let expected_size = (width * height * 4) as usize;
    if pixel_slice.len() < expected_size {
        return Err(PyValueError::new_err(format!(
            "pixel_data too small: {} < {}", pixel_slice.len(), expected_size
        )));
    }

    // Premultiply for tiny-skia
    premultiply_in_place(pixel_slice);

    // Create PixmapMut
    let mut pixmap = PixmapMut::from_bytes(pixel_slice, width, height)
        .ok_or_else(|| PyValueError::new_err("Failed to create PixmapMut"))?;

    // Skia canvas applies transforms in reverse: translate then scale means
    // point -> scale -> translate.  In tiny-skia matrix math that is:
    let transform = Transform::from_scale(sx, sy).post_translate(tx, ty);

    for (i, &obj_id) in order.iter().enumerate() {
        let idx = obj_id as usize;
        let vis_fill = fill_vis[i];
        let vis_stroke = stroke_vis[i];
        let vis_bg_stroke = bg_stroke_vis[i];

        if !vis_fill && !vis_stroke && !vis_bg_stroke {
            continue;
        }

        // Get points for this object
        let pt_start = pool.point_offsets[idx] as usize;
        let pt_end = pool.point_offsets[idx + 1] as usize;
        let npts = pt_end - pt_start;
        if npts < 4 {
            continue;
        }

        // Build the point slice - either from projected_points or pool.points
        let obj_points: Vec<[f64; 3]> = if let Some(ref pv) = proj_view {
            (pt_start..pt_end).map(|j| [pv[[j, 0]], pv[[j, 1]], pv[[j, 2]]]).collect()
        } else {
            pool.points[pt_start..pt_end].to_vec()
        };

        let tolerance = pool.tolerances[idx];

        // Build the path
        let path = match build_path(&obj_points, tolerance) {
            Some(p) => p,
            None => continue,
        };

        let shade_3d = pool.shade_in_3d[idx];

        // Draw order: background stroke -> fill -> foreground stroke

        // Background stroke
        if vis_bg_stroke {
            let bg_sw = pool.bg_stroke_widths[idx];
            if bg_sw > 0.0 {
                let bg_colors = read_colors_from_pool(
                    &pool.bg_stroke_rgbas,
                    &pool.bg_stroke_rgba_offsets,
                    idx,
                );
                if !bg_colors.is_empty() {
                    let paint = if bg_colors.len() == 1 {
                        make_solid_paint(bg_colors[0][0], bg_colors[0][1], bg_colors[0][2], bg_colors[0][3])
                    } else {
                        let (start, end) = if shade_3d {
                            compute_3d_gradient_endpoints(&obj_points)
                        } else {
                            compute_2d_gradient_endpoints(&obj_points, &pool.sheen_directions[idx])
                        };
                        make_gradient_paint(&bg_colors, start, end, transform)
                    };
                    let mut stroke = Stroke::default();
                    stroke.width = (bg_sw * line_width_multiple) as f32;
                    stroke.line_join = map_line_join(pool.joint_types[idx]);
                    stroke.line_cap = map_line_cap(pool.cap_styles[idx]);
                    pixmap.stroke_path(&path, &paint, &stroke, transform, None);
                }
            }
        }

        // Fill
        if vis_fill {
            let fill_colors = if let (Some(ref sfv), Some(ref sfo)) = (&shaded_fill_view, &shaded_fill_off) {
                read_shaded_colors(sfv, sfo.as_slice().unwrap(), idx)
            } else {
                read_colors_from_pool(&pool.fill_rgbas, &pool.fill_rgba_offsets, idx)
            };

            if !fill_colors.is_empty() {
                let paint = if fill_colors.len() == 1 {
                    make_solid_paint(fill_colors[0][0], fill_colors[0][1], fill_colors[0][2], fill_colors[0][3])
                } else {
                    let (start, end) = if shade_3d {
                        compute_3d_gradient_endpoints(&obj_points)
                    } else {
                        compute_2d_gradient_endpoints(&obj_points, &pool.sheen_directions[idx])
                    };
                    make_gradient_paint(&fill_colors, start, end, transform)
                };
                pixmap.fill_path(&path, &paint, FillRule::Winding, transform, None);
            }
        }

        // Foreground stroke
        if vis_stroke {
            let sw = pool.stroke_widths[idx];
            if sw > 0.0 {
                let stroke_colors = if let (Some(ref ssv), Some(ref sso)) = (&shaded_stroke_view, &shaded_stroke_off) {
                    read_shaded_colors(ssv, sso.as_slice().unwrap(), idx)
                } else {
                    read_colors_from_pool(&pool.stroke_rgbas, &pool.stroke_rgba_offsets, idx)
                };

                if !stroke_colors.is_empty() {
                    let paint = if stroke_colors.len() == 1 {
                        make_solid_paint(stroke_colors[0][0], stroke_colors[0][1], stroke_colors[0][2], stroke_colors[0][3])
                    } else {
                        let (start, end) = if shade_3d {
                            compute_3d_gradient_endpoints(&obj_points)
                        } else {
                            compute_2d_gradient_endpoints(&obj_points, &pool.sheen_directions[idx])
                        };
                        make_gradient_paint(&stroke_colors, start, end, transform)
                    };
                    let mut stroke = Stroke::default();
                    stroke.width = (sw * line_width_multiple) as f32;
                    stroke.line_join = map_line_join(pool.joint_types[idx]);
                    stroke.line_cap = map_line_cap(pool.cap_styles[idx]);
                    pixmap.stroke_path(&path, &paint, &stroke, transform, None);
                }
            }
        }
    }

    // Unpremultiply back for downstream pipeline
    unpremultiply_in_place(pixel_slice);

    Ok(())
}
