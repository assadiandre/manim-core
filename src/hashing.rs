use ahash::AHasher;
use pyo3::prelude::*;
use std::hash::Hasher;

use crate::mesh_pool::MeshPool;

/// Hash the entire pool state to a u64, replacing JSON serialization.
///
/// Replaces hashing.py:333 get_hash_from_play_call() for mobject state.
/// Instead of 109K _CustomEncoder.default() calls generating ~1MB JSON per play(),
/// this hashes the contiguous memory directly in <1ms.
///
/// The hash covers:
///   - All points
///   - All fill/stroke/bg_stroke RGBA values
///   - All scalar attributes
///   - Tree structure (parent_ids)
#[pyfunction]
pub fn hash_pool_state(pool: &MeshPool) -> u64 {
    let mut hasher = AHasher::default();

    // Hash points
    for p in &pool.points {
        hasher.write(&p[0].to_le_bytes());
        hasher.write(&p[1].to_le_bytes());
        hasher.write(&p[2].to_le_bytes());
    }

    // Hash point offsets (encodes object boundaries)
    for &off in &pool.point_offsets {
        hasher.write_u32(off);
    }

    // Hash fill colors
    for c in &pool.fill_rgbas {
        hasher.write(&c[0].to_le_bytes());
        hasher.write(&c[1].to_le_bytes());
        hasher.write(&c[2].to_le_bytes());
        hasher.write(&c[3].to_le_bytes());
    }
    for &off in &pool.fill_rgba_offsets {
        hasher.write_u32(off);
    }

    // Hash stroke colors
    for c in &pool.stroke_rgbas {
        hasher.write(&c[0].to_le_bytes());
        hasher.write(&c[1].to_le_bytes());
        hasher.write(&c[2].to_le_bytes());
        hasher.write(&c[3].to_le_bytes());
    }
    for &off in &pool.stroke_rgba_offsets {
        hasher.write_u32(off);
    }

    // Hash bg stroke colors
    for c in &pool.bg_stroke_rgbas {
        hasher.write(&c[0].to_le_bytes());
        hasher.write(&c[1].to_le_bytes());
        hasher.write(&c[2].to_le_bytes());
        hasher.write(&c[3].to_le_bytes());
    }
    for &off in &pool.bg_stroke_rgba_offsets {
        hasher.write_u32(off);
    }

    // Hash scalars
    for &sw in &pool.stroke_widths {
        hasher.write(&sw.to_le_bytes());
    }
    for &bsw in &pool.bg_stroke_widths {
        hasher.write(&bsw.to_le_bytes());
    }
    for &sf in &pool.sheen_factors {
        hasher.write(&sf.to_le_bytes());
    }
    for sd in &pool.sheen_directions {
        hasher.write(&sd[0].to_le_bytes());
        hasher.write(&sd[1].to_le_bytes());
        hasher.write(&sd[2].to_le_bytes());
    }
    for &s3d in &pool.shade_in_3d {
        hasher.write_u8(s3d as u8);
    }

    // Hash tree structure
    for &pid in &pool.parent_ids {
        hasher.write_i32(pid);
    }

    hasher.finish()
}

/// Hash a subset of objects (by their IDs) — useful for partial scene hashing.
#[pyfunction]
pub fn hash_objects(pool: &MeshPool, object_ids: Vec<u32>) -> u64 {
    let mut hasher = AHasher::default();

    for &obj_id in &object_ids {
        let idx = obj_id as usize;

        // Points
        let pstart = pool.point_offsets[idx] as usize;
        let pend = pool.point_offsets[idx + 1] as usize;
        hasher.write_u32((pend - pstart) as u32);
        for i in pstart..pend {
            let p = &pool.points[i];
            hasher.write(&p[0].to_le_bytes());
            hasher.write(&p[1].to_le_bytes());
            hasher.write(&p[2].to_le_bytes());
        }

        // Fill
        let fstart = pool.fill_rgba_offsets[idx] as usize;
        let fend = pool.fill_rgba_offsets[idx + 1] as usize;
        for i in fstart..fend {
            let c = &pool.fill_rgbas[i];
            hasher.write(&c[0].to_le_bytes());
            hasher.write(&c[1].to_le_bytes());
            hasher.write(&c[2].to_le_bytes());
            hasher.write(&c[3].to_le_bytes());
        }

        // Stroke
        let sstart = pool.stroke_rgba_offsets[idx] as usize;
        let send = pool.stroke_rgba_offsets[idx + 1] as usize;
        for i in sstart..send {
            let c = &pool.stroke_rgbas[i];
            hasher.write(&c[0].to_le_bytes());
            hasher.write(&c[1].to_le_bytes());
            hasher.write(&c[2].to_le_bytes());
            hasher.write(&c[3].to_le_bytes());
        }

        // Scalars
        hasher.write(&pool.stroke_widths[idx].to_le_bytes());
        hasher.write(&pool.bg_stroke_widths[idx].to_le_bytes());
        hasher.write(&pool.sheen_factors[idx].to_le_bytes());
        hasher.write(&pool.sheen_directions[idx][0].to_le_bytes());
        hasher.write(&pool.sheen_directions[idx][1].to_le_bytes());
        hasher.write(&pool.sheen_directions[idx][2].to_le_bytes());
        hasher.write_u8(pool.shade_in_3d[idx] as u8);
    }

    hasher.finish()
}
