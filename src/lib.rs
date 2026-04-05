use pyo3::prelude::*;

mod mesh_pool;
mod projection;
mod shading;
mod tree;
mod hashing;
mod interpolation;
mod rendering;
mod batch_render;

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core data structure
    m.add_class::<mesh_pool::MeshPool>()?;

    // Phase 1: Projection + Shading + Z-Sort
    m.add_function(wrap_pyfunction!(projection::project_all_points, m)?)?;
    m.add_function(wrap_pyfunction!(projection::project_points_for_objects, m)?)?;
    m.add_function(wrap_pyfunction!(shading::shade_all_objects, m)?)?;
    m.add_function(wrap_pyfunction!(tree::recompute_family_order, m)?)?;
    m.add_function(wrap_pyfunction!(tree::get_family_order, m)?)?;
    m.add_function(wrap_pyfunction!(tree::z_sort, m)?)?;
    m.add_function(wrap_pyfunction!(tree::get_family_for, m)?)?;

    // Phase 2: Hashing
    m.add_function(wrap_pyfunction!(hashing::hash_pool_state, m)?)?;
    m.add_function(wrap_pyfunction!(hashing::hash_objects, m)?)?;

    // Phase 3: Interpolation
    m.add_function(wrap_pyfunction!(interpolation::clone_pool, m)?)?;
    m.add_function(wrap_pyfunction!(interpolation::interpolate_pools, m)?)?;
    m.add_function(wrap_pyfunction!(interpolation::interpolate_objects, m)?)?;
    m.add_function(wrap_pyfunction!(interpolation::interpolate_object_attrs, m)?)?;

    // Phase 4: Rendering
    m.add_function(wrap_pyfunction!(rendering::prepare_render_data, m)?)?;
    m.add_function(wrap_pyfunction!(rendering::compute_visibility, m)?)?;

    // Phase 5: Batch Render (tiny-skia)
    m.add_function(wrap_pyfunction!(batch_render::batch_render, m)?)?;

    Ok(())
}
