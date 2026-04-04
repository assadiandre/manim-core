use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Contiguous pool storing ALL VMobject data for a scene.
/// Each VMobject gets a pool_id (u32 index) linking it to its slot.
#[pyclass]
pub struct MeshPool {
    // Geometry: all points in one flat buffer, offset table per object
    pub(crate) points: Vec<[f64; 3]>,
    pub(crate) point_offsets: Vec<u32>, // object i owns points[offsets[i]..offsets[i+1]]

    // Colors: same offset pattern
    pub(crate) fill_rgbas: Vec<[f64; 4]>,
    pub(crate) fill_rgba_offsets: Vec<u32>,
    pub(crate) stroke_rgbas: Vec<[f64; 4]>,
    pub(crate) stroke_rgba_offsets: Vec<u32>,
    pub(crate) bg_stroke_rgbas: Vec<[f64; 4]>,
    pub(crate) bg_stroke_rgba_offsets: Vec<u32>,

    // Per-object scalars (parallel arrays, length = num_objects)
    pub(crate) stroke_widths: Vec<f64>,
    pub(crate) bg_stroke_widths: Vec<f64>,
    pub(crate) sheen_factors: Vec<f64>,
    pub(crate) sheen_directions: Vec<[f64; 3]>,
    pub(crate) shade_in_3d: Vec<bool>,

    // Tree structure (replaces recursive get_family)
    pub(crate) parent_ids: Vec<i32>, // -1 for root
    pub(crate) children: Vec<Vec<u32>>,
    pub(crate) family_order: Vec<u32>, // pre-computed DFS traversal
    pub(crate) family_order_dirty: bool,

    pub(crate) num_objects: u32,

    // Slot reuse: freed IDs available for reallocation
    pub(crate) free_ids: Vec<u32>,
}

#[pymethods]
impl MeshPool {
    #[new]
    pub fn new() -> Self {
        MeshPool {
            points: Vec::new(),
            point_offsets: vec![0],
            fill_rgbas: Vec::new(),
            fill_rgba_offsets: vec![0],
            stroke_rgbas: Vec::new(),
            stroke_rgba_offsets: vec![0],
            bg_stroke_rgbas: Vec::new(),
            bg_stroke_rgba_offsets: vec![0],
            stroke_widths: Vec::new(),
            bg_stroke_widths: Vec::new(),
            sheen_factors: Vec::new(),
            sheen_directions: Vec::new(),
            shade_in_3d: Vec::new(),
            parent_ids: Vec::new(),
            children: Vec::new(),
            family_order: Vec::new(),
            family_order_dirty: true,
            num_objects: 0,
            free_ids: Vec::new(),
        }
    }

    /// Register a new VMobject, returns its pool_id.
    /// points: (N, 3), fill_rgbas: (M, 4), stroke_rgbas: (K, 4), etc.
    pub fn register(
        &mut self,
        _py: Python<'_>,
        points: PyReadonlyArray2<f64>,
        fill_rgbas: PyReadonlyArray2<f64>,
        stroke_rgbas: PyReadonlyArray2<f64>,
        bg_stroke_rgbas: PyReadonlyArray2<f64>,
        stroke_width: f64,
        bg_stroke_width: f64,
        sheen_factor: f64,
        sheen_direction: PyReadonlyArray1<f64>,
        shade_in_3d: bool,
        parent_id: i32,
    ) -> PyResult<u32> {
        let pool_id = if let Some(id) = self.free_ids.pop() {
            id
        } else {
            let id = self.num_objects;
            self.num_objects += 1;
            // Extend parallel arrays
            self.stroke_widths.push(0.0);
            self.bg_stroke_widths.push(0.0);
            self.sheen_factors.push(0.0);
            self.sheen_directions.push([0.0; 3]);
            self.shade_in_3d.push(false);
            self.parent_ids.push(-1);
            self.children.push(Vec::new());
            // Add sentinel offsets (will be set below)
            self.point_offsets.push(*self.point_offsets.last().unwrap());
            self.fill_rgba_offsets.push(*self.fill_rgba_offsets.last().unwrap());
            self.stroke_rgba_offsets.push(*self.stroke_rgba_offsets.last().unwrap());
            self.bg_stroke_rgba_offsets.push(*self.bg_stroke_rgba_offsets.last().unwrap());
            id
        };
        let idx = pool_id as usize;

        // Store points
        let pts = points.as_array();
        let _start = self.point_offsets[idx] as usize;
        let pts_vec: Vec<[f64; 3]> = pts
            .rows()
            .into_iter()
            .map(|r| [r[0], r[1], r[2]])
            .collect();
        // For new objects appended at end, just extend
        if idx == (self.num_objects - 1) as usize {
            self.points.extend_from_slice(&pts_vec);
            *self.point_offsets.last_mut().unwrap() = self.points.len() as u32;
        } else {
            // Reused slot: splice in (complex case; for now just append and track)
            self.points.extend_from_slice(&pts_vec);
            self.point_offsets[idx + 1] = self.points.len() as u32;
        }

        // Store fill_rgbas
        let fill = fill_rgbas.as_array();
        let fill_vec: Vec<[f64; 4]> = fill
            .rows()
            .into_iter()
            .map(|r| [r[0], r[1], r[2], r[3]])
            .collect();
        self.fill_rgbas.extend_from_slice(&fill_vec);
        self.fill_rgba_offsets[idx + 1] = self.fill_rgbas.len() as u32;

        // Store stroke_rgbas
        let stroke = stroke_rgbas.as_array();
        let stroke_vec: Vec<[f64; 4]> = stroke
            .rows()
            .into_iter()
            .map(|r| [r[0], r[1], r[2], r[3]])
            .collect();
        self.stroke_rgbas.extend_from_slice(&stroke_vec);
        self.stroke_rgba_offsets[idx + 1] = self.stroke_rgbas.len() as u32;

        // Store bg_stroke_rgbas
        let bg = bg_stroke_rgbas.as_array();
        let bg_vec: Vec<[f64; 4]> = bg
            .rows()
            .into_iter()
            .map(|r| [r[0], r[1], r[2], r[3]])
            .collect();
        self.bg_stroke_rgbas.extend_from_slice(&bg_vec);
        self.bg_stroke_rgba_offsets[idx + 1] = self.bg_stroke_rgbas.len() as u32;

        // Store scalars
        self.stroke_widths[idx] = stroke_width;
        self.bg_stroke_widths[idx] = bg_stroke_width;
        self.sheen_factors[idx] = sheen_factor;
        let sd = sheen_direction.as_array();
        self.sheen_directions[idx] = [sd[0], sd[1], sd[2]];
        self.shade_in_3d[idx] = shade_in_3d;

        // Tree
        self.parent_ids[idx] = parent_id;
        if parent_id >= 0 {
            self.children[parent_id as usize].push(pool_id);
        }
        self.family_order_dirty = true;

        Ok(pool_id)
    }

    /// Update points for an existing object.
    pub fn update_points(
        &mut self,
        pool_id: u32,
        points: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let idx = pool_id as usize;
        let start = self.point_offsets[idx] as usize;
        let end = self.point_offsets[idx + 1] as usize;
        let pts = points.as_array();

        if pts.nrows() != (end - start) {
            return Err(PyValueError::new_err(
                "Point count changed; use update_points_resize for variable-length updates",
            ));
        }

        for (i, row) in pts.rows().into_iter().enumerate() {
            self.points[start + i] = [row[0], row[1], row[2]];
        }
        Ok(())
    }

    /// Update fill RGBA colors for an existing object.
    pub fn update_fill_rgbas(
        &mut self,
        pool_id: u32,
        fill_rgbas: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let idx = pool_id as usize;
        let start = self.fill_rgba_offsets[idx] as usize;
        let end = self.fill_rgba_offsets[idx + 1] as usize;
        let arr = fill_rgbas.as_array();
        if arr.nrows() != (end - start) {
            return Err(PyValueError::new_err("Fill RGBA count changed"));
        }
        for (i, row) in arr.rows().into_iter().enumerate() {
            self.fill_rgbas[start + i] = [row[0], row[1], row[2], row[3]];
        }
        Ok(())
    }

    /// Update stroke RGBA colors for an existing object.
    pub fn update_stroke_rgbas(
        &mut self,
        pool_id: u32,
        stroke_rgbas: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let idx = pool_id as usize;
        let start = self.stroke_rgba_offsets[idx] as usize;
        let end = self.stroke_rgba_offsets[idx + 1] as usize;
        let arr = stroke_rgbas.as_array();
        if arr.nrows() != (end - start) {
            return Err(PyValueError::new_err("Stroke RGBA count changed"));
        }
        for (i, row) in arr.rows().into_iter().enumerate() {
            self.stroke_rgbas[start + i] = [row[0], row[1], row[2], row[3]];
        }
        Ok(())
    }

    /// Update scalar attributes.
    pub fn update_scalars(
        &mut self,
        pool_id: u32,
        stroke_width: f64,
        bg_stroke_width: f64,
        sheen_factor: f64,
        sheen_direction: PyReadonlyArray1<f64>,
        shade_in_3d: bool,
    ) -> PyResult<()> {
        let idx = pool_id as usize;
        self.stroke_widths[idx] = stroke_width;
        self.bg_stroke_widths[idx] = bg_stroke_width;
        self.sheen_factors[idx] = sheen_factor;
        let sd = sheen_direction.as_array();
        self.sheen_directions[idx] = [sd[0], sd[1], sd[2]];
        self.shade_in_3d[idx] = shade_in_3d;
        Ok(())
    }

    /// Get points for a specific object as numpy array.
    pub fn get_points<'py>(&self, py: Python<'py>, pool_id: u32) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let idx = pool_id as usize;
        let start = self.point_offsets[idx] as usize;
        let end = self.point_offsets[idx + 1] as usize;
        let npts = end - start;
        let mut arr = Array2::<f64>::zeros((npts, 3));
        for i in 0..npts {
            let p = &self.points[start + i];
            arr[[i, 0]] = p[0];
            arr[[i, 1]] = p[1];
            arr[[i, 2]] = p[2];
        }
        Ok(arr.into_pyarray_bound(py))
    }

    /// Get all points as a single (total_points, 3) numpy array.
    pub fn get_all_points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.points.len();
        let mut arr = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            arr[[i, 0]] = self.points[i][0];
            arr[[i, 1]] = self.points[i][1];
            arr[[i, 2]] = self.points[i][2];
        }
        arr.into_pyarray_bound(py)
    }

    /// Set parent-child relationship.
    pub fn set_parent(&mut self, child_id: u32, parent_id: i32) -> PyResult<()> {
        let cidx = child_id as usize;
        let old_parent = self.parent_ids[cidx];
        if old_parent >= 0 {
            self.children[old_parent as usize].retain(|&x| x != child_id);
        }
        self.parent_ids[cidx] = parent_id;
        if parent_id >= 0 {
            self.children[parent_id as usize].push(child_id);
        }
        self.family_order_dirty = true;
        Ok(())
    }

    /// Number of registered objects.
    pub fn len(&self) -> u32 {
        self.num_objects
    }

    /// Total number of points across all objects.
    pub fn total_points(&self) -> usize {
        self.points.len()
    }

    /// Get point range for an object: (start, end).
    pub fn point_range(&self, pool_id: u32) -> (u32, u32) {
        let idx = pool_id as usize;
        (self.point_offsets[idx], self.point_offsets[idx + 1])
    }

    /// Mark the family order as dirty (needs recomputation).
    pub fn mark_family_dirty(&mut self) {
        self.family_order_dirty = true;
    }
}
