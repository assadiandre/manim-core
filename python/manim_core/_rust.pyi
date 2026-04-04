"""Type stubs for manim_core._rust (PyO3 extension)."""

import numpy as np
import numpy.typing as npt

class MeshPool:
    def __init__(self) -> None: ...
    def register(
        self,
        points: npt.NDArray[np.float64],
        fill_rgbas: npt.NDArray[np.float64],
        stroke_rgbas: npt.NDArray[np.float64],
        bg_stroke_rgbas: npt.NDArray[np.float64],
        stroke_width: float,
        bg_stroke_width: float,
        sheen_factor: float,
        sheen_direction: npt.NDArray[np.float64],
        shade_in_3d: bool,
        parent_id: int,
    ) -> int: ...
    def update_points(self, pool_id: int, points: npt.NDArray[np.float64]) -> None: ...
    def update_fill_rgbas(self, pool_id: int, fill_rgbas: npt.NDArray[np.float64]) -> None: ...
    def update_stroke_rgbas(self, pool_id: int, stroke_rgbas: npt.NDArray[np.float64]) -> None: ...
    def update_scalars(
        self,
        pool_id: int,
        stroke_width: float,
        bg_stroke_width: float,
        sheen_factor: float,
        sheen_direction: npt.NDArray[np.float64],
        shade_in_3d: bool,
    ) -> None: ...
    def get_points(self, pool_id: int) -> npt.NDArray[np.float64]: ...
    def get_all_points(self) -> npt.NDArray[np.float64]: ...
    def set_parent(self, child_id: int, parent_id: int) -> None: ...
    def len(self) -> int: ...
    def total_points(self) -> int: ...
    def point_range(self, pool_id: int) -> tuple[int, int]: ...

def project_all_points(
    pool: MeshPool,
    frame_center: npt.NDArray[np.float64],
    rotation_matrix: npt.NDArray[np.float64],
    focal_distance: float,
    zoom: float,
) -> npt.NDArray[np.float64]: ...

def project_points_for_objects(
    pool: MeshPool,
    object_ids: npt.NDArray[np.uint32],
    frame_center: npt.NDArray[np.float64],
    rotation_matrix: npt.NDArray[np.float64],
    focal_distance: float,
    zoom: float,
) -> npt.NDArray[np.float64]: ...

def shade_all_objects(
    pool: MeshPool,
    light_source_position: npt.NDArray[np.float64],
    reflectiveness: float,
    shadow: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

def recompute_family_order(pool: MeshPool) -> None: ...

def get_family_order(pool: MeshPool) -> npt.NDArray[np.uint32]: ...

def z_sort(
    pool: MeshPool,
    rotation_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.uint32]: ...

def get_family_for(pool: MeshPool, obj_id: int) -> npt.NDArray[np.uint32]: ...

def hash_pool_state(pool: MeshPool) -> int: ...

def hash_objects(pool: MeshPool, object_ids: list[int]) -> int: ...

def clone_pool(pool: MeshPool) -> MeshPool: ...

def interpolate_pools(
    target: MeshPool,
    start: MeshPool,
    end: MeshPool,
    alpha: float,
    path_func_type: int,
) -> None: ...

def interpolate_objects(
    target: MeshPool,
    start: MeshPool,
    end: MeshPool,
    object_ids: list[int],
    alpha: float,
    path_func_type: int,
) -> None: ...

def prepare_render_data(
    pool: MeshPool,
    render_order: npt.NDArray[np.uint32],
) -> dict: ...

def compute_visibility(
    pool: MeshPool,
    object_ids: npt.NDArray[np.uint32],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_], npt.NDArray[np.bool_]]: ...
