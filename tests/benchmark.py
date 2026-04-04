"""
Benchmark: Rust manim_core vs pure Python/numpy for each hot path.

Run: python tests/benchmark.py
"""
import time
import numpy as np

from manim_core._rust import (
    MeshPool,
    project_all_points,
    shade_all_objects,
    z_sort,
    get_family_order,
    hash_pool_state,
    clone_pool,
    interpolate_pools,
)


def build_sphere_pool(n_faces=1024, pts_per_face=8):
    """Simulate a 1024-face Sphere: n_faces objects, each with pts_per_face points."""
    pool = MeshPool()
    rng = np.random.default_rng(42)

    for i in range(n_faces):
        # Random points on sphere surface
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        center = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ])
        pts = center + rng.normal(0, 0.05, (pts_per_face, 3))

        fill = rng.uniform(0, 1, (1, 4))
        fill[0, 3] = 1.0
        stroke = rng.uniform(0, 1, (1, 4))
        stroke[0, 3] = 1.0

        pool.register(
            np.ascontiguousarray(pts, dtype=np.float64),
            np.ascontiguousarray(fill, dtype=np.float64),
            np.ascontiguousarray(stroke, dtype=np.float64),
            np.zeros((0, 4), dtype=np.float64),
            1.0, 0.0, 0.0,
            np.array([1, 0, 0], dtype=np.float64),
            True, -1,
        )
    return pool


def bench(label, fn, n_iter=100):
    """Run fn n_iter times and report average time."""
    # Warmup
    fn()
    fn()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    elapsed = time.perf_counter() - t0

    avg_ms = (elapsed / n_iter) * 1000
    print(f"  {label}: {avg_ms:.3f} ms/call ({n_iter} iterations, {elapsed:.3f}s total)")
    return avg_ms


def python_project(all_points, fc, rot, fd, zoom):
    """Reference Python projection."""
    pts = all_points - fc
    pts = np.dot(pts, rot.T)
    zs = pts[:, 2]
    for i in (0, 1):
        factor = fd / (fd - zs)
        factor[(fd - zs) < 0] = 1e6
        pts[:, i] *= factor * zoom
    return pts


def main():
    n_faces = 1024
    pts_per_face = 8
    print(f"Building pool: {n_faces} faces, {pts_per_face} pts/face = {n_faces * pts_per_face} total points")

    pool = build_sphere_pool(n_faces, pts_per_face)
    fc = np.zeros(3, dtype=np.float64)
    rot = np.eye(3, dtype=np.float64)
    fd = 20.0
    zoom = 1.0
    light = np.array([-9.0, -7.0, 10.0], dtype=np.float64)

    all_pts = np.array(pool.get_all_points())

    print(f"\n--- Projection ({n_faces * pts_per_face} points) ---")
    bench("Rust project_all_points", lambda: project_all_points(pool, fc, rot, fd, zoom))
    bench("Python np.dot projection", lambda: python_project(all_pts.copy(), fc, rot, fd, zoom))

    print(f"\n--- Shading ({n_faces} objects) ---")
    bench("Rust shade_all_objects", lambda: shade_all_objects(pool, light, 0.2, 0.7))

    print(f"\n--- Z-Sort ({n_faces} objects) ---")
    bench("Rust z_sort", lambda: z_sort(pool, rot))

    print(f"\n--- Family Order ({n_faces} objects) ---")
    # Force recompute each time by dirtying
    def family_bench():
        pool.mark_family_dirty()
        get_family_order(pool)
    bench("Rust get_family_order", family_bench)

    print(f"\n--- Hashing ({n_faces} objects, ~{pool.total_points() * 24 / 1024:.0f} KB) ---")
    bench("Rust hash_pool_state", lambda: hash_pool_state(pool))

    print(f"\n--- Clone ({n_faces} objects) ---")
    bench("Rust clone_pool", lambda: clone_pool(pool))

    print(f"\n--- Interpolation ({n_faces} objects) ---")
    start = clone_pool(pool)
    end = clone_pool(pool)
    target = clone_pool(pool)
    bench("Rust interpolate_pools", lambda: interpolate_pools(target, start, end, 0.5, 0))

    print("\n--- Simulated per-frame pipeline (218 frames) ---")
    n_frames = 218

    def full_frame():
        project_all_points(pool, fc, rot, fd, zoom)
        shade_all_objects(pool, light, 0.2, 0.7)
        z_sort(pool, rot)

    t0 = time.perf_counter()
    for _ in range(n_frames):
        full_frame()
    elapsed = time.perf_counter() - t0
    print(f"  Rust full pipeline: {elapsed:.3f}s for {n_frames} frames ({elapsed/n_frames*1000:.2f} ms/frame)")

    def full_frame_with_interp():
        project_all_points(pool, fc, rot, fd, zoom)
        shade_all_objects(pool, light, 0.2, 0.7)
        z_sort(pool, rot)
        interpolate_pools(target, start, end, 0.5, 0)

    t0 = time.perf_counter()
    for _ in range(n_frames):
        full_frame_with_interp()
    elapsed = time.perf_counter() - t0
    print(f"  Rust full pipeline + interp: {elapsed:.3f}s for {n_frames} frames ({elapsed/n_frames*1000:.2f} ms/frame)")


if __name__ == "__main__":
    main()
