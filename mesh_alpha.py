from pathlib import Path

import open3d as o3d


def create_alpha_mesh(
    input_path: Path,
    output_path: Path,
    visualize: bool = False,
    voxel_size: float = 0.003,
    alpha: float | None = None,
    max_nn: int = 30,
    save_downsampled: bool = False
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input point cloud not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    normal_radius = voxel_size * 6.0
    alpha_value = alpha if alpha is not None else voxel_size * 3.5

    print("Loading point cloud:")
    print(input_path)

    pcd = o3d.io.read_point_cloud(str(input_path))

    if pcd.is_empty():
        raise RuntimeError(f"Point cloud is empty or could not be loaded: {input_path}")

    print(f"Original points: {len(pcd.points)}")

    print(f"Downsampling with voxel_size = {voxel_size}")
    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)

    print(f"Downsampled points: {len(pcd_ds.points)}")

    print("Removing statistical outliers...")
    pcd_ds, _ = pcd_ds.remove_statistical_outlier(
        nb_neighbors=30,
        std_ratio=2.0
    )

    print(f"After statistical outlier removal: {len(pcd_ds.points)}")

    if pcd_ds.is_empty():
        raise RuntimeError("Point cloud became empty after filtering.")

    print("Estimating point cloud normals...")

    pcd_ds.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=max_nn
        )
    )

    print("Orienting point cloud normals...")

    try:
        pcd_ds.orient_normals_consistent_tangent_plane(k=max_nn)
    except Exception as e:
        print("Warning: could not orient point cloud normals consistently.")
        print(e)

    print("Point cloud has normals:", pcd_ds.has_normals())

    if save_downsampled:
        downsampled_path = output_path.parent / f"{input_path.stem}_downsampled_with_normals.ply"
        o3d.io.write_point_cloud(str(downsampled_path), pcd_ds)
        print(f"Downsampled point cloud saved to: {downsampled_path}")

    print(f"Creating alpha shape mesh with alpha = {alpha_value}")

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd_ds,
        alpha_value
    )

    if mesh.is_empty():
        raise RuntimeError(
            "Alpha shape mesh is empty. Try increasing alpha value "
            "or decreasing voxel_size."
        )

    print(f"Mesh vertices before cleaning: {len(mesh.vertices)}")
    print(f"Mesh triangles before cleaning: {len(mesh.triangles)}")

    print("Cleaning mesh...")

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    try:
        mesh.orient_triangles()
    except Exception as e:
        print("Warning: could not orient mesh triangles.")
        print(e)

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    print(f"Mesh vertices after cleaning: {len(mesh.vertices)}")
    print(f"Mesh triangles after cleaning: {len(mesh.triangles)}")
    print("Mesh has triangle normals:", mesh.has_triangle_normals())
    print("Mesh has vertex normals:", mesh.has_vertex_normals())

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(
            "Alpha mesh has no vertices or triangles after cleaning."
        )

    success = o3d.io.write_triangle_mesh(
        str(output_path),
        mesh,
        write_vertex_normals=True,
        write_triangle_uvs=False
    )

    if not success:
        raise RuntimeError(f"Failed to save alpha mesh to: {output_path}")

    if not output_path.exists():
        raise RuntimeError(f"Mesh was not created: {output_path}")

    print(f"Alpha mesh saved to: {output_path}")

    if visualize:
        o3d.visualization.draw_geometries(
            [mesh],
            window_name="Alpha Shape Mesh",
            width=1200,
            height=800,
            mesh_show_back_face=True
        )