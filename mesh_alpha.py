from pathlib import Path

import numpy as np
import open3d as o3d


def create_alpha_mesh(
    input_path: Path,
    output_path: Path,
    visualize: bool = False,
    voxel_size: float = 0.003,
    alpha: float | None = None,
    max_nn: int = 30,
    save_downsampled: bool = False,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input point cloud not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading point cloud:")
    print(input_path)

    pcd = o3d.io.read_point_cloud(str(input_path))

    if pcd.is_empty():
        raise RuntimeError(f"Point cloud is empty or could not be loaded: {input_path}")

    print(f"Original points: {len(pcd.points)}")

    print(f"Downsampling with voxel_size = {voxel_size}")
    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size) # punktu mākoņa samazināšana, izmantojot vokseļu metodi

    print(f"Downsampled points: {len(pcd_ds.points)}")

    if pcd_ds.is_empty():
        raise RuntimeError("Point cloud became empty after downsampling.")

    print("Removing statistical outliers...")
    pcd_ds, _ = pcd_ds.remove_statistical_outlier( # tiek izņemti punkti, kas atrodas pārāk tālu no saviem kaimiņpunktiem 
        nb_neighbors=30,
        std_ratio=2.0,
    )

    print(f"After statistical outlier removal: {len(pcd_ds.points)}")

    if pcd_ds.is_empty():
        raise RuntimeError("Point cloud became empty after filtering.")

    if len(pcd_ds.points) < 4:
        raise RuntimeError(
            "Not enough points to create alpha shape mesh. "
            f"Points after filtering: {len(pcd_ds.points)}"
        )

    print("Calculating nearest-neighbor distances...")

    nn_distances = np.asarray(pcd_ds.compute_nearest_neighbor_distance()) # tiek aprēķināti attālumi līdz tuvākajiem kaimiņu punktiem

    if len(nn_distances) == 0:
        raise RuntimeError("Could not compute nearest-neighbor distances.")

    median_nn = float(np.median(nn_distances))
    mean_nn = float(np.mean(nn_distances))

    if not np.isfinite(median_nn) or median_nn <= 0:
        raise RuntimeError(f"Invalid median nearest-neighbor distance: {median_nn}")

    alpha_value = alpha if alpha is not None else median_nn * 3.5 # mediānas attālums tiek izmantots alfa parametra noteikšanai. 

    print(f"Nearest-neighbor median distance: {median_nn}")
    print(f"Nearest-neighbor mean distance:   {mean_nn}")

    if save_downsampled:
        downsampled_path = output_path.parent / f"{input_path.stem}_downsampled.ply"
        o3d.io.write_point_cloud(str(downsampled_path), pcd_ds)
        print(f"Downsampled point cloud saved to: {downsampled_path}")

    print(f"Creating alpha shape mesh with alpha = {alpha_value}")

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd_ds,
        alpha_value,
    )

    if mesh.is_empty():
        raise RuntimeError(
            "Alpha shape mesh is empty. Try increasing alpha value "
            "or decreasing voxel_size."
        )

    print(f"Mesh vertices before cleaning: {len(mesh.vertices)}")
    print(f"Mesh triangles before cleaning: {len(mesh.triangles)}")

    print("Cleaning mesh...")
    # tiek veikta trijstūru tīkla tīrīšana, no modeļa tiek izņemtas dublētas virsotnes, dublēti trijstūri, deģenerēti trijstūri un neizmantotas virsotnes
    mesh.remove_duplicated_vertices() 
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    try:
        mesh.remove_non_manifold_edges()
    except Exception as e:
        print("Warning: could not remove non-manifold edges.")
        print(e)

    mesh.remove_unreferenced_vertices()

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(
            "Alpha mesh has no vertices or triangles after cleaning."
        )

    try:
        mesh.orient_triangles()
    except Exception as e:
        print("Warning: could not orient mesh triangles.")
        print(e)

    print(f"Mesh vertices after cleaning: {len(mesh.vertices)}")
    print(f"Mesh triangles after cleaning: {len(mesh.triangles)}")

    print(f"Saving alpha mesh to: {output_path}")

    success = o3d.io.write_triangle_mesh(
        str(output_path),
        mesh,
        write_vertex_normals=True,
        write_triangle_uvs=False,
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
            mesh_show_back_face=True,
        )