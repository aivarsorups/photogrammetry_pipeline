from pathlib import Path

import open3d as o3d
import numpy as np


def create_poisson_mesh(input_path: Path, output_path: Path, visualize: bool = False) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input point cloud file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(input_path))

    if pcd.is_empty():
        raise ValueError(f"Input point cloud is empty or could not be read: {input_path}")

    print(f"Loaded point cloud: {input_path}")
    print(f"Original points: {len(pcd.points)}")

    # punktu mākonis tiek samazināts, izmantojot vokseļu metodi
    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    print(f"Downsampled points: {len(pcd.points)}")

    # normāļu aprēķināšanas 
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05,
            max_nn=30
        )
    )

    # normāļu orientācijas saskaņošana
    pcd.orient_normals_consistent_tangent_plane(50)

    # Puasona algoritma realizācija
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=10
    )

    # tiek noņemtas virsotnes ar zemākajām blīvuma vērtībām
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.02)
    mesh.remove_vertices_by_mask(densities < threshold)

    mesh.compute_vertex_normals()

    print("Mesh vertices:", np.asarray(mesh.vertices).shape[0])
    print("Mesh triangles:", np.asarray(mesh.triangles).shape[0])

    success = o3d.io.write_triangle_mesh(str(output_path), mesh)

    if not success:
        raise RuntimeError(f"Could not save mesh to: {output_path}")

    print(f"Poisson mesh saved to: {output_path}")

    if visualize:
        o3d.visualization.draw_geometries(
            [mesh],
            window_name="Poisson Mesh",
            width=1200,
            height=800,
            mesh_show_back_face=True
        )