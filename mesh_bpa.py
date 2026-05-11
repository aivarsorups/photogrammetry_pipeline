from pathlib import Path
import argparse

import open3d as o3d
import numpy as np


def create_bpa_mesh(input_path: Path, output_path: Path, visualize: bool = False) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input point cloud file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(input_path))

    if pcd.is_empty():
        raise ValueError(f"Input point cloud is empty or could not be read: {input_path}")

    if not pcd.has_normals():
        print("Point cloud has no normals. Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1,
                max_nn=30
            )
        )
        pcd.orient_normals_consistent_tangent_plane(30)

    distances = pcd.compute_nearest_neighbor_distance() # katram punktam tiek salasīti  attālumi līdz tuvākajiem kaimiņu punktam
    avg_dist = np.mean(distances)
    median_dist = np.median(distances)

    print("Average NN distance:", avg_dist)
    print("Median NN distance:", median_dist)

    radii = [ # mediānas attālums tiek izmantots bumbiņas rādiusa noteikšanai. 
        1.5 * median_dist,
        2.0 * median_dist,
        4.0 * median_dist
    ]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii)
    )

    mesh.compute_vertex_normals()

    print("Mesh vertices:", np.asarray(mesh.vertices).shape[0])
    print("Mesh triangles:", np.asarray(mesh.triangles).shape[0])

    success = o3d.io.write_triangle_mesh(str(output_path), mesh)

    if not success:
        raise RuntimeError(f"Could not save mesh to: {output_path}")

    print(f"BPA mesh saved to: {output_path}")

    if visualize:
        o3d.visualization.draw_geometries(
            [mesh],
            window_name="Ball-Pivoting Mesh",
            width=1200,
            height=800,
            mesh_show_back_face=True
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create mesh from point cloud using Ball Pivoting Algorithm"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input point cloud file, for example fused.ply"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to output OBJ mesh file"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show mesh visualization window"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if output_path.suffix.lower() != ".obj":
        raise ValueError("Output path must end with .obj")

    create_bpa_mesh(
        input_path=input_path,
        output_path=output_path,
        visualize=args.visualize
    )


if __name__ == "__main__":
    main()