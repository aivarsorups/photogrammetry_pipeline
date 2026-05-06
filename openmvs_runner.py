from pathlib import Path
import subprocess


class OpenMVSPipeline:
    def __init__(self, cfg):
        self.cfg = cfg

    def run_all(self) -> Path:
        self.cfg.openmvs_dir.mkdir(parents=True, exist_ok=True)

        self.interface_colmap()

        try:
            self.densify_point_cloud()
        except Exception as e:
            print("Could not create dense point cloud.")
            print(e)

        print("=== OpenMVS DONE ===")
        print(f"Scene: {self.cfg.openmvs_dir / 'scene.mvs'}")
        print(f"Dense scene: {self.cfg.openmvs_dir / 'scene_dense.mvs'}")
        return self.cfg.openmvs_dir / "scene_dense.ply"
       
    def run_command(self, cmd: list):
        print("\nRunning command:")
        print(" ".join(str(c) for c in cmd))

        result = subprocess.run(
            [str(c) for c in cmd],
            cwd=".",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )

        print(result.stdout)

        if result.returncode != 0:
            raise RuntimeError(f"Command failed with code {result.returncode}")

    def interface_colmap(self):
        print("interface_colmap run")

        cmd = [
            self.cfg.openmvs_interface_path,
            "-i", str(self.cfg.dense_dir),
            "-o", str(self.cfg.openmvs_dir / "scene.mvs"),
            "--image-folder", str(self.cfg.dense_dir / "images"),
        ]

        self.run_command(cmd)

    def densify_point_cloud(self):
        print("densify_point_cloud run")

        cmd = [
            self.cfg.openmvs_densify_path,
            "--input-file", str(self.cfg.openmvs_dir / "scene.mvs"),
            "--working-folder", str(self.cfg.openmvs_dir),
            "--output-file", str(self.cfg.openmvs_dir / "scene_dense.mvs"),
            "--archive-type", "-1",
        ]

        self.run_command(cmd)

    def reconstruct_mesh(self):
        print("reconstruct_mesh run")

        cmd = [
            self.cfg.openmvs_reconstruct_path,
            "--input-file", str(self.cfg.openmvs_dir / "scene_dense.mvs"),
            "--working-folder", str(self.cfg.openmvs_dir),
            "--output-file", str(self.cfg.openmvs_dir / "scene_dense_mesh.ply"),
            "-v", "3",
        ]

        self.run_command(cmd)

    def refine_mesh(self):
        print("refine_mesh run")

        scene_file = self.cfg.openmvs_dir / "scene_dense.mvs"
        mesh_file = self.cfg.openmvs_dir / "scene_dense_mesh.ply"
        output_file = self.cfg.openmvs_dir / "scene_dense_mesh_refine.ply"

        if not scene_file.exists():
            raise FileNotFoundError(f"Scene file not found for refine: {scene_file}")

        if not mesh_file.exists():
            raise FileNotFoundError(f"Mesh file not found for refine: {mesh_file}")

        cmd = [
            self.cfg.openmvs_refine_path,
            str(scene_file),
            "-m", str(mesh_file),
            "-o", str(output_file),
            "--working-folder", str(self.cfg.openmvs_dir),
            "--resolution-level", "0",
            "--max-face-area", "8",
            "-v", "3",
        ]

        self.run_command(cmd)

    def texture_mesh(self, mesh_file: Path, output_file: Path):
        print("texture_mesh run")

        scene_file = self.cfg.openmvs_dir / "scene_dense.mvs"

        if not scene_file.exists():
            raise FileNotFoundError(f"Scene file not found for texture: {scene_file}")

        if not mesh_file.exists():
            raise FileNotFoundError(f"Mesh file not found for texture: {mesh_file}")

        cmd = [
            self.cfg.openmvs_texture_path,
            str(scene_file),
            "--mesh-file", str(mesh_file),
            "-o", str(output_file),
            "--export-type", "obj",
            "--working-folder", str(self.cfg.openmvs_dir),
            "-v", "3",
        ]

        self.run_command(cmd)