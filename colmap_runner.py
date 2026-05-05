from pathlib import Path
import subprocess


class ColmapPipeline:
    def __init__(self, cfg):
        self.cfg = cfg

    def run_all(self):
        self.cfg.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.dense_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.mesh_dir.mkdir(parents=True, exist_ok=True)

        self.feature_extraction()
        self.feature_matching()
        self.mapping()
        self.undistort_images()
        if self.cfg.cuda:
            self.patch_match_stereo()
            self.stereo_fusion()
        else:
            print(
                "CUDA is not available. Skipping COLMAP dense reconstruction "
                "(patch_match_stereo and stereo_fusion)."
            )

        # Ja mesh tiek veidots ar Open3D, šo posmu var neizpildīt.
        # self.poisson_meshing()

    def run_command(self, cmd: list[str]):
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
            output = result.stdout or ""

            if (
                "No good initial image pair found" in output
                or "Failed to create any sparse model" in output
                or "Discarding reconstruction due to no initial pair" in output
            ):
                raise RuntimeError(
                    "COLMAP failed to create a sparse reconstruction. "
                    "No good initial image pair was found.\n\n"
                    "Please provide a better video or image set. Recommended fixes:\n"
                    "- move the camera more slowly;\n"
                    "- make sure neighboring frames overlap by about 70–80%;\n"
                    "- avoid blurry frames;\n"
                    "- avoid plain walls, reflective surfaces, and low-light scenes;\n"
                    "- record the object or room from multiple angles;\n"
                    "- reduce frame filtering strictness if too few frames remain;\n"
                    "- try extracting more frames from the video."
                )

            raise RuntimeError(f"Command failed with code {result.returncode}")
        
    def feature_extraction(self):
        cmd = [
            self.cfg.colmap_path,
            "feature_extractor",
            "--database_path", self.cfg.colmap_db_path,
            "--image_path", self.cfg.filtered_frames_dir,
            "--ImageReader.camera_model", self.cfg.camera_model,
            "--ImageReader.single_camera", "1",
            "--FeatureExtraction.use_gpu", "1" if self.cfg.cuda else "0",
            "--FeatureExtraction.max_image_size", str(self.cfg.max_image_size),
        ]

        self.run_command(cmd)

    def feature_matching(self):
        matcher = "sequential_matcher" if self.cfg.is_sequential else "exhaustive_matcher"

        cmd = [
            self.cfg.colmap_path,
            matcher,
            "--database_path", self.cfg.colmap_db_path,
            "--FeatureMatching.use_gpu", "1" if self.cfg.cuda else "0",
        ]

        self.run_command(cmd)

    def mapping(self):
        self.cfg.sparse_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.cfg.colmap_path,
            "mapper",
            "--database_path", self.cfg.colmap_db_path,
            "--image_path", self.cfg.filtered_frames_dir,
            "--output_path", self.cfg.sparse_dir,
        ]

        self.run_command(cmd)

    def get_sparse_model_dir(self):
        model_dirs = [
            path for path in self.cfg.sparse_dir.iterdir()
            if path.is_dir()
        ]

        if not model_dirs:
            raise FileNotFoundError(
                "No sparse model was created by COLMAP mapper."
            )

        # Parasti COLMAP izveido sparse/0, bet drošāk ir atrast pirmo mapi.
        return sorted(model_dirs)[0]

    def undistort_images(self):
        sparse_model_dir = self.get_sparse_model_dir()

        cmd = [
            self.cfg.colmap_path,
            "image_undistorter",
            "--image_path", self.cfg.filtered_frames_dir,
            "--input_path", sparse_model_dir,
            "--output_path", self.cfg.dense_dir,
            "--output_type", "COLMAP",
        ]

        self.run_command(cmd)

    def patch_match_stereo(self):
        cmd = [
            self.cfg.colmap_path,
            "patch_match_stereo",
            "--workspace_path", self.cfg.dense_dir,
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "1",
            "--PatchMatchStereo.filter", "1",
            "--PatchMatchStereo.max_image_size", str(self.cfg.patch_match_max_image_size),
            "--PatchMatchStereo.window_radius", str(self.cfg.patch_match_window_radius),
            "--PatchMatchStereo.num_iterations", str(self.cfg.patch_match_num_iterations),
            "--PatchMatchStereo.num_samples", str(self.cfg.patch_match_num_samples),
            "--PatchMatchStereo.gpu_index", "0" if self.cfg.cuda else "-1",
            "--log_target", "stdout",
        ]

        self.run_command(cmd)

    def stereo_fusion(self):
        output_cloud = self.cfg.dense_dir / "fused.ply"

        cmd = [
            self.cfg.colmap_path,
            "stereo_fusion",
            "--workspace_path", self.cfg.dense_dir,
            "--workspace_format", "COLMAP",
            "--input_type", "photometric",
            "--output_path", output_cloud,
            "--log_target", "stdout",
        ]

        self.run_command(cmd)

    def poisson_meshing(self):
        input_cloud = self.cfg.dense_dir / "fused.ply"
        output_mesh = self.cfg.mesh_dir / "meshed-poisson.ply"

        cmd = [
            self.cfg.colmap_path,
            "poisson_mesher",
            "--input_path", input_cloud,
            "--output_path", output_mesh,
        ]

        self.run_command(cmd)