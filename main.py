import argparse
import yaml
from pathlib import Path
from pipeline_config import createPipelineConfig
from video_processing import FFmpegFrameExtractor
from frame_quality_filter import FrameQualityFilter
from colmap_runner import ColmapPipeline
from openmvs_runner import OpenMVSPipeline
from mesh_bpa import create_bpa_mesh
from mesh_poisson import create_poisson_mesh
from mesh_alpha import create_alpha_mesh

SCRIPT_DIR = Path(__file__).resolve().parent
TOOL_CONFIG_PATH = SCRIPT_DIR / "tool_config.yaml"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Photogrammetry pipeline for 3D reconstruction"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input video file, image folder, or PLY point cloud file"
    )

    parser.add_argument(
        "--output_folder",
        required=True,
        help="Folder where all results will be saved"
    )

    parser.add_argument(
        "--visualize",
        type=str_to_bool,
        default=False,
        help="Show Open3D visualization windows: true or false"
    )

    parser.add_argument(
    "--bpa",
    action="store_true",
    help="Run Ball Pivoting Algorithm mesh generation"
    )

    parser.add_argument(
        "--poisson",
        action="store_true",
        help="Run Poisson mesh generation"
    )

    parser.add_argument(
        "--alpha",
        action="store_true",
        help="Run Alpha Shapes mesh generation"
    )

    return parser.parse_args()

def str_to_bool(value: str) -> bool:
    if value.lower() in ("true", "1", "yes", "y"):
        return True
    if value.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected true or false")

def get_selected_mesh_algorithms(args) -> list[str]:
    selected = []

    if args.bpa:
        selected.append("bpa")

    if args.poisson:
        selected.append("poisson")

    if args.alpha:
        selected.append("alpha")

    # If user did not select anything, run all algorithms
    if not selected:
        selected = ["bpa", "poisson", "alpha"]

    return selected

def load_tool_config() -> dict:
    if not TOOL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Tool config file not found: {TOOL_CONFIG_PATH}")

    with open(TOOL_CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}

def prepare_output_dir(output_dir: Path):
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path exists but is not a folder: {output_dir}")

        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output folder is not empty: {output_dir}. "
                "Please provide an empty folder or choose a new output folder."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

def detect_input_type(input_path: Path) -> str:
    video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    point_cloud_extensions = {".ply"}

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        suffix = input_path.suffix.lower()

        if suffix in video_extensions:
            return "video"

        if suffix in point_cloud_extensions:
            return "ply"

        raise ValueError(
            f"Unsupported input file type: {suffix}. "
            "Supported file types are video files and .ply point clouds."
        )

    if input_path.is_dir():
        image_files = [
            file for file in input_path.iterdir()
            if file.is_file() and file.suffix.lower() in image_extensions
        ]

        if image_files:
            return "images"

        raise ValueError(
            f"Input folder does not contain supported image files: {input_path}"
        )

    raise ValueError(f"Unsupported input path: {input_path}")


    
def main():
    args = parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output_folder)
    visualize_mesh = args.visualize

    if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

    prepare_output_dir(output_dir)
    input_type = detect_input_type(input_path)

    print(f"Detected input type: {input_type}")

    cfg = createPipelineConfig(
                output_dir=output_dir,
                tool_config=load_tool_config()
    )
    
    if input_type == "video":
        frames_raw_dir = output_dir / "frames_raw"
        frames_filtered_dir = output_dir / "frames_filtered"

        extractor = FFmpegFrameExtractor(cfg.ffmpeg_path)
        extractor.extract_frames(
            video_path=input_path,
            output_dir=frames_raw_dir,
            fps=2.0
        )

        frame_filter = FrameQualityFilter()
        frame_filter.filter_frames(
            input_dir=frames_raw_dir,
            output_dir=frames_filtered_dir,
            blur_threshold=10.0,
            duplicate_threshold=5.0
        )


    elif input_type == "images":
        frames_filtered_dir = output_dir / "frames_filtered"

        frame_filter = FrameQualityFilter()
        frame_filter.filter_frames(
            input_dir=input_path,
            output_dir=frames_filtered_dir,
            blur_threshold=10.0,
            duplicate_threshold=5.0
        )

    elif input_type == "ply":
        print("PLY input detected. Skipping COLMAP and using point cloud directly.")

    else:
        raise ValueError(f"Unknown input type: {input_type}")

    if input_type in ["video", "images"]:
        colmap = ColmapPipeline(cfg)
        colmap.run_all()

        openMvs = OpenMVSPipeline(cfg)
        point_cloud_result = openMvs.run_all()


    selected_algorithms = get_selected_mesh_algorithms(args)

    if input_type == "ply":
        point_cloud_result = input_path

    if "bpa" in selected_algorithms:
        create_bpa_mesh(
            input_path=point_cloud_result,
            output_path=cfg.result_dir / "bpa_mesh.obj",
            visualize=visualize_mesh
        )

    if "poisson" in selected_algorithms:
        create_poisson_mesh(
            input_path=point_cloud_result,
            output_path=cfg.result_dir / "poisson_mesh.obj",
            visualize=visualize_mesh
        )

    if "alpha" in selected_algorithms:
        create_alpha_mesh(
            input_path=point_cloud_result,
            output_path=cfg.result_dir / "alpha_mesh.obj",
            visualize=visualize_mesh
        )

if __name__ == "__main__":
    main()
    