import subprocess

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    colmap_path: str
    ffmpeg_path: str
    openmvs_interface_path: str
    openmvs_densify_path: str
    openmvs_reconstruct_path: str
    openmvs_refine_path: str
    openmvs_texture_path: str

    filtered_frames_dir: Path
    colmap_db_path: Path
    sparse_dir: Path
    dense_dir: Path
    mesh_dir: Path
    openmvs_dir: Path
    result_dir: Path

    camera_model: str = "SIMPLE_RADIAL"
    max_image_size: int = 2000
    is_sequential: bool = False

    patch_match_max_image_size: int = 1600
    patch_match_window_radius: int = 4
    patch_match_num_iterations: int = 3
    patch_match_num_samples: int = 10

    cuda: bool = False


def createPipelineConfig(output_dir: Path, tool_config: dict) -> PipelineConfig:
    colmap_dir = output_dir / "colmap"

    return PipelineConfig(
        colmap_path=get_tool_path(tool_config, 'colmap'),
        ffmpeg_path=get_tool_path(tool_config, 'ffmpeg'),
        openmvs_interface_path=get_tool_path(tool_config, 'openmvs_interface'),
        openmvs_densify_path=get_tool_path(tool_config, 'openmvs_densify'),
        openmvs_reconstruct_path=get_tool_path(tool_config, 'openmvs_reconstruct'),
        openmvs_refine_path=get_tool_path(tool_config, 'openmvs_refine'),
        openmvs_texture_path=get_tool_path(tool_config,'openmvs_texture'),

        filtered_frames_dir=output_dir / "frames_filtered",
        colmap_db_path=colmap_dir / "database.db",
        sparse_dir=colmap_dir / "sparse",
        dense_dir=colmap_dir / "dense",
        mesh_dir=output_dir / "meshes",
        openmvs_dir=output_dir / "openmvs",
        result_dir=output_dir / "result",

        cuda=is_cuda_available(),
    )

def get_tool_path(tool_config: dict, tool_name: str) -> str:
    
    tools = tool_config.get("tools", {})

    if tool_name not in tools:
        raise KeyError(
            f"Tool '{tool_name}' not found in {tool_config}. "
            f"Available tools: {list(tools.keys())}"
        )

    return tools[tool_name]

def is_cuda_available() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        if result.returncode == 0:
            print("CUDA/NVIDIA GPU detected.")
            return True

        print("CUDA/NVIDIA GPU not detected.")
        print(result.stdout)
        return False

    except FileNotFoundError:
        print("nvidia-smi not found. CUDA is probably not available.")
        return False