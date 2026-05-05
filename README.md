# Photogrammetry Pipeline

Automated photogrammetry pipeline for 3D reconstruction from video files, image folders, or PLY point clouds.

The pipeline supports:

- video input;
- image folder input;
- PLY point cloud input;
- frame extraction with FFmpeg;
- image filtering;
- COLMAP reconstruction;
- OpenMVS dense reconstruction;
- Open3D mesh generation using:
  - Ball Pivoting Algorithm;
  - Poisson Surface Reconstruction;
  - Alpha Shapes.

---

## Requirements

### Python

Required:

- Python 3.12 64-bit
- Windows 10/11 64-bit

Important: 32-bit Python is not supported because the `open3d` package requires a 64-bit Python environment.

To check your Python version and architecture, run:

```bash
python --version
python -c "import platform; print(platform.architecture())"

Expected result:
Python 3.12.x
('64bit', 'WindowsPE')
```

Install Python requirements
run 
python -m pip install -r requirements.txt


### External tools

The following tools must be installed separately. Their executable paths should be configured in `tool_config.yaml`.:

- FFmpeg
download: https://www.ffmpeg.org/download.html

- COLMAP
download from https://github.com/colmap/colmap/releases
if CUDA available colmap-x64-windows-cuda.zip
if CUDA unavailable colmap-x64-windows-nocuda.zip

- OpenMVS
download https://github.com/cdcseacave/openMVS/releases



### Exmples to run:

Run BPA, Poisson, Alpha mesh algorithms without visualization:
    python main.py --input "C:\\path\\to\\cloud.ply" --output_folder "C:\\path\\to\\output"

Run only BPA:
    python main.py --input "C:\\path\\to\\cloud.ply" --output_folder "C:\\path\\to\\output" --bpa

Run BPA with visualization:
    python main.py --input "C:\\path\\to\\cloud.ply" --output_folder "C:\\path\\to\\output" --bpa --visualize true

Run Poisson and Alpha without visualization:
    python main.py --input "C:\\path\\to\\cloud.ply" --output_folder "C:\\path\\to\\output" --poisson --alpha --visualize false
