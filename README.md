Install requirements
run 
python -m pip install -r requirements.txt
## External tools

The following tools must be installed separately:

- FFmpeg
download: https://www.ffmpeg.org/download.html

- COLMAP
download from https://github.com/colmap/colmap/releases
if CUDA available colmap-x64-windows-cuda.zip
if CUDA unavailable colmap-x64-windows-nocuda.zip

- OpenMVS
download https://github.com/cdcseacave/openMVS/releases
Their executable paths are configured in `tool_config.yaml`.

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