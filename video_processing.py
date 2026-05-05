from pathlib import Path
import subprocess


class FFmpegFrameExtractor:
    def __init__(self, ffmpeg_path: str):
        self.ffmpeg_path = ffmpeg_path

    def extract_frames(self, video_path: Path, output_dir: Path, fps: float):
        output_dir.mkdir(parents=True, exist_ok=True)

        output_pattern = output_dir / "frame_%06d.png"

        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            str(output_pattern),
        ]

        self.run_command(cmd)

    @staticmethod
    def run_command(cmd: list[str]):
        print("\nRunning command:")
        print(" ".join(str(c) for c in cmd))

        result = subprocess.run(
            [str(c) for c in cmd],
            cwd=".",
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Command failed with code {result.returncode}")