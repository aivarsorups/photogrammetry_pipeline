from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image


class FrameQualityFilter:
    IMAGE_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    }

    def filter_frames(
        self,
        input_dir: Path,
        output_dir: Path,
        blur_threshold: float,
        duplicate_threshold: float
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        self.validate_images_have_same_dimensions(input_dir)

        images = self.get_image_files(input_dir)

        prev_gray = None
        saved_index = 0

        for image_path in images:
            img = cv2.imread(str(image_path))

            if img is None:
                print(f"Could not read image, skipped: {image_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            blur_score = self.variance_of_laplacian(gray)

            if blur_score < blur_threshold:
                print(f"blurScore: {blur_score:.2f} | threshold: {blur_threshold}")
                print(f"Excluded blur: {image_path}")
                continue

            is_duplicate = False

            if prev_gray is not None:
                diff = self.mean_absolute_difference(prev_gray, gray)

                if diff < duplicate_threshold:
                    is_duplicate = True

            if not is_duplicate:
                output_path = output_dir / f"img_{saved_index:06d}.jpg"
                cv2.imwrite(str(output_path), img)

                prev_gray = gray.copy()
                saved_index += 1
            else:
                print(f"Excluded duplicate: {image_path}")

        print(f"Saved filtered frames: {saved_index}")
        print(f"Filtered frames directory: {output_dir}")

    def variance_of_laplacian(self, gray: np.ndarray) -> float:
        laplacian = cv2.Laplacian(gray, cv2.CV_16S)
        return float(laplacian.var())

    def mean_absolute_difference(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))

        diff = cv2.absdiff(a, b)
        return float(np.mean(diff))

    def validate_images_have_same_dimensions(self, image_dir: Path) -> None:
        if not image_dir.exists():
            raise FileNotFoundError(f"Input folder does not exist: {image_dir}")

        if not image_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a folder: {image_dir}")

        image_files = self.get_image_files(image_dir)

        if not image_files:
            raise FileNotFoundError(f"No image files found in input folder: {image_dir}")

        first_image_path = image_files[0]

        with Image.open(first_image_path) as first_image:
            expected_width, expected_height = first_image.size

        print(
            f"Expected image dimension: "
            f"{expected_width}x{expected_height} "
            f"from {first_image_path.name}"
        )

        for image_path in image_files:
            with Image.open(image_path) as image:
                width, height = image.size

            if width != expected_width or height != expected_height:
                raise ValueError(
                    f"Image dimensions mismatch. "
                    f"Expected {expected_width}x{expected_height}, "
                    f"but got {width}x{height} "
                    f"in file: {image_path.name}"
                )

        print(
            f"All input images have same dimension: "
            f"{expected_width}x{expected_height}. "
            f"Total images: {len(image_files)}"
        )

    def get_image_files(self, image_dir: Path) -> list[Path]:
        return sorted(
            [
                path for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS
            ],
            key=lambda path: path.name
        )

    def is_image_file(self, path: Path) -> bool:
        return path.suffix.lower() in self.IMAGE_EXTENSIONS

    def replace_extension_with_jpg(self, file_name: str) -> str:
        return str(Path(file_name).with_suffix(".jpg"))