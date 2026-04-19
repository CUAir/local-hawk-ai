#!/usr/bin/env python3
"""
GPS + SIFT + Laplacian Pyramid Blending Pipeline

Stitches drone images using:
1. GPS coordinates (from CSV telemetry) for initial image placement
2. SIFT feature matching with RANSAC for visual alignment and pose refinement
3. Voronoi seam finding + Laplacian pyramid blending for seamless output

Usage:
    python main_gps_sift.py <image_folder> [--csv metadata.csv] [--output outputs]
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Keep OpenMP behavior consistent on macOS.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, str(Path(__file__).parent))

from stitcher import stitch_geolocated_images
from dji_dataextraction import process_folder as dji_process_folder, write_metadata_to_csv

REFERENCE_ALTITUDE = 100.0  # meters — altitude-normalization baseline


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _read_csv(csv_path: str) -> Dict[str, Tuple[float, float, float, float]]:
    """Read metadata CSV → {image_name: (lat, lon, alt, heading_deg)}."""
    metadata: Dict[str, Tuple[float, float, float, float]] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Image"]
            lat = float(row["Latitude"])
            lon = float(row["Longitude"])
            alt = float(row["Altitude"])
            heading = float(row["Degrees_Clockwise_from_North"])
            metadata[name] = (lat, lon, alt, heading)
    return metadata


def _clamp_altitude_outliers(
    metadata: Dict[str, Tuple[float, float, float, float]],
    tolerance: float = 0.40,
) -> Dict[str, Tuple[float, float, float, float]]:
    """Clamp altitude outliers to median (preserve image coverage, fix scale)."""
    alts = sorted(v[2] for v in metadata.values())
    median_alt = alts[len(alts) // 2]
    return {
        img: (
            lat,
            lon,
            median_alt if abs(alt - median_alt) / (median_alt + 1e-6) > tolerance else alt,
            heading,
        )
        for img, (lat, lon, alt, heading) in metadata.items()
    }


def _rotate_image(image: np.ndarray, heading_deg: float) -> np.ndarray:
    """Rotate image so North is up (heading_deg is clockwise from North)."""
    angle = -heading_deg
    if angle == 0:
        return image
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_cols = int(rows * sin + cols * cos)
    new_rows = int(rows * cos + cols * sin)
    M[0, 2] += new_cols / 2 - cols / 2
    M[1, 2] += new_rows / 2 - rows / 2
    return cv2.warpAffine(image, M, (new_cols, new_rows))


def _resize_image(image: np.ndarray, altitude: float) -> np.ndarray:
    """Scale image proportional to altitude relative to REFERENCE_ALTITUDE."""
    if altitude <= 0:
        return image
    factor = altitude / REFERENCE_ALTITUDE
    return cv2.resize(image, None, fx=factor, fy=factor)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """CLAHE on L channel (LAB) for visual brightness normalization before blending.
    _extract_one applies a separate CLAHE on grayscale for feature detection — no conflict."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(4, 4))
    l = clahe.apply(l)
    l = cv2.multiply(l, 0.75)  # avoid oversharpening after CLAHE
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


class GpsSiftPipeline:
    """
    GPS-guided SIFT stitching pipeline with Laplacian pyramid blending.

    Stages:
    1. Load images and CSV metadata (auto-extract from DJI XMP if CSV absent)
    2. Pre-process: normalize (CLAHE) → resize (altitude) → rotate (north-up)
    3. Stitch: GPS-guided SIFT matching + global refinement + Voronoi seams +
               Laplacian pyramid blend
    4. Save timestamped output
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dng"}

    def __init__(
        self,
        output_dir: Optional[str] = None,
        test_type: str = "gps_sift",
        verbose: bool = False,
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        self.test_type = test_type
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, image_folder: str, csv_path: Optional[str] = None) -> str:
        """
        Run the full GPS + SIFT + Laplacian pipeline.

        Args:
            image_folder: Folder containing drone images.
            csv_path: Path to telemetry CSV. If None, looks for
                      <image_folder>.csv; extracts from DJI XMP if missing.

        Returns:
            Path to the saved orthomosaic JPEG.
        """
        t0 = time.time()
        folder = Path(image_folder)

        # --- Resolve / auto-generate CSV ---
        if csv_path is None:
            csv_path = str(folder.with_suffix(".csv"))
        if not Path(csv_path).exists():
            self.logger.info("CSV not found at %s — extracting from DJI metadata...", csv_path)
            rows, stats = dji_process_folder(str(folder))
            if not rows:
                raise RuntimeError("No telemetry found in image XMP/EXIF metadata")
            write_metadata_to_csv(rows, csv_path)
            self.logger.info(
                "Extracted %d rows → %s  (skipped %d missing lat/lon)",
                stats["rows_written"],
                csv_path,
                stats["missing_latlon"],
            )

        # --- Load metadata ---
        metadata = _read_csv(csv_path)
        if not metadata:
            raise ValueError(f"No entries found in CSV: {csv_path}")
        metadata = _clamp_altitude_outliers(metadata)
        self.logger.info("Loaded metadata for %d images", len(metadata))

        # --- Load and pre-process images ---
        # Iterate in CSV row order (temporal capture order) so the stitcher's
        # anchor image (index 0) matches the first-captured image, consistent
        # with with_stitch_main.py behaviour on macOS (os.listdir ≈ creation order).
        images: List[np.ndarray] = []
        coordinates: List[Tuple[float, float]] = []
        skipped = 0

        for name in metadata:  # dict preserves CSV insertion order (Python 3.7+)
            path = folder / name
            if not path.exists():
                skipped += 1
                continue
            img = cv2.imread(str(path))
            if img is None:
                self.logger.warning("Failed to load %s — skipping", name)
                skipped += 1
                continue
            lat, lon, alt, heading = metadata[name]
            img = _normalize_image(img)
            img = _resize_image(img, alt)
            img = _rotate_image(img, heading)
            images.append(img)
            coordinates.append((lat, lon))

        if not images:
            raise ValueError("No valid images found with matching metadata")

        self.logger.info(
            "Loaded %d images (skipped %d) in %.1fs",
            len(images),
            skipped,
            time.time() - t0,
        )

        # --- Stitch ---
        t_stitch = time.time()
        canvas, placed_indices, _ = stitch_geolocated_images(images, coordinates)
        elapsed_stitch = time.time() - t_stitch

        if canvas is None:
            raise RuntimeError("Stitcher returned no output")

        self.logger.info(
            "Stitched %d/%d images in %.1fs",
            len(placed_indices),
            len(images),
            elapsed_stitch,
        )

        # --- Save output ---
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pictureset = re.sub(r"[^a-z0-9]+", "-", folder.name.lower()).strip("-")
        filename = f"{date_str}_{pictureset}_{self.test_type}.jpg"

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(self.output_dir / filename)
        else:
            out_path = str(Path(__file__).parent / filename)

        cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.logger.info("Saved → %s  (total %.1fs)", out_path, time.time() - t0)
        return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPS + SIFT + Laplacian pyramid blending pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image_folder", help="Folder containing drone images")
    parser.add_argument(
        "--csv",
        default=None,
        help="Metadata CSV path (default: <image_folder>.csv; auto-extracted if absent)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for the result JPEG (default: mapping/ folder)",
    )
    parser.add_argument(
        "--test-type",
        default="gps_sift",
        help="Label used in the output filename: YYYY-MM-DD_HH-MM-SS_<pictureset>_<test-type>.jpg",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    pipeline = GpsSiftPipeline(
        output_dir=args.output,
        test_type=args.test_type,
        verbose=args.verbose,
    )
    out_path = pipeline.run(args.image_folder, args.csv)
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
