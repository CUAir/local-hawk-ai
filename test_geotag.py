#!/usr/bin/env python3
"""Test detection + geotagging with real flight images."""

import json
import base64
from pathlib import Path

from constructs.detection import GDDetection
from constructs.geotagging import geotag_candidate
from constructs.image_types import (
    GeoLocation, ImageMeta, Base64Image, LabelTypes
)

# Ground truth coordinates
GROUND_TRUTH = {
    LabelTypes.MANNEQUIN: GeoLocation(lat=42.4451978, lon=-76.4412644),
}


def test_image(detector: GDDetection, image_path: Path, json_path: Path, all_errors: list):
    # Load image as base64
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    # Load telemetry
    with open(json_path, "r") as f:
        data = json.load(f)

    tel = data["telemetry"]
    gps = tel["gps"]

    # Build metadata
    origin = GeoLocation(lat=gps["latitude"], lon=gps["longitude"], alt=tel["altitude"])
    meta = ImageMeta(location=origin, heading=tel["planeYaw"])
    source = Base64Image(id=1, base64_image=b64, meta=meta)

    # Run detection
    candidates = detector.detect_candidates(source)

    print(f"\n=== {image_path.name} ===")
    print(f"Drone: lat={origin.lat:.6f}, lon={origin.lon:.6f}, alt={origin.alt:.1f}m, heading={meta.heading:.1f}°")
    print(f"Detections: {len(candidates)}")

    # Get best mannequin detection (highest score)
    mannequins = [c for c in candidates if c.label == LabelTypes.MANNEQUIN]
    if mannequins:
        best = max(mannequins, key=lambda c: c.score)
        geo = geotag_candidate(best)
        if geo and LabelTypes.MANNEQUIN in GROUND_TRUTH:
            truth = GROUND_TRUTH[LabelTypes.MANNEQUIN]
            error = truth.distance_to(geo)
            all_errors.append(error)
            print(f"  Best mannequin: score={best.score:.3f}")
            print(f"    Predicted: lat={geo.lat:.6f}, lon={geo.lon:.6f}")
            print(f"    Actual:    lat={truth.lat:.6f}, lon={truth.lon:.6f}")
            print(f"    ERROR: {error:.2f}m")


if __name__ == "__main__":
    data_dir = Path("data")
    detector = GDDetection()

    # Find all image/json pairs
    images = sorted(data_dir.glob("GOPR*.JPG"))
    print(f"Found {len(images)} images to process")
    print(f"Ground truth: lat={GROUND_TRUTH[LabelTypes.MANNEQUIN].lat}, lon={GROUND_TRUTH[LabelTypes.MANNEQUIN].lon}")

    all_errors = []

    for img_path in images:
        json_path = data_dir / f"{img_path.stem}_gs.json"
        if json_path.exists():
            test_image(detector, img_path, json_path, all_errors)
        else:
            print(f"\nSkipping {img_path.name}: no JSON metadata")

    # Summary
    if all_errors:
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Images with detections: {len(all_errors)}")
        print(f"Mean error: {sum(all_errors) / len(all_errors):.2f}m")
        print(f"Min error:  {min(all_errors):.2f}m")
        print(f"Max error:  {max(all_errors):.2f}m")

        # Count within thresholds
        within_5m = sum(1 for e in all_errors if e <= 5)
        within_10m = sum(1 for e in all_errors if e <= 10)
        within_15m = sum(1 for e in all_errors if e <= 15)
        print(f"Within 5m:  {within_5m}/{len(all_errors)} ({100*within_5m/len(all_errors):.1f}%)")
        print(f"Within 10m: {within_10m}/{len(all_errors)} ({100*within_10m/len(all_errors):.1f}%)")
        print(f"Within 15m: {within_15m}/{len(all_errors)} ({100*within_15m/len(all_errors):.1f}%)")
