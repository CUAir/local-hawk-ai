#!/usr/bin/env python3
"""Robust DJI telemetry extraction for mapping CSV generation.

Output CSV columns (pipeline-compatible):
- Image
- Latitude
- Longitude
- Altitude
- Degrees_Clockwise_from_North
- Timestamp (optional helper column)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

from PIL import ExifTags, Image


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DJI_METADATA_FIELDS = [
    "GpsLatitude",
    "GpsLongitude",
    "AbsoluteAltitude",
    "RelativeAltitude",
    "GimbalRollDegree",
    "GimbalYawDegree",
    "GimbalPitchDegree",
    "FlightRollDegree",
    "FlightYawDegree",
    "FlightPitchDegree",
]

XML_NAMESPACES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "drone-dji": "http://www.dji.com/drone-dji/1.0/",
}

XMP_START = b"<x:xmpmeta"
XMP_END = b"</x:xmpmeta>"
DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".dng")


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            value = value.strip().strip("+")
        return float(value)
    except Exception:
        return None


def _rational_to_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            denom = float(value.denominator)
            if denom == 0:
                return None
            return float(value.numerator) / denom
        if isinstance(value, tuple) and len(value) == 2:
            denom = float(value[1])
            if denom == 0:
                return None
            return float(value[0]) / denom
        return float(value)
    except Exception:
        return None


def _dms_to_decimal(dms: object, ref: object) -> Optional[float]:
    try:
        if dms is None:
            return None
        parts = list(dms)
        if len(parts) < 3:
            return None
        deg = _rational_to_float(parts[0])
        minute = _rational_to_float(parts[1])
        second = _rational_to_float(parts[2])
        if deg is None or minute is None or second is None:
            return None
        dec = deg + minute / 60.0 + second / 3600.0
        if ref in ("S", "W", b"S", b"W"):
            dec = -dec
        return float(dec)
    except Exception:
        return None


def extract_xmp_packets(image_path: str) -> List[bytes]:
    """Extract individual XMP packets from a file."""
    try:
        data = Path(image_path).read_bytes()
    except Exception as exc:
        logger.error(f"Failed to read {image_path}: {exc}")
        return []

    packets: List[bytes] = []
    idx = 0
    end_len = len(XMP_END)
    while True:
        start = data.find(XMP_START, idx)
        if start < 0:
            break
        end = data.find(XMP_END, start)
        if end < 0:
            break
        end += end_len
        packets.append(data[start:end])
        idx = end
    return packets


def extract_dji_xmp_metadata(image_path: str) -> Dict[str, float]:
    """Extract DJI telemetry from XMP attributes."""
    out: Dict[str, float] = {}
    packets = extract_xmp_packets(image_path)
    for packet in packets:
        try:
            root = ET.fromstring(packet)
        except ET.ParseError:
            continue
        for desc in root.findall(".//rdf:Description", XML_NAMESPACES):
            for field in DJI_METADATA_FIELDS:
                attr = f"{{http://www.dji.com/drone-dji/1.0/}}{field}"
                if attr in desc.attrib:
                    val = _safe_float(desc.attrib[attr])
                    if val is not None:
                        out[field] = val
    return out


def extract_exif_metadata(image_path: str) -> Dict[str, object]:
    """Extract fallback GPS/yaw/time metadata from EXIF."""
    out: Dict[str, object] = {}
    try:
        img = Image.open(image_path)
        exif = img._getexif() or {}
    except Exception:
        return out

    tag_to_id = {v: k for k, v in ExifTags.TAGS.items()}
    gps_tag_id = tag_to_id.get("GPSInfo")
    dt_tag_id = tag_to_id.get("DateTimeOriginal")

    if dt_tag_id is not None:
        dt_raw = exif.get(dt_tag_id)
        if dt_raw:
            out["DateTimeOriginal"] = str(dt_raw)

    if gps_tag_id is None:
        return out

    gps = exif.get(gps_tag_id)
    if not isinstance(gps, dict):
        return out

    lat = _dms_to_decimal(gps.get(2), gps.get(1))
    lon = _dms_to_decimal(gps.get(4), gps.get(3))
    alt = _rational_to_float(gps.get(6))
    alt_ref = gps.get(5, 0)
    yaw = _rational_to_float(gps.get(17))

    if lat is not None:
        out["GpsLatitude"] = float(lat)
    if lon is not None:
        out["GpsLongitude"] = float(lon)
    if alt is not None:
        out["GpsAltitude"] = -abs(alt) if alt_ref == 1 else float(alt)
    if yaw is not None:
        out["GPSImgDirection"] = float(yaw)

    return out


def extract_gs_json_metadata(image_path: str) -> Dict[str, object]:
    """Extract telemetry from a GoPro-style sidecar JSON (``<image>_gs.json``).

    Produces keys aligned with the DJI XMP schema so :func:`normalize_row`
    can consume it unchanged. ``telemetry.altitude`` in these files is a
    low, relative height; callers may choose to override/zero it before
    writing the CSV.
    """
    p = Path(image_path)
    json_path = p.with_name(f"{p.stem}_gs.json")
    if not json_path.exists():
        return {}
    try:
        data = json.loads(json_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Failed to read %s: %s", json_path, exc)
        return {}

    tel = data.get("telemetry") or {}
    gps = tel.get("gps") or {}

    out: Dict[str, object] = {}
    lat = _safe_float(gps.get("latitude"))
    lon = _safe_float(gps.get("longitude"))
    alt = _safe_float(tel.get("altitude"))
    yaw = _safe_float(tel.get("planeYaw"))
    ts = data.get("timestamp_utc") or data.get("timestamp")

    if lat is not None:
        out["GpsLatitude"] = lat
    if lon is not None:
        out["GpsLongitude"] = lon
    if alt is not None:
        out["AbsoluteAltitude"] = alt
    if yaw is not None:
        out["FlightYawDegree"] = yaw
    if ts is not None:
        out["DateTimeOriginal"] = str(ts)
    return out


def extract_telemetry(image_path: str) -> Tuple[Dict[str, object], Dict[str, bool]]:
    """Extract merged telemetry with XMP-first, EXIF+sidecar JSON fallback."""
    xmp = extract_dji_xmp_metadata(image_path)
    exif = extract_exif_metadata(image_path)
    gs_json = extract_gs_json_metadata(image_path)

    meta: Dict[str, object] = {}
    meta.update(xmp)

    # Fill from EXIF only when DJI XMP field is absent.
    if "GpsLatitude" not in meta and "GpsLatitude" in exif:
        meta["GpsLatitude"] = exif["GpsLatitude"]
    if "GpsLongitude" not in meta and "GpsLongitude" in exif:
        meta["GpsLongitude"] = exif["GpsLongitude"]
    if "RelativeAltitude" not in meta and "AbsoluteAltitude" not in meta and "GpsAltitude" in exif:
        meta["AbsoluteAltitude"] = exif["GpsAltitude"]
    if "FlightYawDegree" not in meta and "GimbalYawDegree" not in meta and "GPSImgDirection" in exif:
        meta["FlightYawDegree"] = exif["GPSImgDirection"]

    # Fill from GoPro sidecar JSON last (lowest priority) so DJI/EXIF win if present.
    if "GpsLatitude" not in meta and "GpsLatitude" in gs_json:
        meta["GpsLatitude"] = gs_json["GpsLatitude"]
    if "GpsLongitude" not in meta and "GpsLongitude" in gs_json:
        meta["GpsLongitude"] = gs_json["GpsLongitude"]
    if ("RelativeAltitude" not in meta and "AbsoluteAltitude" not in meta
            and "AbsoluteAltitude" in gs_json):
        meta["AbsoluteAltitude"] = gs_json["AbsoluteAltitude"]
    if ("FlightYawDegree" not in meta and "GimbalYawDegree" not in meta
            and "FlightYawDegree" in gs_json):
        meta["FlightYawDegree"] = gs_json["FlightYawDegree"]
    if "DateTimeOriginal" not in meta and "DateTimeOriginal" in gs_json:
        meta["DateTimeOriginal"] = gs_json["DateTimeOriginal"]

    if "DateTimeOriginal" in exif:
        meta["DateTimeOriginal"] = exif["DateTimeOriginal"]

    sources = {
        "xmp": bool(xmp),
        "exif": bool(exif),
        "gs_json": bool(gs_json),
    }
    return meta, sources


def normalize_row(
    filename: str,
    telemetry: Dict[str, object],
    require_telemetry: bool = False,
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    """Map merged metadata into pipeline CSV row schema.

    If ``require_telemetry`` is True, rows missing altitude OR heading from
    any real source (i.e. would otherwise be silently defaulted to 0.0) are
    rejected, returning ``(None, 'missing_<field>')``.
    """
    lat = _safe_float(telemetry.get("GpsLatitude"))
    lon = _safe_float(telemetry.get("GpsLongitude"))
    if lat is None or lon is None:
        return None, "missing_latlon"

    alt = _safe_float(telemetry.get("RelativeAltitude"))
    if alt is None:
        alt = _safe_float(telemetry.get("AbsoluteAltitude"))
    alt_defaulted = alt is None
    if alt is None:
        alt = 0.0

    yaw = _safe_float(telemetry.get("FlightYawDegree"))
    if yaw is None:
        yaw = _safe_float(telemetry.get("GimbalYawDegree"))
    yaw_defaulted = yaw is None
    if yaw is None:
        yaw = 0.0

    if require_telemetry:
        if alt_defaulted:
            return None, "missing_altitude"
        if yaw_defaulted:
            return None, "missing_yaw"

    ts = str(telemetry.get("DateTimeOriginal", "")).strip()

    row = {
        "Image": filename,
        "Latitude": float(lat),
        "Longitude": float(lon),
        "Altitude": float(alt),
        "Degrees_Clockwise_from_North": float(yaw),
        "Timestamp": ts,
    }
    return row, None


def iter_image_files(folder_path: Path, recursive: bool, extensions: Tuple[str, ...]) -> Iterable[Path]:
    if recursive:
        files = (p for p in folder_path.rglob("*") if p.is_file())
    else:
        files = (p for p in folder_path.iterdir() if p.is_file())

    ext_set = {e.lower() for e in extensions}
    selected = [p for p in files if p.suffix.lower() in ext_set]
    selected.sort(key=lambda p: p.name)
    return selected


def process_folder(
    folder_path: str,
    recursive: bool = False,
    extensions: Tuple[str, ...] = DEFAULT_EXTENSIONS,
    sort_by: str = "datetime",
    require_telemetry: bool = False,
    min_altitude_m: Optional[float] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Process images and return normalized CSV rows plus stats.

    When ``require_telemetry`` is True, rows whose altitude or yaw would be
    silently defaulted are dropped (counted under ``missing_altitude`` /
    ``missing_yaw``). When ``min_altitude_m`` is set, rows below that altitude
    are dropped (counted under ``dropped_low_altitude``).
    """
    base = Path(folder_path)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Folder path does not exist or is not a directory: {folder_path}")

    stats = {
        "files_seen": 0,
        "rows_written": 0,
        "xmp_source": 0,
        "exif_source": 0,
        "gs_json_source": 0,
        "missing_latlon": 0,
        "missing_altitude": 0,
        "missing_yaw": 0,
        "fallback_altitude": 0,
        "fallback_yaw": 0,
        "dropped_low_altitude": 0,
    }
    rows: List[Dict[str, object]] = []

    for path in iter_image_files(base, recursive=recursive, extensions=extensions):
        stats["files_seen"] += 1
        telemetry, sources = extract_telemetry(str(path))
        if sources["xmp"]:
            stats["xmp_source"] += 1
        if sources["exif"]:
            stats["exif_source"] += 1
        if sources.get("gs_json"):
            stats["gs_json_source"] += 1

        row, reason = normalize_row(path.name, telemetry, require_telemetry=require_telemetry)
        if row is None:
            if reason == "missing_latlon":
                stats["missing_latlon"] += 1
            elif reason == "missing_altitude":
                stats["missing_altitude"] += 1
            elif reason == "missing_yaw":
                stats["missing_yaw"] += 1
            continue

        if _safe_float(telemetry.get("RelativeAltitude")) is None and _safe_float(telemetry.get("AbsoluteAltitude")) is None:
            stats["fallback_altitude"] += 1
        if _safe_float(telemetry.get("FlightYawDegree")) is None and _safe_float(telemetry.get("GimbalYawDegree")) is None:
            stats["fallback_yaw"] += 1

        if min_altitude_m is not None and float(row["Altitude"]) < float(min_altitude_m):
            stats["dropped_low_altitude"] += 1
            continue

        rows.append(row)

    if sort_by == "datetime":
        rows.sort(key=lambda r: (0, r["Timestamp"], r["Image"]) if r.get("Timestamp") else (1, r["Image"], r["Image"]))
    else:
        rows.sort(key=lambda r: str(r["Image"]))

    stats["rows_written"] = len(rows)
    return rows, stats


def write_metadata_to_csv(
    rows: List[Dict[str, object]],
    output_file: str,
    include_timestamp: bool = True,
) -> bool:
    """Write normalized telemetry rows to CSV."""
    if not rows:
        logger.warning("No metadata rows to write to CSV")
        return False

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Image",
        "Latitude",
        "Longitude",
        "Altitude",
        "Degrees_Clockwise_from_North",
    ]
    if include_timestamp:
        fieldnames.append("Timestamp")

    try:
        with output_path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                out_row = {k: row.get(k, "") for k in fieldnames}
                writer.writerow(out_row)
        logger.info(f"Metadata written to {output_file}")
        return True
    except IOError as exc:
        logger.error(f"Failed to write CSV file {output_file}: {exc}")
        return False


def _parse_extensions(ext_text: str) -> Tuple[str, ...]:
    parts = [p.strip().lower() for p in ext_text.split(",") if p.strip()]
    if not parts:
        return DEFAULT_EXTENSIONS
    normalized = []
    for p in parts:
        normalized.append(p if p.startswith(".") else f".{p}")
    return tuple(sorted(set(normalized)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract DJI XMP/EXIF telemetry and write mapping CSV"
    )
    parser.add_argument("image_folder", help="Folder containing DJI images")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path (default: <image_folder>.csv)",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated extensions to include (e.g. .jpg,.jpeg,.dng)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subfolders",
    )
    parser.add_argument(
        "--sort-by",
        choices=["datetime", "filename"],
        default="datetime",
        help="Output row ordering",
    )
    parser.add_argument(
        "--no-timestamp-column",
        action="store_true",
        help="Do not include Timestamp column in CSV",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any input images were skipped due to missing lat/lon",
    )
    parser.add_argument(
        "--require-telemetry",
        action="store_true",
        help="Reject rows whose altitude or heading would be silently defaulted to 0 "
             "(i.e. neither XMP, EXIF, nor sidecar JSON provided them).",
    )
    parser.add_argument(
        "--min-altitude",
        type=float,
        default=None,
        help="Reject rows with altitude below this threshold (meters).",
    )
    parser.add_argument(
        "--min-altitude-ft",
        type=float,
        default=None,
        help="Same as --min-altitude but specified in feet. Overrides --min-altitude.",
    )
    args = parser.parse_args()

    folder_path = args.image_folder
    output_file = args.output_csv if args.output_csv else f"{folder_path}.csv"
    extensions = _parse_extensions(args.extensions)

    min_altitude_m = args.min_altitude
    if args.min_altitude_ft is not None:
        min_altitude_m = args.min_altitude_ft * 0.3048

    rows, stats = process_folder(
        folder_path=folder_path,
        recursive=bool(args.recursive),
        extensions=extensions,
        sort_by=args.sort_by,
        require_telemetry=bool(args.require_telemetry),
        min_altitude_m=min_altitude_m,
    )

    logger.info(
        "Telemetry summary: files=%d rows=%d xmp=%d exif=%d gs_json=%d "
        "missing_latlon=%d missing_altitude=%d missing_yaw=%d "
        "fallback_altitude=%d fallback_yaw=%d dropped_low_altitude=%d",
        stats["files_seen"],
        stats["rows_written"],
        stats["xmp_source"],
        stats["exif_source"],
        stats["gs_json_source"],
        stats["missing_latlon"],
        stats["missing_altitude"],
        stats["missing_yaw"],
        stats["fallback_altitude"],
        stats["fallback_yaw"],
        stats["dropped_low_altitude"],
    )

    if args.strict and stats["missing_latlon"] > 0:
        logger.error(
            "Strict mode enabled and %d images were skipped for missing lat/lon",
            stats["missing_latlon"],
        )
        raise SystemExit(1)

    ok = write_metadata_to_csv(
        rows,
        output_file,
        include_timestamp=not args.no_timestamp_column,
    )
    if ok:
        print(f"Telemetry CSV written: {output_file}")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
