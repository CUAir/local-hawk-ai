"""Nadir geotagging helpers for detection candidates.

Faithful to the reference model in hawk-ai: 90° default horizontal FOV,
square-pixel focal length from image width, perfect nadir, flat ground,
origin.alt as AGL, yaw/heading rotation into east/north, then inverse haversine.
Gimbal pitch/roll are unused.

Copied from hawk-ai/constructs/geotagging.py to perform geolocation tagging
locally after hawk-ai performs Gemini classification.
"""
from __future__ import annotations

import base64
import math
from io import BytesIO
from typing import Optional, Sequence, Tuple

from PIL import Image

from constructs.image_types import CandidateImage, GeoLocation

EARTH_R_M = 6371000.0
DEFAULT_HFOV_DEG = 90.0


def bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    """Floating-point center of an xyxy box."""
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def geotag_pixel(
    origin: GeoLocation,
    heading_deg: float,
    px: float,
    py: float,
    image_width: float,
    image_height: float,
    hfov_deg: float = DEFAULT_HFOV_DEG,
) -> GeoLocation:
    """Project a pixel to WGS-84 lat/lon under the reference nadir model.

    Args:
        origin: Drone/aircraft GPS position with altitude (AGL)
        heading_deg: Aircraft heading in degrees clockwise from north
        px, py: Pixel coordinates to project
        image_width, image_height: Image dimensions in pixels
        hfov_deg: Horizontal field of view in degrees (default 90°)

    Returns:
        GeoLocation with projected lat/lon
    """
    hfov_rad = math.radians(hfov_deg)
    fx = (image_width / 2.0) / math.tan(hfov_rad / 2.0)
    x_cam = (px - image_width / 2.0) / fx
    y_cam = (image_height / 2.0 - py) / fx

    altitude = origin.alt
    east_body = altitude * x_cam
    north_body = altitude * y_cam

    # Rotate by heading (yaw) from body frame to Earth frame
    yaw_rad = math.radians(heading_deg)
    cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
    east_m = cos_yaw * east_body + sin_yaw * north_body
    north_m = -sin_yaw * east_body + cos_yaw * north_body

    distance = math.sqrt(east_m * east_m + north_m * north_m)
    bearing = math.atan2(east_m, north_m)
    lat, lon = _inverse_haversine(
        math.radians(origin.lat),
        math.radians(origin.lon),
        distance,
        bearing,
    )
    return GeoLocation(lat=lat, lon=lon)


def geotag_candidate(
    candidate: CandidateImage,
    hfov_deg: float = DEFAULT_HFOV_DEG,
) -> Optional[GeoLocation]:
    """Geotag a candidate from source metadata and bbox center, or None.

    Args:
        candidate: CandidateImage with source.meta containing GPS and heading
        hfov_deg: Horizontal field of view in degrees

    Returns:
        GeoLocation with projected ground coordinates, or None if data is invalid
    """
    source = getattr(candidate, "source", None)
    meta = getattr(source, "meta", None) if source is not None else None
    location = getattr(meta, "location", None) if meta is not None else None
    if source is None or meta is None or location is None:
        return None

    heading = getattr(meta, "heading", 0.0)
    if not _all_finite(
        location.lat,
        location.lon,
        location.alt,
        heading,
        hfov_deg,
    ):
        return None
    if location.alt <= 0.0:
        return None
    if not _valid_hfov(hfov_deg):
        return None

    bbox = getattr(candidate, "bbox", None)
    if not _valid_bbox(bbox):
        return None
    cx, cy = bbox_center(bbox)
    if not _all_finite(cx, cy):
        return None

    size = _image_size_from_b64(getattr(source, "base64_image", None))
    if size is None:
        return None
    image_width, image_height = size

    try:
        result = geotag_pixel(
            location,
            heading_deg=float(heading),
            px=cx,
            py=cy,
            image_width=float(image_width),
            image_height=float(image_height),
            hfov_deg=float(hfov_deg),
        )
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None

    if not _all_finite(result.lat, result.lon):
        return None
    return result


def _inverse_haversine(
    lat_rad: float,
    lon_rad: float,
    dist_m: float,
    bearing_rad: float,
) -> Tuple[float, float]:
    """Compute new lat/lon given origin, distance, and bearing using spherical Earth."""
    radius = EARTH_R_M
    new_lat = math.asin(
        math.sin(lat_rad) * math.cos(dist_m / radius)
        + math.cos(lat_rad) * math.sin(dist_m / radius) * math.cos(bearing_rad)
    )
    new_lon = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(dist_m / radius) * math.cos(lat_rad),
        math.cos(dist_m / radius) - math.sin(lat_rad) * math.sin(new_lat),
    )
    return math.degrees(new_lat), math.degrees(new_lon)


def _all_finite(*values: object) -> bool:
    for value in values:
        try:
            if not math.isfinite(float(value)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _valid_hfov(hfov_deg: float) -> bool:
    try:
        hfov = float(hfov_deg)
    except (TypeError, ValueError):
        return False
    return math.isfinite(hfov) and 0.0 < hfov < 180.0


def _valid_bbox(bbox: object) -> bool:
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)):
        return False
    if len(bbox) != 4:
        return False
    return _all_finite(*bbox)


def _image_size_from_b64(b64_image: object) -> Optional[Tuple[int, int]]:
    """Extract image dimensions from base64-encoded image data."""
    if not isinstance(b64_image, (str, bytes)) or not b64_image:
        return None
    # Strip data URL prefix if present
    if isinstance(b64_image, str) and "," in b64_image:
        b64_image = b64_image.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64_image, validate=False)
    except (TypeError, ValueError):
        return None
    if not raw:
        return None
    try:
        with Image.open(BytesIO(raw)) as img:
            width, height = img.size
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height
