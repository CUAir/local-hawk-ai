"""
ground_projection.py
====================
Projects a CandidateImage bounding box onto the ground plane,
returning the estimated GPS coordinates of the detected object.

Only dependency on the rest of the codebase: the data classes in
dtypes.py (GeoLocation, ImageMeta, Base64Image, CandidateImage).
"""

import math
import base64
import struct
import zlib
from dataclasses import dataclass
from typing import Optional

# Import the shared data classes
from constructs.image_types import CandidateImage, GeoLocation


@dataclass
class GroundProjection:
    """Result of projecting a bounding-box detection onto the ground plane."""
    lat: float
    lon: float

    def __repr__(self) -> str:
        return f"GroundProjection(lat={self.lat:.7f}, lon={self.lon:.7f})"

    def to_geo_location(self) -> GeoLocation:
        return GeoLocation(lat=self.lat, lon=self.lon, alt=0.0)


class GroundProjector:
    """
    Projects a CandidateImage's bounding box onto the ground plane using a
    simplified nadir flat-earth model (matches detect.py's _project_to_ground).

    The image dimensions are read directly from the base64-encoded image data
    so no filesystem access or OpenCV is needed.

    Usage
    -----
        projector = GroundProjector()
        result = projector.project(candidate)
        print(result.lat, result.lon)
    """

    # FOV scale factors from detect.py (horizontal wider than vertical)
    _FOV_X = 1.5
    _FOV_Y = 1.0
    _METRES_PER_DEGREE = 111_111.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(self, candidate: CandidateImage) -> Optional[GroundProjection]:
        """
        Input
        -----
        candidate : CandidateImage
            Must have:
            - candidate.bbox            — [x1, y1, x2, y2] in pixels
            - candidate.source.meta.location.lat/lon/alt
            - candidate.source.base64_image — base64-encoded JPEG or PNG

        Output
        ------
        GroundProjection(lat, lon)  — estimated ground position of the object,
        or None if the candidate is missing required telemetry.
        """
        meta = candidate.source.meta
        if meta is None or meta.location is None:
            return None

        drone = meta.location
        if drone.alt <= 0:
            return None

        img_w, img_h = self._image_dimensions(candidate.source.base64_image)
        if img_w == 0 or img_h == 0:
            return None

        x1, y1, x2, y2 = candidate.bbox

        # Pixel centre normalised to UV space: (-0.5 … +0.5)
        u = (x1 + x2) / 2 / img_w - 0.5
        v = 0.5 - (y1 + y2) / 2 / img_h   # flip Y (image top → geo south)

        # Ground offset in metres
        dx_m = u * drone.alt * self._FOV_X
        dy_m = v * drone.alt * self._FOV_Y

        # Convert metres → decimal degrees
        lat = drone.lat + dy_m / self._METRES_PER_DEGREE
        lon = drone.lon + dx_m / (
            self._METRES_PER_DEGREE * math.cos(math.radians(drone.lat))
        )

        return GroundProjection(lat=lat, lon=lon)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _image_dimensions(b64_string: str) -> tuple[int, int]:
        """
        Return (width, height) of a base64-encoded JPEG or PNG without
        decoding the full pixel data (reads only the necessary header bytes).

        Returns (0, 0) on any parse failure.
        """
        try:
            # Strip data-URL prefix if present ("data:image/jpeg;base64,...")
            if "," in b64_string:
                b64_string = b64_string.split(",", 1)[1]

            raw = base64.b64decode(b64_string)

            # ---- JPEG ----
            # SOI marker: FF D8
            # Scan for SOFx frames (FF C0 / FF C1 / FF C2) which hold dimensions
            if raw[:2] == b"\xff\xd8":
                i = 2
                while i < len(raw) - 1:
                    if raw[i] != 0xFF:
                        break
                    marker = raw[i + 1]
                    i += 2
                    if marker in (0xC0, 0xC1, 0xC2):
                        # SOF: 2-byte length, 1-byte precision, 2-byte h, 2-byte w
                        h = struct.unpack(">H", raw[i + 3: i + 5])[0]
                        w = struct.unpack(">H", raw[i + 5: i + 7])[0]
                        return w, h
                    # Skip this segment
                    seg_len = struct.unpack(">H", raw[i: i + 2])[0]
                    i += seg_len

            # ---- PNG ----
            # PNG signature: 8 bytes, then IHDR chunk (4 len + 4 type + 13 data)
            # Width at bytes 16-19, height at 20-23
            if raw[:8] == b"\x89PNG\r\n\x1a\n":
                w = struct.unpack(">I", raw[16:20])[0]
                h = struct.unpack(">I", raw[20:24])[0]
                return w, h

        except Exception:
            pass

        return 0, 0


# ---------------------------------------------------------------------------
# Convenience function — single call, no class instantiation needed
# ---------------------------------------------------------------------------

def project_candidate(candidate: CandidateImage) -> Optional[GroundProjection]:
    """
    One-liner wrapper around GroundProjector.project().

    Input  : CandidateImage (with base64 image + GPS telemetry)
    Output : GroundProjection(lat, lon) or None
    """
    return GroundProjector().project(candidate)


# ---------------------------------------------------------------------------
# Smoke test using the sample JSON from the assignment
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from dtypes import Base64Image, ImageMeta, GeoLocation, LabelTypes

    # ---------- build the object tree from the sample JSON ----------
    sample = {
        "id": 8,
        "assignment": {
            "image": {
                "telemetry": {
                    "gps": {
                        "latitude":  42.26044875023213,
                        "longitude": -76.12668339532011,
                    },
                    "altitude":  250.60493895061634,
                    "planeYaw":  135.93516235638106,
                    "gimOrt": {
                        "pitch": -44.88883683340723,
                        "roll":  -7.0132955083057436,
                    },
                }
            }
        },
        # truncated for brevity — use the real base64 string in practice
        "base64_image": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
    }

    tel = sample["assignment"]["image"]["telemetry"]
    gps = tel["gps"]

    meta = ImageMeta(
        location=GeoLocation(
            lat=gps["latitude"],
            lon=gps["longitude"],
            alt=tel["altitude"],
        ),
        heading=tel["planeYaw"],
        gimbal_pitch=tel["gimOrt"]["pitch"],
        gimbal_roll=tel["gimOrt"]["roll"],
        has_real_geo=True,
    )

    source = Base64Image(
        id=sample["id"],
        base64_image=sample["base64_image"],
        meta=meta,
        assignment=sample["assignment"],
    )

    # Dummy bbox — replace with real DINO output
    candidate = CandidateImage(
        bbox=[400, 300, 500, 400],
        score=0.85,
        source=source,
        label=LabelTypes.MANNEQUIN,
    )

    result = project_candidate(candidate)
    if result:
        print(f"Projected ground position:")
        print(f"  lat = {result.lat:.7f}")
        print(f"  lon = {result.lon:.7f}")
    else:
        print("Projection failed — check telemetry / image data.")