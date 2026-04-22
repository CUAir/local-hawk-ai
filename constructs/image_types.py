from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math
import random
from enum import Enum

# a poorly named file, TODO: change the name
# Data classes ----------------------------------------------------

class LabelTypes(Enum):
    TENT = "tent"
    MANNEQUIN = "mannequin"
    UNKNOWN = "unknown"

    
@dataclass
class GeoLocation:
    """A WGS-84 coordinate with optional altitude."""
    lat: float
    lon: float
    alt: float = 0.0

    def distance_to(self, other: "GeoLocation") -> float:
        """Haversine distance in metres between two points on Earth."""
        R = 6_371_000  # Earth radius (m)
        phi1, phi2 = math.radians(self.lat), math.radians(other.lat)
        dphi = math.radians(other.lat - self.lat)
        dlam = math.radians(other.lon - self.lon)
        a = (math.sin(dphi / 2) ** 2
             + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class ImageMeta:
    """Metadata for a single image file, optionally enriched with GPS."""
    location: Optional[GeoLocation] = None
    heading: float = 0.0          # compass heading (degrees, 0 = North)
    timestamp: float = 0.0
    gimbal_pitch: float = 0.0
    gimbal_roll: float = 0.0
    has_real_geo: bool = False     # True when parsed from _gs.json


@dataclass
class Base64Image:
    id : int
    base64_image : str
    meta: Optional[ImageMeta] = None
    assignment: Optional[dict] = None  # raw assignment object from GS, returned as-is on GET

@dataclass
class CandidateImage:
    bbox : List[int]    # bounding boxes in x1 y1 x2 y2 format
    score : int         # score from GroundingDINO
    source : Base64Image   # base 64 image
    label : LabelTypes = LabelTypes.UNKNOWN        # phrased used for GroundingDINO

    def __str__ (self):
        return f"bbox: {self.bbox} | score: {self.score} | label: {self.label.value}"

@dataclass
class ObjectCluster:
    """A group of Candidates believed to represent the same physical object
    across multiple images (only meaningful in geo mode)."""
    candidates: List[CandidateImage] = field(default_factory=list)
    cluster_id: str = field(
        default_factory=lambda: f"obj_{random.randint(1000, 9999)}"
    )
    verified_label: str = "unknown"
    verified_conf: float = 0.0

    @property
    def center(self) -> GeoLocation:
        locs = [c.world_loc for c in self.candidates if c.world_loc]
        if not locs:
            return GeoLocation(0, 0)
        return GeoLocation(
            lat=sum(l.lat for l in locs) / len(locs),
            lon=sum(l.lon for l in locs) / len(locs),
        )