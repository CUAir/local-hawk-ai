from communication.work_client import WorkClient
from models.classifiers import ImageNet
from models.detectors import MaskRCNN
from vision.detectors.abstract_detector import AbstractDetector
from vision.classifiers.abstract_classifier import AbstractClassifier
from constructs.classification import Classification, LabelType
from constructs.roi import ROI
import argparse
import math
import PIL.Image as Image
from multiprocessing import Process
import socket
import os
import csv
import shutil
import threading
from pathlib import Path
from datetime import datetime
from http.server import ThreadingHTTPServer
from communication.intsys_gs_api import MapCommandHandler, FrontendHandler, ResultStore, EXPORT_DIR, notify_sse, _parse_label
import requests
import io
import base64
import json
from utils.helper import print_green, print_red, print_yellow
import time
import threading
import logging
from dataclasses import asdict
from constructs.image_types import Base64Image, LabelTypes, ImageMeta, GeoLocation, CandidateImage
from constructs.geotagging import geotag_candidate


# Keep color only for section headers; all other logs are plain text.
header = print_green

logger = logging.getLogger(__name__)


# --- Autopilot send: ground-test altitude fallback ---------------------------
# geotag_candidate() returns None when location.alt <= 0 (constructs/geotagging.py),
# because the nadir projection scales ground offset linearly with altitude: at 0 m
# every pixel in the frame collapses onto the aircraft's own position. Meanwhile
# mini-plane-system reports MAVLink relative_alt (pixhawk/pixhawk.py), which sits at
# ~0 on the bench, so the manual "Send to Autopilot" button could never fire on the
# ground.
#
# When the reported AGL is non-positive we substitute a nominal altitude so the send
# still goes through. The resulting lat/lon is a TEST coordinate, not a real fix --
# its offset from the aircraft is fabricated from this constant. Set
# HAWKAI_FALLBACK_ALT_M=0 to restore the strict on-ground refusal.
DEFAULT_FALLBACK_AGL_M = 30.0


def _fallback_agl_m() -> float:
    """Configured stand-in AGL in metres, or 0.0 when the fallback is disabled."""
    try:
        value = float(os.environ.get("HAWKAI_FALLBACK_ALT_M", DEFAULT_FALLBACK_AGL_M))
    except (TypeError, ValueError):
        return DEFAULT_FALLBACK_AGL_M
    return value if math.isfinite(value) and value > 0.0 else 0.0


def _resolve_agl(alt_raw):
    """Return (altitude_m, substituted) for the autopilot projection.

    `substituted` is True when the telemetry altitude was unusable (non-positive,
    missing or non-finite) and the fallback stood in for it. When the fallback is
    disabled the unusable value is passed through so geotag_candidate() still
    refuses the projection.
    """
    try:
        alt = float(alt_raw)
    except (TypeError, ValueError):
        alt = 0.0
    if not math.isfinite(alt):
        alt = 0.0
    if alt > 0.0:
        return alt, False
    fallback = _fallback_agl_m()
    if fallback <= 0.0:
        return alt, False
    return fallback, True


def _telemetry_fields(tel: dict) -> tuple:
    """Extract (lat, lon, alt, heading) from a gs-backend telemetry block.

    gs-backend nests GPS under "gps" and names the heading "planeYaw". Readers
    that assumed flat "latitude"/"longitude"/"yaw" keys silently got nothing for
    every image, so this is the single definition of where those values live.
    Used by _build_image_meta() (the cloud-upload meta) -- _compute_target_geolocation
    below keeps its own inline extraction as-is rather than being rewired to share
    this, to avoid touching a function the currently-pulled codebase already owns.

    Returns (None, None, None, None) when lat/lon are absent.
    """
    tel = tel or {}
    gps = tel.get('gps') or {}
    lat = gps.get('latitude')
    lon = gps.get('longitude')
    if lat is None or lon is None:
        return None, None, None, None
    heading = tel.get('planeYaw')
    if heading is None:
        heading = tel.get('yaw')
    return lat, lon, tel.get('altitude'), (heading if heading is not None else 0.0)


def _compute_target_geolocation(assignment: dict, bbox: list, image_path) -> tuple:
    """Compute geotagged target lat/lon from assignment telemetry and bbox.

    Returns (target_lat, target_lon) or (None, None) if computation fails.
    """
    try:
        if not bbox or len(bbox) != 4:
            return None, None
        img_data = (assignment or {}).get('image') or {}
        tel = img_data.get('telemetry') or {}
        gps = tel.get('gps') or {}
        lat = gps.get('latitude')
        lon = gps.get('longitude')
        alt = tel.get('altitude')
        heading = tel.get('planeYaw') or tel.get('yaw') or 0.0
        if lat is None or lon is None or alt is None:
            return None, None
        # Load image and compute geotag
        with open(image_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        meta = ImageMeta(
            location=GeoLocation(lat=float(lat), lon=float(lon), alt=float(alt)),
            heading=float(heading)
        )
        source = Base64Image(id=0, base64_image=b64, meta=meta)
        candidate = CandidateImage(bbox=bbox, score=0, source=source, label=LabelTypes.UNKNOWN)
        geo = geotag_candidate(candidate)
        if geo:
            return geo.lat, geo.lon
    except Exception as e:
        logger.debug(f"Geotag computation failed: {e}")
    return None, None


def _ensure_export_dir() -> bool:
    """Ensure EXPORT_DIR exists, even if deleted mid-runtime."""
    try:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print_red(f"[export] Failed to create export directory: {e}")
        return False


# Path of this run's log file, published module-level so the log-panel source
# resolver can reach it (the filename embeds the start timestamp).
CURRENT_LOG_PATH: Path = None


def _setup_file_logging() -> Path:
    global CURRENT_LOG_PATH
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = logs_dir / f"local_hawk_ai_{date_str}.log"
    CURRENT_LOG_PATH = log_path
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    return log_path


GD_model = None
_GD_MODEL_INIT_ATTEMPTED = False
_GD_MODEL_LOCK = threading.Lock()


def _get_gd_model():
    """
    Lazily initialize GroundingDINO once.
    This avoids heavy model/bootstrap work (and HF network calls) when running
    mapping-only mode, where gd_backup is never used.
    """
    global GD_model, _GD_MODEL_INIT_ATTEMPTED
    if GD_model is not None:
        return GD_model
    if _GD_MODEL_INIT_ATTEMPTED:
        return None
    with _GD_MODEL_LOCK:
        if GD_model is not None:
            return GD_model
        if _GD_MODEL_INIT_ATTEMPTED:
            return None
        _GD_MODEL_INIT_ATTEMPTED = True
        try:
            from constructs.detection import GDDetection
            GD_model = GDDetection()
            return GD_model
        except Exception as e:
            print_red(f"[gd_backup] GroundingDINO init failed: {e}")
            GD_model = None
            return None

MAX_BOX_FRACTION = 0.5

# Stand down the local GroundingDINO pass once the cloud has produced a result
# for BOTH labels within this many seconds - it's genuinely redundant at that
# point, and GD's single detection pass can't selectively skip just one label.
GD_SKIP_IF_CLOUD_FRESH_S = 120.0

# Mapping pipeline constants (mirrors hawk-ai/main.py)
MAPPING_SESSION_DIR = Path(__file__).parent / "mapping" / "current_session"
MAPPING_OUTPUT_DIR  = Path(__file__).parent / "mapping" / "outputs"
MAPPING_CSV_PATH    = MAPPING_SESSION_DIR / "metadata.csv"
IDLE_MAPPING_TIMEOUT_SECONDS = 20
IDLE_MAPPING_POLL_SECONDS = 1
# How often to ask the cloud whether its mapping run has finished. Cloud stitches
# take minutes, so this is deliberately slower than the idle monitor's 1s tick.
CLOUD_MAP_POLL_SECONDS = 15


# Mapping Helper Functions (mirrors hawk-ai/main.py, adapted for PIL images) --------

def _reset_session() -> None:
    """Clear the current session folder and write a fresh CSV header."""
    shutil.rmtree(MAPPING_SESSION_DIR, ignore_errors=True)
    (MAPPING_SESSION_DIR / "images").mkdir(parents=True, exist_ok=True)
    with open(MAPPING_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Image", "Latitude", "Longitude", "Altitude", "Degrees_Clockwise_from_North"
        ])
        writer.writeheader()


def _count_csv_rows() -> int:
    """Return number of data rows in the metadata CSV (0 if file missing)."""
    try:
        with open(MAPPING_CSV_PATH, "r") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0


def _save_image_for_mapping_local(image: Image.Image, metadata: dict) -> bool:
    """
    Save a PIL image + GPS telemetry to the mapping session folder and append
    a row to metadata.csv. Silently returns False if GPS data is absent.

    Field mapping: metadata["telemetry"]["yaw"] -> Degrees_Clockwise_from_North
    (same meaning as hawk-ai's image.meta.heading)
    """
    telemetry = (metadata or {}).get("telemetry")
    if not telemetry:
        return False
    lat = telemetry.get("latitude")
    lon = telemetry.get("longitude")
    if lat is None or lon is None:
        return False
    try:
        image_id = metadata["id"]
        img_filename = f"{image_id}.jpg"
        img_path = MAPPING_SESSION_DIR / "images" / img_filename
        image.save(str(img_path), format="JPEG")
        alt = telemetry.get("altitude", 0)
        yaw = telemetry.get("yaw", 0)
        with open(MAPPING_CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Image", "Latitude", "Longitude", "Altitude", "Degrees_Clockwise_from_North"
            ])
            writer.writerow({
                "Image": img_filename,
                "Latitude": lat,
                "Longitude": lon,
                "Altitude": alt,
                "Degrees_Clockwise_from_North": yaw,
            })
        print(f"[mapping] Saved {img_filename} to session")
        return True
    except Exception as e:
        print_red(f"[mapping] Failed to save image {metadata.get('id')}: {e}")
        return False


def _run_pipeline_local(mapper) -> None:
    """
    Run GpsSiftPipeline in a background thread (local-hawk-ai is synchronous,
    no asyncio event loop). Called via threading.Thread from mapper.trigger_pipeline().
    Mirrors hawk-ai's _run_pipeline + _run_mapping logic.
    """
    from mapping.main_gps_sift import GpsSiftPipeline

    mapper.mapping_running = True
    n_images = _count_csv_rows()

    if n_images < 2:
        print_yellow(f"[mapping] Only {n_images} image(s) in session — skipping pipeline")
        _reset_session()
        mapper.mapping_running = False
        return

    print_green(f"[mapping] Trigger received — running pipeline on {n_images} images...")
    try:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        final_out = str(MAPPING_OUTPUT_DIR / f"map_{date_str}.jpg")
        pipeline = GpsSiftPipeline(output_dir=str(MAPPING_OUTPUT_DIR))
        raw_path = pipeline.run(
            str(MAPPING_SESSION_DIR / "images"),
            str(MAPPING_CSV_PATH),
        )
        Path(raw_path).rename(final_out)
        mapper.mapping_result = final_out
        print(f"[mapping] Done → {final_out}")
    except Exception as e:
        print_red(f"[mapping] Pipeline error: {e}")
    finally:
        _reset_session()
        mapper.mapping_running = False


class Mapper:
    def __init__(self, work_client : WorkClient):
        self.work_client = work_client
        self.mapping = False
        if not os.path.exists('images.csv'):
            with open('images.csv', 'w') as f:
                f.write('Image,Latitude,Longitude,Altitude,Degrees_Clockwise_from_North\n')

        # Hawk-ai-style mapping state
        self.mapping_running = False
        self.mapping_result = None
        self.last_image_received_ts = time.time()
        self.last_auto_trigger_ts = 0.0

        # Cloud mapping mirror state. cloud_mapping_result is the local path of
        # the most recently downloaded cloud map; _seen_cloud_results tracks the
        # cloud-side paths we have already pulled so a steady-state poll does not
        # re-download the same map every tick.
        self.cloud_mapping_result = None
        self._seen_cloud_results = set()

    def trigger_pipeline(self):
        """Trigger the hawk-ai GpsSiftPipeline in a background thread."""
        if self.mapping_running:
            print_yellow("[mapping] Pipeline already running — ignoring trigger")
            return
        threading.Thread(target=_run_pipeline_local, args=(self,), daemon=True).start()
        print("[mapping] Pipeline triggered in background thread")

    def mark_image_received(self):
        """Record timestamp of the latest received image."""
        self.last_image_received_ts = time.time()

    def poll_cloud_map_once(self):
        """Pull the cloud orthomosaic down once a cloud run has finished.

        Saved alongside the locally-produced maps in MAPPING_OUTPUT_DIR under the
        cloud's own filename with `_cloud` appended, so `map_<date>.jpg` on the
        cloud lands here as `map_<date>_cloud.jpg` and sorts next to the local
        map from the same flight.

        Only downloads once the cloud reports `running: false` — mapping_result
        is set before the run flips that flag back, so pulling while it is still
        true can fetch the *previous* flight's map.
        """
        cloud_status = self.work_client.get_cloud_mapping_status()
        if not cloud_status:
            return

        if cloud_status.get("running"):
            return

        remote_path = cloud_status.get("last_result")
        if not remote_path or remote_path in self._seen_cloud_results:
            return

        remote_name = Path(remote_path).name
        if not remote_name:
            return

        stem = Path(remote_name).stem
        suffix = Path(remote_name).suffix or ".jpg"
        dest = MAPPING_OUTPUT_DIR / f"{stem}_cloud{suffix}"

        # Mark as seen up front: on a persistent failure we would otherwise
        # retry this same map on every tick for the rest of the flight.
        self._seen_cloud_results.add(remote_path)

        if dest.exists():
            print(f"[mapping] Cloud map already present locally → {dest}")
            self.cloud_mapping_result = str(dest)
            return

        print_green(f"[mapping] Cloud run finished ({remote_name}) — downloading...")
        if self.work_client.download_cloud_mapping_result(dest):
            self.cloud_mapping_result = str(dest)
        else:
            print_yellow(f"[mapping] Could not download cloud map {remote_name}")

    def maybe_trigger_pipeline_on_idle(self, timeout_seconds: float = IDLE_MAPPING_TIMEOUT_SECONDS):
        """
        Auto-trigger mapping when image ingest has been idle for long enough.
        Requires at least 2 images in session and no active pipeline.
        """
        if self.mapping_running:
            return
        if time.time() - self.last_image_received_ts < timeout_seconds:
            return
        n_images = _count_csv_rows()
        if n_images < 2:
            return
        if time.time() - self.last_auto_trigger_ts < timeout_seconds:
            return
        print(
            f"[mapping] Idle for {timeout_seconds}s with {n_images} images; auto-triggering pipeline"
        )
        self.last_auto_trigger_ts = time.time()
        self.trigger_pipeline()

class VisionClient:
    def __init__(self, work_client : WorkClient, mapper : Mapper, result_store: ResultStore, autopilot_host: str = None, result_interval_seconds: float = 10.0, classify_every_n: int = 1):
        header("\n[vision] Initializing Work Client")
        self.work_client = work_client
        # print("Getting target attributes")
        # self.target_attr = self.work_client.get_target_attributes()
        header("[vision] Initializing Mapper")
        self.mapper = mapper
        self.result_store = result_store
        # Autopilot configuration
        self.autopilot_host = autopilot_host
        self.autopilot_url = None
        if autopilot_host:
            # Project-Emu's mavproxy_cuairapi exposes POST /targets_set, not /target
            # (mavproxy_cuairapi/views/targets.py) — no /target route exists anywhere.
            self.autopilot_url = f"http://{autopilot_host}/targets_set"
        # incremental id for autopilot messages
        self._autopilot_id = 1

        self.result_interval_seconds = max(1.0, float(result_interval_seconds))

        # Classification gate: every image is still pulled, exported and handed to
        # mapping, but only every Nth is run through GroundingDINO and uploaded to
        # the cloud. A count, not a duration - the assignment's image.timestamp is
        # unreliable (real exports carry 1970 dates), and a wall-clock gate would
        # give unpredictable ground spacing while the loop drains a backlog.
        self.classify_every_n = max(1, int(classify_every_n))
        self._cycle_count = 0

        self._send_lock = threading.Lock()
        # Single background thread: poll cloud for results, then send to autopilot
        self._result_scheduler_thread = threading.Thread(
            target=self._result_scheduler_loop,
            daemon=True,
        )
        self._result_scheduler_thread.start()

        # Track signatures of cloud detections we've already seen to
        # avoid duplicate processing (assignment id, bbox, score).
        self._seen_cloud_signatures = set()

        # GD backup: best detected candidate per label from the most recent image
        self._gd_best_mannequin: tuple = None  # (assignment, ROI, Classification)
        self._gd_best_tent: tuple = None        # (assignment, ROI, Classification)

        # "Session start" (T0): receive-timestamp of the first image pulled while
        # EXPORT_DIR was empty. Re-established any time the folder empties again
        # (e.g. after a "clear_exports" command), since request_image() re-checks
        # disk state on every pull rather than latching a one-time flag.
        self._session_start_ts: int = None
        self._session_start_lock = threading.Lock()

    def _result_scheduler_loop(self):
        while True:
            remaining = int(self.result_interval_seconds)
            while remaining > 0:
                # Log every 5 seconds, then every second in the last 5 seconds.
                if remaining <= 5 or remaining % 5 == 0:
                    print_yellow(f"[scheduler] Next cloud poll in {remaining}s")
                time.sleep(1)
                remaining -= 1

            # Poll cloud once for new best images. Autopilot sends are no longer
            # automatic here — they only happen when a user presses "Send to
            # Autopilot" on a dashboard card (see send_meta_to_autopilot()).
            try:
                self._poll_cloud_once()
            except Exception as e:
                print_yellow(f"[scheduler] Cloud poll failed: {e}")

            # loop repeats after interval

    def _build_candidate_from_entry(self, assignment: dict, roi: ROI, classification: Classification, meta_filename: str = None, base64_override: str = None):
        """Build a CandidateImage suitable for geotag_candidate().

        Returns (candidate, alt_substituted), where alt_substituted is True when the
        assignment's telemetry altitude was unusable and _resolve_agl() supplied a
        stand-in so an on-ground manual send can still be projected.
        """
        # Try to get base64 full image from meta file
        b64 = None
        try:
            if base64_override:
                b64 = base64_override
            elif meta_filename:
                mfpath = EXPORT_DIR / meta_filename
                if mfpath.exists():
                    with open(mfpath, 'r') as _mf:
                        try:
                            m = json.load(_mf)
                        except Exception:
                            m = {}
                    full_fname = m.get('full_image')
                    if full_fname and (EXPORT_DIR / full_fname).exists():
                        with open(EXPORT_DIR / full_fname, 'rb') as _fimg:
                            import base64 as _b64
                            b64 = 'data:image/jpeg;base64,' + _b64.b64encode(_fimg.read()).decode('utf-8')
        except Exception:
            b64 = None

        # Fallback: fetch from GS via assignment
        if not b64 and assignment and isinstance(assignment, dict):
            try:
                img_endpoint = None
                if isinstance(assignment.get('image'), dict):
                    img_endpoint = assignment.get('image').get('imageUrl') or assignment.get('image').get('localImageUrl')
                if img_endpoint and getattr(self, 'mapper', None) and getattr(self.mapper, 'work_client', None):
                    fetched = self.mapper.work_client.get_image(img_endpoint)
                    if fetched is not None:
                        buf = io.BytesIO()
                        fetched.save(buf, format='JPEG')
                        import base64 as _b64
                        b64 = 'data:image/jpeg;base64,' + _b64.b64encode(buf.getvalue()).decode('utf-8')
            except Exception:
                b64 = None

        # Build ImageMeta from assignment telemetry (best-effort)
        meta = None
        alt_substituted = False
        try:
            lat = None; lon = None; alt = None; heading = 0.0
            if assignment and isinstance(assignment, dict):
                img = assignment.get('image') or {}
                tel = img.get('telemetry') or img.get('meta') or {}
                gps = tel.get('gps') or {}
                lat = gps.get('latitude') or tel.get('latitude') or tel.get('lat')
                lon = gps.get('longitude') or tel.get('longitude') or tel.get('lon')
                alt = tel.get('altitude') or gps.get('altitude') or tel.get('alt')
                heading = tel.get('yaw') or tel.get('planeYaw') or 0.0
            if lat is not None and lon is not None:
                agl, alt_substituted = _resolve_agl(alt)
                meta = ImageMeta(location=GeoLocation(lat=float(lat), lon=float(lon), alt=agl), heading=float(heading))
        except Exception:
            meta = None
            alt_substituted = False

        base_src = Base64Image(id=assignment.get('id') if isinstance(assignment, dict) else 0, base64_image=b64 or '', meta=meta, assignment=assignment)
        bbox = []
        if roi is not None:
            try:
                bbox = [int(roi.top_left[0]), int(roi.top_left[1]), int(roi.bottom_right[0]), int(roi.bottom_right[1])]
            except Exception:
                bbox = []
        score = 0.0
        try:
            score = float(classification.label[1]) if classification is not None else 0.0
        except Exception:
            score = 0.0
        cand = CandidateImage(bbox=bbox, score=score, source=base_src, label=LabelTypes.UNKNOWN)
        return cand, alt_substituted

    def send_entry_to_autopilot(self, entry, target_type_str: str) -> dict:
        """Build a candidate from `entry`, project it, and POST it to the autopilot endpoint.

        `entry` is (assignment, roi, classification, model_source, gemini_reason, meta_filename).
        Returns {"status": "success"|"error", "message": str} for callers (e.g. the manual
        "Send to Autopilot" button) that need to relay the outcome back to a user.
        """
        if not self.autopilot_url:
            msg = "Autopilot URL not configured; skipping send"
            print_yellow(f"[autopilot] {msg}")
            return {"status": "error", "message": msg}

        if entry is None:
            msg = f"No {target_type_str} entry to send"
            print_yellow(f"[autopilot] {msg}")
            return {"status": "error", "message": msg}

        try:
            assignment = entry[0]
            roi = entry[1]
            classification = entry[2]
            meta_fn = entry[5] if len(entry) > 5 else None

            cand = None
            alt_substituted = False
            try:
                cand, alt_substituted = self._build_candidate_from_entry(assignment, roi, classification, meta_filename=meta_fn)
            except Exception as e:
                print_red(f"[autopilot] Failed to build candidate for projection: {e}")
                cand = None

            if alt_substituted:
                print_yellow(
                    f"[autopilot] Telemetry AGL was <= 0 (aircraft on the ground?); "
                    f"projecting with a {_fallback_agl_m():.1f} m fallback - the "
                    f"coordinate below is for testing, not a real target fix"
                )

            lat = None; lon = None
            if cand is not None:
                try:
                    geo = geotag_candidate(cand)
                    if geo:
                        lat = geo.lat
                        lon = geo.lon
                except Exception as e:
                    print_red(f"[autopilot] Geotagging error: {e}")

            if lat is None or lon is None:
                msg = f"Could not determine lat/lon for {target_type_str}; skipping send"
                print_red(f"[autopilot] {msg}")
                return {"status": "error", "message": msg}

            # Shape required by Project-Emu's pathplanning.update_targets(), which
            # reads target['geotag']['gpsLocation']['latitude'/'longitude'] — a flat
            # {lat, lng, ...} body raises a KeyError there (caught, returns 500).
            payload = {
                "id": int(self._autopilot_id),
                "geotag": {
                    "id": int(self._autopilot_id),
                    "gpsLocation": {
                        "latitude": float(lat),
                        "longitude": float(lon),
                    },
                },
                "target_type": target_type_str.upper(),
            }
            print("PAYLOAD:", payload)
            try:
                # PIPELINE_AUDIT.md F34: a retrying wrapper here can double-send a
                # target if the first response was merely lost, not un-sent -
                # the same zero-retry rule the capture-upload path already
                # correctly follows for non-idempotent sends. This also drops
                # the up-to-11s _send_lock hold the retry loop caused; a lost
                # send just means the operator (this is a manual, button-
                # triggered send) clicks again.
                resp = requests.post(self.autopilot_url, json=payload, timeout=5)
            except Exception as e:
                msg = f"Failed to POST to autopilot: {e}"
                print_red(f"[autopilot] {msg}")
                return {"status": "error", "message": msg}

            if resp is not None and 200 <= resp.status_code < 300:
                print(f"[autopilot] Sent {target_type_str} -> {self.autopilot_url} (id={self._autopilot_id}, status={resp.status_code})")
                self._autopilot_id += 1
                note = " [TEST ALT]" if alt_substituted else ""
                return {"status": "success", "message": f"Sent {target_type_str} to autopilot (lat={lat:.6f}, lon={lon:.6f}){note}"}
            else:
                status = resp.status_code if resp is not None else "no response"
                msg = f"Autopilot rejected payload (status={status})"
                print_red(f"[autopilot] {msg}")
                return {"status": "error", "message": msg}
        except Exception as e:
            msg = f"Unexpected error processing entry: {e}"
            print_red(f"[autopilot] {msg}")
            return {"status": "error", "message": msg}

    def send_meta_to_autopilot(self, meta_filename: str) -> dict:
        """Send one specific exported detection (by meta filename) to autopilot.

        This is the entry point for the dashboard's manual "Send to Autopilot"
        button, since autopilot sends are no longer automatic. Rebuilds the
        ROI/Classification from the meta JSON + its full image on disk, the
        same way ResultStore.rebuild_from_disk() does when restoring state.
        """
        safe_name = Path(meta_filename or "").name
        if not safe_name.startswith("meta_") or not safe_name.endswith(".json"):
            return {"status": "error", "message": "Invalid meta_filename"}

        meta_path = EXPORT_DIR / safe_name
        if not meta_path.exists():
            return {"status": "error", "message": f"Meta file not found: {safe_name}"}

        with self._send_lock:
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                label = _parse_label(meta.get("label"))
                if label == LabelType.TENT:
                    target_type_str = "tent"
                elif label == LabelType.MANNEQUIN:
                    target_type_str = "person"
                else:
                    return {"status": "error", "message": f"Unknown label in meta file: {meta.get('label')}"}

                bbox = meta.get("bbox") or []
                full_image_name = meta.get("full_image")
                if len(bbox) != 4 or not full_image_name:
                    return {"status": "error", "message": "Meta file missing bbox or full_image"}

                full_path = EXPORT_DIR / full_image_name
                if not full_path.exists():
                    return {"status": "error", "message": f"Full image not found: {full_image_name}"}

                x1, y1, x2, y2 = [int(v) for v in bbox]
                full_image = Image.open(full_path).convert("RGB")
                width, height = full_image.size
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))
                if x2 <= x1 or y2 <= y1:
                    return {"status": "error", "message": "Degenerate bbox in meta file"}

                roi = ROI(roi=full_image.crop((x1, y1, x2, y2)), top_left=(x1, y1), bottom_right=(x2, y2))
                classification = Classification(label=label, number_conf=float(meta.get("score", 0.0)))
                entry = (
                    meta.get("assignment"),
                    roi,
                    classification,
                    meta.get("model_source", ""),
                    meta.get("gemini_reason"),
                    safe_name,
                )
                return self.send_entry_to_autopilot(entry, target_type_str)
            except Exception as e:
                msg = f"Failed to send meta file to autopilot: {e}"
                print_red(f"[autopilot] {msg}")
                return {"status": "error", "message": msg}

    def _poll_cloud_once(self):
        """Perform a single poll of the cloud for mannequin and tent best images.
        """
        # Mannequin
        try:
            assign, roi, clf, gemini_reason, model_source, full_image = self.work_client.get_mannequin_image()
            if assign is not None and roi is not None and clf is not None:
                # Detect duplicate: same assignment id, bbox, and confidence
                try:
                    aid = assign.get('id') if assign else 'noid'
                    bbox = list(roi.top_left) + list(roi.bottom_right) if roi is not None else []
                    score = float(clf.label[1]) if clf is not None else 0.0
                    sig = (str(aid), tuple(bbox), round(float(score), 6))
                except Exception:
                    sig = None
                is_dup = (sig is not None and sig in self._seen_cloud_signatures)
                if is_dup:
                    # Duplicate detected — notify frontend and skip updating result_store
                    try:
                        notify_sse('duplicate', {'label': 'mannequin', 'assignment_id': aid, 'bbox': bbox, 'score': score, 'message': 'duplicate detection — skipped autopilot send'})
                    except Exception:
                        pass
                # proceed with persisting and recording only if not duplicate
                if not is_dup:
                    if not _ensure_export_dir():
                        print_red("[poller] Skipping mannequin export write: export directory unavailable")
                        return
                    ts = int(time.time() * 1000)
                    label_name = 'mannequin'
                    aid = assign.get('id') if assign else 'noid'
                    full_fn = EXPORT_DIR / f"full_{label_name}_{aid}_{ts}.jpg"
                    roi_fn = EXPORT_DIR / f"roi_{label_name}_{aid}_{ts}.jpg"
                    # Save the true full image (as returned by the cloud) so the dashboard
                    # can show it with the bbox overlay, plus the ROI crop separately.
                    try:
                        import os as _os
                        import io as _io
                        buf = _io.BytesIO()
                        (full_image if full_image is not None else roi.image).save(buf, format='JPEG')
                        data = buf.getvalue()
                        tmp_full = str(full_fn) + '.tmp'
                        with open(tmp_full, 'wb') as _f:
                            _f.write(data)
                            _f.flush()
                            _os.fsync(_f.fileno())
                        _os.replace(tmp_full, str(full_fn))
                    except Exception:
                        pass
                    try:
                        import os as _os
                        import io as _io
                        buf2 = _io.BytesIO()
                        roi.image.save(buf2, format='JPEG')
                        data2 = buf2.getvalue()
                        tmp_roi = str(roi_fn) + '.tmp'
                        with open(tmp_roi, 'wb') as _f2:
                            _f2.write(data2)
                            _f2.flush()
                            _os.fsync(_f2.fileno())
                        _os.replace(tmp_roi, str(roi_fn))
                    except Exception:
                        pass
                    session_start_ts, since_session_start_ms = self._since_session_start(ts)
                    bbox = list(roi.top_left) + list(roi.bottom_right)
                    target_lat, target_lon = _compute_target_geolocation(assign, bbox, full_fn)
                    meta = {
                        "timestamp": ts,
                        "label": label_name,
                        "assignment_id": aid,
                        "assignment": assign,
                        "model_source": model_source or "cloud_pull",
                        "gemini_reason": gemini_reason,
                        "score": float(clf.label[1]) if clf is not None else 0.0,
                        "full_image": str(full_fn.name),
                        "roi_image": str(roi_fn.name),
                        "bbox": bbox,
                        "pushed": True,
                        "session_start_ts": session_start_ts,
                        "since_session_start_ms": since_session_start_ms,
                        "target_lat": target_lat,
                        "target_lon": target_lon,
                    }
                    try:
                        import os as _os
                        meta_fn = EXPORT_DIR / f"meta_{label_name}_{aid}_{ts}.json"
                        tmp_meta = str(meta_fn) + '.tmp'
                        with open(tmp_meta, 'w') as _mf:
                            json.dump(meta, _mf)
                            _mf.flush()
                            _os.fsync(_mf.fileno())
                        _os.replace(tmp_meta, str(meta_fn))
                        mf_name = meta_fn.name
                    except Exception:
                        mf_name = None

                    # Mark signature seen to avoid future duplicates
                    try:
                        if sig is not None:
                            self._seen_cloud_signatures.add(sig)
                    except Exception:
                        pass

                    try:
                        # Update result store and notify frontend of the new cloud pull
                        self.result_store.update(LabelType.MANNEQUIN, assign, roi, clf, "cloud_pull", None, mf_name)
                        print_green("[poller] Updated mannequin from cloud pull")
                        try:
                            # small pause to ensure filesystem visibility after atomic rename
                            time.sleep(0.05)
                            notify_sse('gs_pull', {'label': 'mannequin', 'meta': mf_name})
                        except Exception:
                            pass
                    except Exception as e:
                        print_red(f"[poller] Failed to update mannequin result_store: {e}")
        except Exception as e:
            print_yellow(f"[poller] Mannequin pull error: {e}")

        # Tent
        try:
            assign, roi, clf, gemini_reason, model_source, full_image = self.work_client.get_tent_image()
            if assign is not None and roi is not None and clf is not None:
                # Duplicate detection for tent
                try:
                    aid = assign.get('id') if assign else 'noid'
                    bbox = list(roi.top_left) + list(roi.bottom_right) if roi is not None else []
                    score = float(clf.label[1]) if clf is not None else 0.0
                    sig = (str(aid), tuple(bbox), round(float(score), 6))
                except Exception:
                    sig = None
                is_dup = (sig is not None and sig in self._seen_cloud_signatures)
                if is_dup:
                    try:
                        notify_sse('duplicate', {'label': 'tent', 'assignment_id': aid, 'bbox': bbox, 'score': score, 'message': 'duplicate detection — skipped autopilot send'})
                    except Exception:
                        pass
                if not is_dup:
                    try:
                        if not _ensure_export_dir():
                            print_red("[poller] Skipping tent export write: export directory unavailable")
                            return
                        ts = int(time.time() * 1000)
                        label_name = 'tent'
                        aid = assign.get('id') if assign else 'noid'
                        full_fn = EXPORT_DIR / f"full_{label_name}_{aid}_{ts}.jpg"
                        roi_fn = EXPORT_DIR / f"roi_{label_name}_{aid}_{ts}.jpg"
                        # Save the true full image (as returned by the cloud) so the dashboard
                        # can show it with the bbox overlay, plus the ROI crop separately.
                        try:
                            import os as _os
                            import io as _io
                            buf = _io.BytesIO()
                            (full_image if full_image is not None else roi.image).save(buf, format='JPEG')
                            data = buf.getvalue()
                            tmp_full = str(full_fn) + '.tmp'
                            with open(tmp_full, 'wb') as _f:
                                _f.write(data)
                                _f.flush()
                                _os.fsync(_f.fileno())
                            _os.replace(tmp_full, str(full_fn))
                        except Exception:
                            pass
                        try:
                            import os as _os
                            import io as _io
                            buf2 = _io.BytesIO()
                            roi.image.save(buf2, format='JPEG')
                            data2 = buf2.getvalue()
                            tmp_roi = str(roi_fn) + '.tmp'
                            with open(tmp_roi, 'wb') as _f2:
                                _f2.write(data2)
                                _f2.flush()
                                _os.fsync(_f2.fileno())
                            _os.replace(tmp_roi, str(roi_fn))
                        except Exception:
                            pass
                        try:
                            import os as _os
                            meta_fn = EXPORT_DIR / f"meta_{label_name}_{aid}_{ts}.json"
                            tmp_meta = str(meta_fn) + '.tmp'
                            session_start_ts, since_session_start_ms = self._since_session_start(ts)
                            bbox = list(roi.top_left) + list(roi.bottom_right)
                            target_lat, target_lon = _compute_target_geolocation(assign, bbox, full_fn)
                            with open(tmp_meta, 'w') as _mf:
                                json.dump({
                                    "timestamp": ts,
                                    "label": label_name,
                                    "assignment_id": aid,
                                    "assignment": assign,
                                    "model_source": model_source or "cloud_pull",
                                    "gemini_reason": gemini_reason,
                                    "score": float(clf.label[1]) if clf is not None else 0.0,
                                    "full_image": str(full_fn.name),
                                    "roi_image": str(roi_fn.name),
                                    "bbox": bbox,
                                    "pushed": True,
                                    "session_start_ts": session_start_ts,
                                    "since_session_start_ms": since_session_start_ms,
                                    "target_lat": target_lat,
                                    "target_lon": target_lon,
                                }, _mf)
                                _mf.flush()
                                _os.fsync(_mf.fileno())
                            _os.replace(tmp_meta, str(meta_fn))
                            mf_name = meta_fn.name
                        except Exception:
                            mf_name = None
                    except Exception:
                        mf_name = None

                    # mark signature and update store
                    try:
                        if sig is not None:
                            self._seen_cloud_signatures.add(sig)
                    except Exception:
                        pass

                    try:
                        self.result_store.update(LabelType.TENT, assign, roi, clf, "cloud_pull", None, mf_name)
                        print_green("[poller] Updated tent from cloud pull")
                        try:
                            # small pause to ensure filesystem visibility after atomic rename
                            time.sleep(0.05)
                            notify_sse('gs_pull', {'label': 'tent', 'meta': mf_name})
                        except Exception:
                            pass
                    except Exception as e:
                        print_red(f"[poller] Failed to update tent result_store: {e}")
        except Exception as e:
            print_yellow(f"[poller] Tent pull error: {e}")

    def run_task(self):
        print("\n[worker] Starting task cycle ========")

        print_green("[worker] Requesting image from imaging GS")
        self.request_image()

        # Cloud upload always runs, every image - it's the source of truth and
        # the target can move in/out of frame. Local GD's redundant pass is
        # gated inside run_model() itself (classify_every_n + cloud freshness).
        print_green("[worker] Uploading image to cloud")
        self.run_model()

        print("[worker] Task cycle complete ========\n")


    

    # Request image from imaging ground server via work_client.py
    def request_image(self):
        self.assignment = None
        self.image = None
        self.metadata = None
        while True:
            self.assignment, metadata = self.work_client.get_image_assignment()
            if self.assignment == None:
                continue
            image = self.work_client.get_image(metadata["endpoint"])
            break
        if image is None:
            # PIPELINE_AUDIT.md F11: the code below still writes a meta_gs_*
            # record and fires the gs_pull SSE for this assignment even
            # though there is no image -- that behavior is unchanged here,
            # this just makes the failure visible instead of a silent
            # phantom "last image" update on the dashboard.
            logger.error(
                "get_image() returned None for assignment id=%s endpoint=%s — "
                "export/meta for this pull will reference a missing image.",
                metadata.get('id'), metadata.get('endpoint'),
            )
        self.image = image
        # Held on the instance so run_model()/send_image() can attach GPS to the
        # cloud upload - it used to be a local here, which is why every image
        # reached hawk-ai with meta=None and its mapping session stayed empty.
        self.metadata = metadata
        logger.info("Image received from GS — id=%s endpoint=%s", metadata.get('id'), metadata.get('endpoint'))
        self.mapper.mark_image_received()

        # Export the raw ground-station pull (no processing yet) to EXPORT_DIR
        try:
            if not _ensure_export_dir():
                raise RuntimeError("export directory unavailable")
            ts = int(time.time() * 1000)

            # Establish (or re-establish) T0: the receive-time of the first
            # image pulled while EXPORT_DIR was empty. Re-checked on every
            # pull (not a one-time flag) so it re-baselines after a
            # "clear_exports" wipes the folder.
            try:
                if not any(EXPORT_DIR.iterdir()):
                    with self._session_start_lock:
                        self._session_start_ts = ts
            except Exception:
                pass

            aid = self.assignment.get('id') if self.assignment else 'noid'
            full_fn = EXPORT_DIR / f"full_gs_{aid}_{ts}.jpg"
            try:
                self.image.save(str(full_fn), format="JPEG")
            except Exception as e:
                logger.error("Failed to write %s (image=%r): %s", full_fn, self.image, e)

            # Minimal metadata: timestamp and assignment only (no processing fields)
            try:
                meta = {
                    "timestamp": ts,
                    "assignment_id": aid,
                    "assignment": self.assignment,
                    "model_source": "gs_pull",
                    "full_image": str(full_fn.name),
                    # indicate this is a raw GS pull
                    "pushed": False,
                }
                meta_fn = EXPORT_DIR / f"meta_gs_{aid}_{ts}.json"
                import json as _json
                with open(meta_fn, 'w') as mf:
                    _json.dump(meta, mf)
                # Notify frontend via SSE that a new GS pull arrived
                try:
                    notify_sse('gs_pull', { 'meta': meta_fn.name, 'timestamp': ts })
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

        # Add image to mapping folder
        # self.mapper.add_image(image, metadata)

        # Save to hawk-ai-style mapping session (side effect, does not affect detection flow)
        _save_image_for_mapping_local(image, metadata)

    def _build_image_meta(self) -> dict:
        """Build the `meta` block for the cloud upload from assignment telemetry.

        Returns None when the assignment carried no GPS, which is the only case
        the cloud is entitled to skip for mapping. Built here rather than in
        WorkClient because core imports work_client, not the other way round,
        and `_telemetry_fields` is the single reader for this shape.
        """
        lat, lon, alt, heading = _telemetry_fields((self.metadata or {}).get("telemetry"))
        if lat is None or lon is None:
            return None
        return asdict(ImageMeta(
            location=GeoLocation(lat=float(lat), lon=float(lon), alt=float(alt or 0.0)),
            heading=float(heading),
            has_real_geo=True,
        ))

    # Perform autonomous detection and classification
    def run_model(self):
        # Local GD is a fallback for when the cloud is down or slow, so it stands
        # down once the cloud is confirmed actively producing results for both
        # labels - classify_every_n still caps how often it's even considered,
        # so it doesn't peg Pi CPU/GPU every single tick before the cloud has
        # anything either.
        self._cycle_count += 1
        due_for_local_gd = (self._cycle_count % self.classify_every_n == 0)
        if due_for_local_gd:
            if self.result_store.both_cloud_fresh(GD_SKIP_IF_CLOUD_FRESH_S):
                print_yellow(
                    f"[worker] Skipping local GD - cloud has results for both labels "
                    f"within the last {GD_SKIP_IF_CLOUD_FRESH_S:.0f}s"
                )
            else:
                print_green("[worker] Running local backup detection")
                self.gd_backup()
        try:
            response = self.work_client.send_image(self.image, self.assignment,
                                                   meta=self._build_image_meta())
            if 200 <= response.status_code < 300:
                print(f"[cloud] Image upload accepted (status={response.status_code})")
            else:
                print_red(f"[cloud] Image upload failed (status={response.status_code})")
        except Exception as e:
            print_red(f"[cloud] Image upload failed: {e}")
    
    def _since_session_start(self, ts: int):
        """Return (session_start_ts, since_session_start_ms) for a classification
        made at time `ts`, or (None, None) if T0 hasn't been established yet
        (no image has been pulled into an empty EXPORT_DIR this session)."""
        with self._session_start_lock:
            t0 = self._session_start_ts
        if t0 is None:
            return None, None
        return t0, ts - t0

    def gd_backup(self):
        """Run GroundingDINO on the current image and cache the highest-scoring
        candidate for each of MANNEQUIN and TENT as a fallback in case the
        cloud server has no result (204)."""
        if self.image is None or self.assignment is None:
            return
        model = _get_gd_model()
        if model is None:
            # If model init fails, skip GD backup and continue cloud upload flow.
            return

        import io as _io
        import base64 as _b64

        # Convert PIL image -> Base64Image so detect_candidates can consume it
        buf = _io.BytesIO()
        self.image.save(buf, format="JPEG")
        img_b64_str = _b64.b64encode(buf.getvalue()).decode("utf-8")
        base64_image = Base64Image(
            id=self.assignment["id"],
            base64_image=img_b64_str,
            assignment=self.assignment,
        )

        try:
            candidates = model.detect_candidates(
                base64_image,
                max_box_fraction=MAX_BOX_FRACTION,
                save_file=False,
            )
        except Exception as e:
            print_red(f"[gd_backup] Detection failed: {e}")
            return

        print_green(f"[gd_backup] Detection complete: {len(candidates)} candidate(s)")

        best_mannequin = None
        best_tent = None

        for candidate in candidates:
            if candidate.label == LabelTypes.MANNEQUIN:
                if best_mannequin is None or candidate.score > best_mannequin.score:
                    best_mannequin = candidate
            elif candidate.label == LabelTypes.TENT:
                if best_tent is None or candidate.score > best_tent.score:
                    best_tent = candidate

        def _candidate_to_roi_classification(candidate, label_type):
            """Convert a CandidateImage into (ROI, Classification) using the source PIL image."""
            x1, y1, x2, y2 = candidate.bbox
            source_pil = self.image
            cropped = source_pil.crop((x1, y1, x2, y2))
            roi = ROI(roi=cropped, top_left=(x1, y1), bottom_right=(x2, y2))
            classification = Classification(label=label_type, number_conf=candidate.score)
            return roi, classification

        if best_mannequin is not None:
            roi, clf = _candidate_to_roi_classification(best_mannequin, LabelType.MANNEQUIN)
            self._gd_best_mannequin = (self.assignment, roi, clf)
            print(f"[gd_backup] Cached mannequin candidate (score={best_mannequin.score:.3f})")
            # Export cached GD backup to disk for inspection / frontend
            try:
                if not _ensure_export_dir():
                    raise RuntimeError("export directory unavailable")
                ts = int(time.time() * 1000)
                label_name = "mannequin"
                aid = self.assignment.get('id') if self.assignment else 'noid'
                full_fn = EXPORT_DIR / f"full_{label_name}_{aid}_{ts}.jpg"
                roi_fn = EXPORT_DIR / f"roi_{label_name}_{aid}_{ts}.jpg"
                # Save full image and roi crop
                try:
                    self.image.save(str(full_fn), format="JPEG")
                except Exception:
                    pass
                try:
                    roi.image.save(str(roi_fn), format="JPEG")
                except Exception:
                    pass
                # Write metadata sidecar JSON
                try:
                    session_start_ts, since_session_start_ms = self._since_session_start(ts)
                    bbox = list(roi.top_left) + list(roi.bottom_right)
                    target_lat, target_lon = _compute_target_geolocation(self.assignment, bbox, full_fn)
                    meta = {
                        "timestamp": ts,
                        "label": label_name,
                        "assignment_id": aid,
                        "assignment": self.assignment,
                        "model_source": "gd_backup",
                        "gemini_reason": None,
                        "score": float(best_mannequin.score),
                        "full_image": str(full_fn.name),
                        "roi_image": str(roi_fn.name),
                        "bbox": bbox,
                        "pushed": False,
                        "session_start_ts": session_start_ts,
                        "since_session_start_ms": since_session_start_ms,
                        "target_lat": target_lat,
                        "target_lon": target_lon,
                    }
                    meta_fn = EXPORT_DIR / f"meta_{label_name}_{aid}_{ts}.json"
                    with open(meta_fn, "w") as mf:
                        import json
                        json.dump(meta, mf)
                    mf_name = meta_fn.name
                    try:
                        # Update ResultStore with GD backup entry so it can be used when no cloud pull exists
                        if getattr(self, 'result_store', None):
                            self.result_store.update(LabelType.MANNEQUIN, self.assignment, roi, Classification(label=LabelType.MANNEQUIN, number_conf=float(best_mannequin.score)), 'gd_backup', None, mf_name)
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass
        else:
            print_yellow("[gd_backup] No mannequin candidate found in current image")

        if best_tent is not None:
            roi, clf = _candidate_to_roi_classification(best_tent, LabelType.TENT)
            self._gd_best_tent = (self.assignment, roi, clf)
            print(f"[gd_backup] Cached tent candidate (score={best_tent.score:.3f})")
            # Export cached GD backup to disk for inspection / frontend
            try:
                if not _ensure_export_dir():
                    raise RuntimeError("export directory unavailable")
                ts = int(time.time() * 1000)
                label_name = "tent"
                aid = self.assignment.get('id') if self.assignment else 'noid'
                full_fn = EXPORT_DIR / f"full_{label_name}_{aid}_{ts}.jpg"
                roi_fn = EXPORT_DIR / f"roi_{label_name}_{aid}_{ts}.jpg"
                try:
                    self.image.save(str(full_fn), format="JPEG")
                except Exception:
                    pass
                try:
                    roi.image.save(str(roi_fn), format="JPEG")
                except Exception:
                    pass
                try:
                    session_start_ts, since_session_start_ms = self._since_session_start(ts)
                    bbox = list(roi.top_left) + list(roi.bottom_right)
                    target_lat, target_lon = _compute_target_geolocation(self.assignment, bbox, full_fn)
                    meta = {
                        "timestamp": ts,
                        "label": label_name,
                        "assignment_id": aid,
                        "assignment": self.assignment,
                        "model_source": "gd_backup",
                        "gemini_reason": None,
                        "score": float(best_tent.score),
                        "full_image": str(full_fn.name),
                        "roi_image": str(roi_fn.name),
                        "bbox": bbox,
                        "pushed": False,
                        "session_start_ts": session_start_ts,
                        "since_session_start_ms": since_session_start_ms,
                        "target_lat": target_lat,
                        "target_lon": target_lon,
                    }
                    meta_fn = EXPORT_DIR / f"meta_{label_name}_{aid}_{ts}.json"
                    with open(meta_fn, "w") as mf:
                        import json
                        json.dump(meta, mf)
                    mf_name = meta_fn.name
                    try:
                        if getattr(self, 'result_store', None):
                            self.result_store.update(LabelType.TENT, self.assignment, roi, Classification(label=LabelType.TENT, number_conf=float(best_tent.score)), 'gd_backup', None, mf_name)
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass
        else:
            print_yellow("[gd_backup] No tent candidate found in current image")


def start_server(mapper: Mapper, result_store: ResultStore, port=8080,
                 mps_base_url: str = None, log_sources: dict = None):
    header(f"\n[server] API/command HTTP server started on port {port}")

    # Set the mapper and result_store in the handler class
    MapCommandHandler.mapper = mapper
    MapCommandHandler.result_store = result_store
    MapCommandHandler.mps_base_url = mps_base_url
    MapCommandHandler.log_sources = log_sources or {}

    if mps_base_url:
        header(f"[capture] Capture controls enabled, MPS at {mps_base_url}")
    else:
        print_yellow("[capture] Capture controls disabled (start with --mps to enable)")

    # Create and start the HTTP server
    server = ThreadingHTTPServer(('0.0.0.0', port), MapCommandHandler)

    try:
        server.serve_forever()
    except Exception as e:
        print_red(f"Error in API/command HTTP server: {e}")


def start_frontend_server(port: int, server_port: int):
    header(f"\n[frontend] Frontend HTTP server started on port {port}")

    FrontendHandler.server_port = server_port
    server = ThreadingHTTPServer(('0.0.0.0', port), FrontendHandler)

    try:
        server.serve_forever()
    except Exception as e:
        print_red(f"Error in frontend HTTP server: {e}")

def worker_loop(work_client: WorkClient, mapper: Mapper, result_store: ResultStore, autopilot_host: str = None, result_interval_seconds: float = 10.0, classify_every_n: int = 1):
    header("\n[worker] Starting worker loop")
    worker = VisionClient(work_client, mapper, result_store, autopilot_host, result_interval_seconds, classify_every_n)
    MapCommandHandler.vision_client = worker
    while True:
        try:
            worker.run_task()
        except Exception as e:
            print_red(f"[worker] Unhandled worker error: {e}")
            print("[worker] MOST LIKELY BECAUSE gs-backend isn't running or isn't connected")
            time.sleep(2)


def idle_mapping_monitor_loop(mapper: Mapper, timeout_seconds: float):
    """Background loop for fallback auto-trigger when no images arrive."""
    if timeout_seconds <= 0:
        print_yellow("[mapping] Idle monitor disabled (--map-idle-timeout=0)")
        return
    header(
        f"\n[mapping] Idle monitor started (timeout={timeout_seconds}s)"
    )
    while True:
        try:
            mapper.maybe_trigger_pipeline_on_idle(timeout_seconds)
        except Exception as e:
            print_yellow(f"[mapping] Idle monitor error: {e}")
        time.sleep(IDLE_MAPPING_POLL_SECONDS)

def cloud_map_monitor_loop(mapper: Mapper, interval_seconds: float):
    """Background loop that mirrors finished cloud orthomosaics onto this machine.

    Runs independently of the local mapping pipeline — the cloud may be asked to
    stitch by anyone, so we watch its status rather than assuming we triggered it.
    """
    header(f"\n[mapping] Cloud map monitor started (poll={interval_seconds}s)")
    while True:
        try:
            mapper.poll_cloud_map_once()
        except Exception as e:
            print_yellow(f"[mapping] Cloud map monitor error: {e}")
        time.sleep(interval_seconds)


def _build_log_sources(gs_backend_log: str, mps_address: str = None) -> dict:
    """Allowlist of tailable log files, keyed by id.

    Paths are resolved once here; the handler re-checks at read time. A source
    whose file does not exist yet is still listed (so the tab appears, greyed)
    because gs-backend only creates logs/server.log on first run.
    """
    sources = {}

    if CURRENT_LOG_PATH is not None:
        sources['lhai'] = {
            "label": "local-hawk-ai",
            "kind": "file",
            "path": Path(CURRENT_LOG_PATH).resolve(),
        }

    if gs_backend_log:
        try:
            sources['gsbackend'] = {
                "label": "gs-backend",
                "kind": "file",
                # Not resolve(strict=True): the file may not exist until
                # gs-backend has served its first request.
                "path": Path(gs_backend_log).expanduser().resolve(),
            }
        except Exception as e:
            print_yellow(f"[logs] Ignoring --gs-backend-log ({gs_backend_log}): {e}")

    # The aircraft log is fetched over HTTP, not read from disk. Only offered
    # when --mps is set, so the tab simply does not appear otherwise.
    if mps_address:
        sources['mps'] = {
            "label": "MPS (aircraft)",
            "kind": "proxy",
            "path": None,
        }

    return sources


def main(
    gs_ip_address: str,
    cs_ip_address: str,
    server_port: int = 8080,
    frontend_port: int = 8081,
    result_interval_seconds: float = 10.0,
    map_idle_timeout: float = IDLE_MAPPING_TIMEOUT_SECONDS,
    autopilot_host: str = None,
    mapping_only: bool = False,
    enable_map_idle_trigger: bool = False,
    mps_address: str = None,
    gs_backend_log: str = None,
    classify_every_n: int = 1,
):
    log_path = _setup_file_logging()
    logger.info("Local Hawk-AI client started — gs=%s cs=%s log=%s", gs_ip_address, cs_ip_address, log_path)
    header(
        f"\n[startup] GS={gs_ip_address}, CS={cs_ip_address}, server_port={server_port}, frontend_port={frontend_port}, "
        f"send_interval={result_interval_seconds}s, map_idle_timeout={map_idle_timeout}s, autopilot={autopilot_host}"
    )
    # Create worker(s) with detector and classifier
    work_client = WorkClient(gs_ip_address, cs_ip_address)
    mapper = Mapper(work_client)
    result_store = ResultStore()
    result_store.rebuild_from_disk(EXPORT_DIR)

    # Initialize mapping session directory (wipes stale data, creates clean dirs)
    _reset_session()

    log_sources = _build_log_sources(gs_backend_log, mps_address)

    # Start the API/command HTTP server in a background thread so it runs concurrently
    threading.Thread(
        target=start_server,
        args=(mapper, result_store, server_port, mps_address, log_sources),
        daemon=True,
    ).start()

    # Start the frontend HTTP server in its own background thread
    threading.Thread(target=start_frontend_server, args=(frontend_port, server_port), daemon=True).start()

    # Idle mapping monitor is opt-in; default is explicit-trigger-only mapping.
    if enable_map_idle_trigger:
        threading.Thread(
            target=idle_mapping_monitor_loop,
            args=(mapper, map_idle_timeout),
            daemon=True,
        ).start()
    else:
        print_yellow("[mapping] Auto-trigger disabled; waiting for explicit trigger_mapping requests")

    # Mirror finished cloud orthomosaics into MAPPING_OUTPUT_DIR as <name>_cloud.jpg
    threading.Thread(
        target=cloud_map_monitor_loop,
        args=(mapper, CLOUD_MAP_POLL_SECONDS),
        daemon=True,
    ).start()

    # Wait until the server has bound the port (or timeout)
    def _wait_for_port(host: str, port: int, timeout: float = 3.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection((host, port), timeout=0.5):
                    return True
            except Exception:
                time.sleep(0.05)
        return False

    bound = _wait_for_port('127.0.0.1', server_port, timeout=3.0)
    if not bound:
        print_red(f"[startup] Warning: API/command server did not bind port {server_port} within timeout")

    # Run workers in the main process concurrently with the server thread
    # unless we are only testing mapping trigger/server behavior.
    if mapping_only:
        print_yellow("[startup] Mapping-only mode enabled: worker loop disabled")
        while True:
            time.sleep(60)

    # PIPELINE_AUDIT.md F15: _get_gd_model() otherwise loads lazily on the
    # first classify cycle, putting a cold GroundingDINO load on the critical
    # path of an early image. Warmed in the background so it doesn't block
    # the dashboard/API server from becoming available. Skipped above in
    # mapping_only mode, matching _get_gd_model()'s own reason for being lazy
    # in the first place - gd_backup() is never called there.
    threading.Thread(target=_get_gd_model, daemon=True, name="gd-model-warmup").start()

    worker_loop(work_client, mapper, result_store, autopilot_host, result_interval_seconds, classify_every_n)
    # Create processes
    # mapper_process = Process(target=start_mapping_server, args=(mapper, map_server_port))
    # worker_process1 = Process(target=worker_loop, args=(work_client, mapper))
    # worker_process2 = Process(target=worker_loop, args=(work_client, mapper))

    # Start processes
    # mapper_process.start()
    # worker_process1.start()
    # worker_process2.start()

    # try:
    #     # Wait for processes to complete
    #     # mapper_process.join()
    #     worker_process1.join()
    #     worker_process2.join()
    # except KeyboardInterrupt:
    #     print("Received interrupt, shutting down...", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Intelligent Systems Client")
    parser.add_argument('--local', action='store_true', help="Use local IP address")
    parser.add_argument('--gsip', type=str, default="127.0.0.1:9000", help="Specify ground station custom IP address") # 192.168.1.2:9000"; 10.48.199.45:9000
    parser.add_argument('--csip', type=str, default="34.106.149.232:8000", help="Specify cloud server custom IP address")
    parser.add_argument('--server-port', type=int, default=9080, help="Port for the API/command HTTP server (dashboard API, SSE, exports, mapping/command endpoints)")
    parser.add_argument('--frontend-port', type=int, default=9081, help="Port for the frontend dashboard HTTP server")
    parser.add_argument('--interval-seconds', type=float, default=10.0, help="Run send_result() every F seconds")
    parser.add_argument('--aip', type=str, default="192.168.1.3:8001", help="Autopilot host/IP to POST target payloads to")
    parser.add_argument('--map-idle-timeout', type=float, default=IDLE_MAPPING_TIMEOUT_SECONDS,
                        help="Seconds of ingest idle time before mapping auto-triggers (0 to disable)")
    parser.add_argument('--mapping-only', action='store_true',
                        help="Run only map server + mapping trigger logic (disable worker loop logs)")
    parser.add_argument('--enable-map-idle-trigger', action='store_true',
                        help="Enable automatic idle-time mapping trigger (disabled by default)")
    parser.add_argument('--mps', type=str, default=None,
                        help="MPS (aircraft) address for dashboard capture controls, e.g. 192.168.1.10:8000. "
                             "Omit to disable capture controls entirely.")
    parser.add_argument('--gs-backend-log', type=str,
                        default=str(Path(__file__).resolve().parent.parent / 'gs-backend' / 'logs' / 'server.log'),
                        help="Path to gs-backend's logs/server.log for the log panel "
                             "(default: sibling gs-backend checkout)")
    parser.add_argument('--classify-every-n', type=int, default=1,
                        help="Run local GroundingDINO on only every Nth image (gated further by "
                             "cloud freshness - see both_cloud_fresh). Every image is still pulled, "
                             "exported and fed to mapping, and every image is still uploaded to the "
                             "cloud regardless of this gate. A COUNT, not a duration: with MPS "
                             "capturing every 2s, the default of 2 means local GD runs at most once "
                             "per 4s. Use 1 to allow it every image.")

    args = parser.parse_args()

    if args.local:
        gs_ip_address = "127.0.0.1:9000"
        cs_ip_address = "127.0.0.1:8000"
        a_ip_address = "127.0.0.1:8001"
    else:
        gs_ip_address = args.gsip
        cs_ip_address = args.csip
        a_ip_address = args.aip

    main(
        gs_ip_address,
        cs_ip_address,
        args.server_port,
        args.frontend_port,
        args.interval_seconds,
        args.map_idle_timeout,
        a_ip_address,
        args.mapping_only,
        args.enable_map_idle_trigger,
        args.mps,
        args.gs_backend_log,
        args.classify_every_n,
    )
