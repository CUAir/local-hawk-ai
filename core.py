from communication.work_client import WorkClient
from models.classifiers import ImageNet
from models.detectors import MaskRCNN
from vision.detectors.abstract_detector import AbstractDetector
from vision.classifiers.abstract_classifier import AbstractClassifier
from constructs.classification import Classification, LabelType
from constructs.roi import ROI
import argparse
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
from communication.intsys_gs_api import MapCommandHandler, ResultStore, EXPORT_DIR, notify_sse
import requests
import io
import base64
import json
from utils.helper import print_green, print_red, print_yellow
import time
import threading
import logging
from constructs.image_types import Base64Image, LabelTypes, ImageMeta, GeoLocation, CandidateImage
from constructs.projection import GroundProjector


# Keep color only for section headers; all other logs are plain text.
header = print_green

logger = logging.getLogger(__name__)


def _setup_file_logging() -> Path:
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = logs_dir / f"local_hawk_ai_{date_str}.log"
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

# Mapping pipeline constants (mirrors hawk-ai/main.py)
MAPPING_SESSION_DIR = Path(__file__).parent / "mapping" / "current_session"
MAPPING_OUTPUT_DIR  = Path(__file__).parent / "mapping"
MAPPING_CSV_PATH    = MAPPING_SESSION_DIR / "metadata.csv"
IDLE_MAPPING_TIMEOUT_SECONDS = 20
IDLE_MAPPING_POLL_SECONDS = 1


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
    def __init__(self, work_client : WorkClient, mapper : Mapper, result_store: ResultStore, autopilot_host: str = None, autopilot_port: int = None, result_interval_seconds: float = 10.0):
        header("\n[vision] Initializing Work Client")
        self.work_client = work_client
        # print("Getting target attributes")
        # self.target_attr = self.work_client.get_target_attributes()
        header("[vision] Initializing Mapper")
        self.mapper = mapper
        self.result_store = result_store
        # Autopilot configuration
        self.autopilot_host = autopilot_host
        self.autopilot_port = autopilot_port
        self.autopilot_url = None
        if autopilot_host and autopilot_port:
            self.autopilot_url = f"http://{autopilot_host}:{autopilot_port}/target"
        # incremental id for autopilot messages
        self._autopilot_id = 0
        # projector for ground coordinates
        self._projector = GroundProjector()

        self.result_interval_seconds = max(1.0, float(result_interval_seconds))
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

    def _result_scheduler_loop(self):
        while True:
            remaining = int(self.result_interval_seconds)
            while remaining > 0:
                # Log every 5 seconds, then every second in the last 5 seconds.
                if remaining <= 5 or remaining % 5 == 0:
                    print_yellow(f"[scheduler] Next cloud poll / autopilot send in {remaining}s")
                time.sleep(1)
                remaining -= 1

            # First: poll cloud once for new best images. Do not hold the send lock
            # while polling (poll may perform I/O and filesystem writes).
            try:
                self._poll_cloud_once()
            except Exception as e:
                print_yellow(f"[scheduler] Cloud poll failed: {e}")

            # Then: attempt to send results to autopilot (non-blocking acquire)
            if not self._send_lock.acquire(blocking=False):
                print_yellow("[scheduler] send_to_autopilot() already running; skipping this tick")
                continue
            try:
                print_green("\n[scheduler] Sending current results to autopilot")
                try:
                    self.send_to_autopilot()
                except Exception as e:
                    print_red(f"[scheduler] send_to_autopilot() failed: {e}")
            finally:
                self._send_lock.release()

            # loop repeats after interval

    def _build_candidate_from_entry(self, assignment: dict, roi: ROI, classification: Classification, meta_filename: str = None, base64_override: str = None) -> CandidateImage:
        """Construct a CandidateImage (with Base64Image.source.meta) suitable for GroundProjector.project()."""
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
                meta = ImageMeta(location=GeoLocation(lat=float(lat), lon=float(lon), alt=float(alt or 0.0)), heading=float(heading))
        except Exception:
            meta = None

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
        return cand

    def send_to_autopilot(self):
        """Select best entries and POST payloads to the configured autopilot endpoint."""
        if not self.autopilot_url:
            print_yellow("[autopilot] Autopilot URL not configured; skipping send")
            return

        def _process_entry(entry, target_type_str):
            if entry is None:
                print_yellow(f"[autopilot] No {target_type_str} entry to send")
                return
            try:
                assignment = entry[0]
                roi = entry[1]
                classification = entry[2]
                meta_fn = entry[5] if len(entry) > 5 else None

                cand = None
                try:
                    cand = self._build_candidate_from_entry(assignment, roi, classification, meta_filename=meta_fn)
                except Exception as e:
                    print_red(f"[autopilot] Failed to build candidate for projection: {e}")
                    cand = None

                lat = None; lon = None
                if cand is not None:
                    try:
                        proj = self._projector.project(cand)
                        if proj:
                            lat = proj.lat
                            lon = proj.lon
                    except Exception as e:
                        print_red(f"[autopilot] Projection error: {e}")

                if lat is None or lon is None:
                    print_red(f"[autopilot] Could not determine lat/lon for {target_type_str}; skipping send")
                    return

                payload = {
                    "lat": float(lat),
                    "lng": float(lon),
                    "target_type": target_type_str,
                    "id": int(self._autopilot_id),
                }
                print("PAYLOAD:", payload)
                try:
                    resp = requests.post(self.autopilot_url, json=payload, timeout=5)
                    if 200 <= resp.status_code < 300:
                        print(f"[autopilot] Sent {target_type_str} -> {self.autopilot_url} (id={self._autopilot_id}, status={resp.status_code})")
                        self._autopilot_id += 1
                    else:
                        print_red(f"[autopilot] Autopilot rejected payload (status={resp.status_code})")
                except Exception as e:
                    print_red(f"[autopilot] Failed to POST to autopilot: {e}")
            except Exception as e:
                print_red(f"[autopilot] Unexpected error processing entry: {e}")

        try:
            m_entry = self.result_store.get_mannequin() if self.result_store else None
            if m_entry:
                _process_entry(m_entry, "person")
        except Exception:
            pass

        try:
            t_entry = self.result_store.get_tent() if self.result_store else None
            if t_entry:
                _process_entry(t_entry, "tent")
        except Exception:
            pass

    def _cloud_poller_loop(self):
        """Deprecated: polling loop replaced by scheduler. Keep for compatibility."""
        # Keep the old loop available but not used; new scheduler calls
        # `_poll_cloud_once()` on its interval instead.
        while True:
            try:
                self._poll_cloud_once()
            except Exception as e:
                print_yellow(f"[poller] Unexpected polling error: {e}")
            time.sleep(self.result_interval_seconds)

    def _poll_cloud_once(self):
        """Perform a single poll of the cloud for mannequin and tent best images.

        This is extracted from the previous `_cloud_poller_loop` to allow the
        scheduler to poll once and then proceed to send results to autopilot.
        """
        # Mannequin
        try:
            assign, roi, clf = self.work_client.get_mannequin_image()
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
                    ts = int(time.time() * 1000)
                    label_name = 'mannequin'
                    aid = assign.get('id') if assign else 'noid'
                    full_fn = EXPORT_DIR / f"full_{label_name}_{aid}_{ts}.jpg"
                    roi_fn = EXPORT_DIR / f"roi_{label_name}_{aid}_{ts}.jpg"
                    # Save ROI crop as both roi and full (no full image available)
                    try:
                        import os as _os
                        import io as _io
                        buf = _io.BytesIO()
                        roi.roi.save(buf, format='JPEG')
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
                        roi.roi.save(buf2, format='JPEG')
                        data2 = buf2.getvalue()
                        tmp_roi = str(roi_fn) + '.tmp'
                        with open(tmp_roi, 'wb') as _f2:
                            _f2.write(data2)
                            _f2.flush()
                            _os.fsync(_f2.fileno())
                        _os.replace(tmp_roi, str(roi_fn))
                    except Exception:
                        pass
                    meta = {
                        "timestamp": ts,
                        "label": label_name,
                        "assignment_id": aid,
                        "assignment": assign,
                        "model_source": "cloud_pull",
                        "gemini_reason": None,
                        "score": float(clf.label[1]) if clf is not None else 0.0,
                        "full_image": str(full_fn.name),
                        "roi_image": str(roi_fn.name),
                        "bbox": list(roi.top_left) + list(roi.bottom_right),
                        "pushed": True,
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
            assign, roi, clf = self.work_client.get_tent_image()
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
                        ts = int(time.time() * 1000)
                        label_name = 'tent'
                        aid = assign.get('id') if assign else 'noid'
                        full_fn = EXPORT_DIR / f"full_{label_name}_{aid}_{ts}.jpg"
                        roi_fn = EXPORT_DIR / f"roi_{label_name}_{aid}_{ts}.jpg"
                        try:
                            import os as _os
                            import io as _io
                            buf = _io.BytesIO()
                            roi.roi.save(buf, format='JPEG')
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
                            roi.roi.save(buf2, format='JPEG')
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
                            with open(tmp_meta, 'w') as _mf:
                                json.dump({
                                    "timestamp": ts,
                                    "label": label_name,
                                    "assignment_id": aid,
                                    "assignment": assign,
                                    "model_source": "cloud_pull",
                                    "gemini_reason": None,
                                    "score": float(clf.label[1]) if clf is not None else 0.0,
                                    "full_image": str(full_fn.name),
                                    "roi_image": str(roi_fn.name),
                                    "bbox": list(roi.top_left) + list(roi.bottom_right),
                                    "pushed": True,
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
        time.sleep(1)
        print_yellow("[worker] Uploading image + running backup detection")
        self.run_model()

        print("[worker] Task cycle complete ========\n")
        time.sleep(1) # sleep 1 second, blocking


    

    # Request image from imaging ground server via work_client.py
    def request_image(self):
        self.assignment = None
        self.image = None
        while True:
            self.assignment, metadata = self.work_client.get_image_assignment()
            if self.assignment == None:
                continue
            image = self.work_client.get_image(metadata["endpoint"])
            break
        self.image = image
        logger.info("Image received from GS — id=%s endpoint=%s", metadata.get('id'), metadata.get('endpoint'))
        self.mapper.mark_image_received()

        # Export the raw ground-station pull (no processing yet) to EXPORT_DIR
        try:
            ts = int(time.time() * 1000)
            aid = self.assignment.get('id') if self.assignment else 'noid'
            full_fn = EXPORT_DIR / f"full_gs_{aid}_{ts}.jpg"
            try:
                self.image.save(str(full_fn), format="JPEG")
            except Exception:
                pass

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

    # Perform autonomous detection and classification
    def run_model(self):
        self.gd_backup()
        try:
            response = self.work_client.send_image(self.image, self.assignment)
            if 200 <= response.status_code < 300:
                print(f"[cloud] Image upload accepted (status={response.status_code})")
            else:
                print_red(f"[cloud] Image upload failed (status={response.status_code})")
        except Exception as e:
            print_red(f"[cloud] Image upload failed: {e}")
    
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
                    roi.roi.save(str(roi_fn), format="JPEG")
                except Exception:
                    pass
                # Write metadata sidecar JSON
                try:
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
                        "bbox": list(roi.top_left) + list(roi.bottom_right),
                            # Mark that this is a GD backup candidate (not a cloud push)
                            "pushed": False,
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
                    roi.roi.save(str(roi_fn), format="JPEG")
                except Exception:
                    pass
                try:
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
                        "bbox": list(roi.top_left) + list(roi.bottom_right),
                        # Mark that this is a GD backup candidate (not a cloud push)
                        "pushed": False,
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


def start_mapping_server(mapper: Mapper, result_store: ResultStore, port=8080):
    header(f"\n[mapping] Map HTTP server started on port {port}")

    # Set the mapper and result_store in the handler class
    MapCommandHandler.mapper = mapper
    MapCommandHandler.result_store = result_store

    # Create and start the HTTP server
    server = ThreadingHTTPServer(('0.0.0.0', port), MapCommandHandler)

    try:
        server.serve_forever()
    except Exception as e:
        print_red(f"Error in map HTTP server: {e}")

def worker_loop(work_client: WorkClient, mapper: Mapper, result_store: ResultStore, autopilot_host: str = None, autopilot_port: int = None, result_interval_seconds: float = 10.0):
    header("\n[worker] Starting worker loop")
    worker = VisionClient(work_client, mapper, result_store, autopilot_host, autopilot_port, result_interval_seconds)
    while True:
        try:
            worker.run_task()
        except Exception as e:
            print_red(f"[worker] Unhandled worker error: {e}")


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

def main(
    gs_ip_address: str,
    cs_ip_address: str,
    map_server_port: int = 8080,
    result_interval_seconds: float = 10.0,
    map_idle_timeout: float = IDLE_MAPPING_TIMEOUT_SECONDS,
    autopilot_host: str = None,
    autopilot_port: int = None,
    mapping_only: bool = False,
    enable_map_idle_trigger: bool = False,
):
    log_path = _setup_file_logging()
    logger.info("Local Hawk-AI client started — gs=%s cs=%s log=%s", gs_ip_address, cs_ip_address, log_path)
    header(
        f"\n[startup] GS={gs_ip_address}, CS={cs_ip_address}, map_port={map_server_port}, "
        f"send_interval={result_interval_seconds}s, map_idle_timeout={map_idle_timeout}s, autopilot={autopilot_host}:{autopilot_port}"
    )
    # Create worker(s) with detector and classifier
    work_client = WorkClient(gs_ip_address, cs_ip_address)
    mapper = Mapper(work_client)
    result_store = ResultStore()

    # Initialize mapping session directory (wipes stale data, creates clean dirs)
    _reset_session()

    # Start mapping HTTP server in a background thread so it runs concurrently
    threading.Thread(target=start_mapping_server, args=(mapper, result_store, map_server_port), daemon=True).start()

    # Idle mapping monitor is opt-in; default is explicit-trigger-only mapping.
    if enable_map_idle_trigger:
        threading.Thread(
            target=idle_mapping_monitor_loop,
            args=(mapper, map_idle_timeout),
            daemon=True,
        ).start()
    else:
        print_yellow("[mapping] Auto-trigger disabled; waiting for explicit trigger_mapping requests")

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

    bound = _wait_for_port('127.0.0.1', map_server_port, timeout=3.0)
    if not bound:
        print_red(f"[startup] Warning: mapping server did not bind port {map_server_port} within timeout")

    # Run workers in the main process concurrently with the server thread
    # unless we are only testing mapping trigger/server behavior.
    if mapping_only:
        print_yellow("[startup] Mapping-only mode enabled: worker loop disabled")
        while True:
            time.sleep(60)
    worker_loop(work_client, mapper, result_store, autopilot_host, autopilot_port, result_interval_seconds)
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
    parser.add_argument('--csip', type=str, default="34.106.160.143:8000", help="Specify cloud server custom IP address")
    parser.add_argument('--map-port', type=int, default=8080, help="Port for the map command HTTP server")
    parser.add_argument('--interval-seconds', type=float, default=20.0, help="Run send_result() every F seconds")
    parser.add_argument('--autopilot-host', type=str, default=None, help="Autopilot host/IP to POST target payloads to")
    parser.add_argument('--autopilot-port', type=int, default="8001", help="Autopilot port to POST target payloads to")
    parser.add_argument('--map-idle-timeout', type=float, default=IDLE_MAPPING_TIMEOUT_SECONDS,
                        help="Seconds of ingest idle time before mapping auto-triggers (0 to disable)")
    parser.add_argument('--mapping-only', action='store_true',
                        help="Run only map server + mapping trigger logic (disable worker loop logs)")
    parser.add_argument('--enable-map-idle-trigger', action='store_true',
                        help="Enable automatic idle-time mapping trigger (disabled by default)")

    args = parser.parse_args()

    if args.local:
        gs_ip_address = "127.0.0.1:9000"
        cs_ip_address = "127.0.0.1:8000"
    else:
        gs_ip_address = args.gsip
        cs_ip_address = args.csip

    main(
        gs_ip_address,
        cs_ip_address,
        args.map_port,
        args.interval_seconds,
        args.map_idle_timeout,
        args.autopilot_host,
        args.autopilot_port,
        args.mapping_only,
        args.enable_map_idle_trigger,
    )
