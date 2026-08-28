from http.server import BaseHTTPRequestHandler
from urllib.parse import unquote, urlsplit, parse_qs
import json as _json
import json
import os
import io
import base64
import hashlib
import threading
import requests
from utils.helper import print_green, print_red, print_yellow
from constructs.roi import ROI
from constructs.classification import Classification, LabelType
from PIL import Image
import time
import json
from pathlib import Path

# Directory to save exported images pushed by cloud
EXPORT_DIR = Path(__file__).parent.parent / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


def render_frontend_index(hostname: str, server_port: int) -> bytes:
    """Read frontend/index.html and point its API calls at the API/command server.

    The page may be served statically from a different port than the API
    server, so it needs to know where to send its fetch/EventSource calls.
    We inject that as `window.MAP_API_BASE`, based on the requester's
    hostname and the configured server_port.
    """
    index_file = FRONTEND_DIR / "index.html"
    content = index_file.read_text(encoding="utf-8")
    api_base = f"http://{hostname}:{server_port}"
    return content.replace("__MAP_API_BASE__", api_base).encode("utf-8")


def ensure_export_dir() -> bool:
    """Ensure EXPORT_DIR exists, even if deleted while process is running."""
    try:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print_red(f"[export] Failed to create export directory: {e}")
        return False

# Simple Server-Sent Events (SSE) support for notifying frontend of new GS pulls
SSE_CLIENTS = []
SSE_LOCK = threading.Lock()

# --- Continuous-capture control (proxied to MPS on the aircraft) -------------
# Serializes capture commands so two dashboards can't both pass the "is it
# idle?" check and issue two starts.
CAPTURE_CMD_LOCK = threading.Lock()
# (connect, read). Status is probed often, so it must fail fast.
CAPTURE_STATUS_TIMEOUT = (1.0, 1.5)
CAPTURE_CMD_TIMEOUT = (1.0, 5.0)
# MPS's own default (mini-plane-system/server/server.py start_cc).
DEFAULT_CAPTURE_INTERVAL = 2.0
# Bounds for an operator-supplied interval. Below ~0.5s the GoPro cannot keep up
# with the shutter; above 60s is almost certainly a typo rather than an intent.
MIN_CAPTURE_INTERVAL = 0.5
MAX_CAPTURE_INTERVAL = 60.0

# --- Log tailing -------------------------------------------------------------
LOG_READ_SEMAPHORE = threading.Semaphore(4)
# Bounds concurrent proxy reads toward the aircraft; that link also carries
# image uploads, so it must never see a pile of in-flight log requests.
LOG_PROXY_SEMAPHORE = threading.Semaphore(2)
LOG_PROXY_TIMEOUT = (2.0, 5.0)
# Circuit breaker: after this many consecutive failures, stop dialling the
# aircraft for a while and answer from the cached failure instead.
LOG_PROXY_FAIL_THRESHOLD = 3
LOG_PROXY_COOLDOWN_S = 30.0
_LOG_PROXY_STATE = {"fails": 0, "open_until": 0.0}
_LOG_PROXY_LOCK = threading.Lock()
LOG_MAX_BYTES_DEFAULT = 65536
LOG_MAX_BYTES_MIN = 4096
LOG_MAX_BYTES_MAX = 262144
LOG_GEN_HEAD_BYTES = 256


def _log_gen_token(path: Path, st) -> str:
    """Identity token for a log file's current incarnation.

    Device/inode plus a hash of the file's FIRST COMPLETE LINE. Two properties
    are required and both are easy to get wrong:

    - Stable across appends. Hashing a fixed-size head fails here: for a file
      shorter than the head size, appending changes the bytes read and so
      changes the token, making every poll look like a restart.
    - Different across incarnations. Inode alone fails here: MPS truncates its
      log in place via open(path, "w") at camera.py:53, leaving the inode
      unchanged. The first line carries a fresh timestamp after a restart, so
      hashing it catches exactly that case.

    Before the first newline exists the token is dev:ino only; the offset > size
    check still catches a reset in that window.
    """
    head = b""
    try:
        with open(path, "rb") as f:
            chunk = f.read(LOG_GEN_HEAD_BYTES)
        nl = chunk.find(b"\n")
        head = chunk[:nl] if nl != -1 else b""
    except Exception:
        pass
    seed = f"{getattr(st, 'st_dev', 0)}:{getattr(st, 'st_ino', 0)}:".encode()
    return hashlib.sha1(seed + head).hexdigest()[:16]


def read_log_chunk(path: Path, offset: int = 0, gen: str = "", max_bytes: int = LOG_MAX_BYTES_DEFAULT) -> dict:
    """Read new bytes from a log file, returning whole lines only.

    Byte-offset incremental tail. Returns a dict shaped for the /api/logs
    response. Never raises: failures come back as {"ok": False, "reason": ...}
    so the frontend has a single code path.
    """
    max_bytes = max(LOG_MAX_BYTES_MIN, min(int(max_bytes or LOG_MAX_BYTES_DEFAULT), LOG_MAX_BYTES_MAX))
    try:
        if path is None or not Path(path).is_file():
            return {"ok": False, "reason": "missing", "detail": str(path)}
        path = Path(path)
        st = path.stat()
    except PermissionError as e:
        return {"ok": False, "reason": "permission", "detail": str(e)}
    except Exception as e:
        return {"ok": False, "reason": "error", "detail": str(e)}

    size = st.st_size
    cur_gen = _log_gen_token(path, st)

    # The caller's offset is usable only if it came with a matching gen and
    # points inside the file. A gen mismatch means the file was restarted
    # (truncated in place, rotated, or replaced); an offset past EOF means the
    # same. Either way we resync to the tail rather than emit garbage.
    usable = (
        bool(gen)
        and gen == cur_gen
        and offset is not None
        and 0 <= offset <= size
    )

    dropped = 0
    if usable:
        resync = False
        start = offset
        if size - start > max_bytes:
            dropped = (size - start) - max_bytes
            start = size - max_bytes
    else:
        resync = True
        start = max(0, size - max_bytes)
        # Report the skipped span even on a first read (offset 0): opening a
        # months-old server.log should say so rather than silently show a tail.
        dropped = max(0, start - (offset or 0))

    try:
        with LOG_READ_SEMAPHORE:
            with open(path, "rb") as f:
                f.seek(start)
                raw = f.read(max_bytes)
    except PermissionError as e:
        return {"ok": False, "reason": "permission", "detail": str(e)}
    except Exception as e:
        return {"ok": False, "reason": "error", "detail": str(e)}

    new_offset = start + len(raw)

    # Any time we seeked rather than continued, `start` almost certainly landed
    # mid-line. Drop that leading fragment so the first rendered line is real.
    if start > 0 and (resync or dropped):
        nl = raw.find(b"\n")
        raw = raw[nl + 1:] if nl != -1 else b""

    # We're racing a live writer: a trailing partial line must be withheld and
    # re-read next poll, or lines get split across responses.
    if raw and not raw.endswith(b"\n"):
        nl = raw.rfind(b"\n")
        if nl == -1:
            new_offset -= len(raw)
            raw = b""
        else:
            new_offset -= (len(raw) - nl - 1)
            raw = raw[:nl + 1]

    text = raw.decode("utf-8", errors="replace")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    return {
        "ok": True,
        "gen": cur_gen,
        "offset": new_offset,
        "size": size,
        "reset": resync,
        "dropped_bytes": dropped,
        "lines": lines,
        "mtime": st.st_mtime,
    }

def notify_sse(event: str, data: dict):
    """Send an SSE event to all connected clients. Removes dead clients."""
    payload = {
        'event': event,
        'data': data,
    }
    msg = f"event: {event}\ndata: {_json.dumps(data)}\n\n".encode('utf-8')
    with SSE_LOCK:
        for client in list(SSE_CLIENTS):
            wfile = client.get('wfile')
            try:
                wfile.write(msg)
                wfile.flush()
            except Exception:
                try:
                    SSE_CLIENTS.remove(client)
                except Exception:
                    pass


class ResultStore:
    """Thread-safe store for detection results pushed by the cloud server."""

    def __init__(self):
        self._lock = threading.Lock()
        # Separate storage for cloud-pulled entries (most-recent wins)
        self._cloud: dict = {"tent": None, "mannequin": None}
        # Storage for best GD backups per label (highest-confidence wins)
        self._gd_best: dict = {"tent": None, "mannequin": None}

    def update(self, label: LabelType, assignment: dict, roi: ROI, classification: Classification, model_source: str = "", gemini_reason: str = "", meta_filename: str = None):
       """Update the store.

       Rules:
       - Cloud-pulled entries (`model_source` contains 'cloud') are stored as the canonical cloud best
         and always preferred when present.
       - GD backup entries (`model_source == 'gd_backup'`) are kept only if they have the highest
         confidence seen so far for that label (used when no cloud entry exists).
       """
       with self._lock:
            entry = (assignment, roi, classification, model_source, gemini_reason, meta_filename)
            # Normalize label to string
            lbl = None
            try:
                if label == LabelType.MANNEQUIN or str(label).lower().find('mannequin') >= 0:
                    lbl = 'mannequin'
                elif label == LabelType.TENT or str(label).lower().find('tent') >= 0:
                    lbl = 'tent'
            except Exception:
                lbl = None

            if lbl is None:
                print_yellow(f"[result_store] Received unknown label: {label}")
                return

            # Cloud entries win unconditionally (most-recent cloud_pull overrides anything)
            if model_source and 'cloud' in model_source:
                self._cloud[lbl] = entry
                try:
                    conf = float(classification.label[1]) if classification is not None else 0.0
                except Exception:
                    conf = 0.0
                print(f"[result_store] Updated cloud {lbl} (conf={conf:.3f}, model={model_source})")
                return

            # GD backup: keep only the highest-confidence GD backup for this label
            if model_source == 'gd_backup':
                try:
                    conf = float(classification.label[1]) if classification is not None else 0.0
                except Exception:
                    conf = 0.0
                prev = self._gd_best.get(lbl)
                prev_conf = -1.0
                if prev is not None and len(prev) >= 3 and prev[2] is not None:
                    try:
                        prev_conf = float(prev[2].label[1])
                    except Exception:
                        prev_conf = -1.0
                if prev is None or conf > prev_conf:
                    self._gd_best[lbl] = entry
                    print(f"[result_store] Updated GD backup {lbl} (conf={conf:.3f})")
                else:
                    print(f"[result_store] Kept existing GD backup {lbl} (conf={prev_conf:.3f}) over new {conf:.3f}")
                return

            # Fallback: treat other model sources as cloud entries
            self._cloud[lbl] = entry
            print(f"[result_store] Updated cloud-like {lbl} (model={model_source})")

    def get_mannequin(self):
        with self._lock:
            # Prefer cloud entry if present, else GD backup
            return self._cloud.get('mannequin') or self._gd_best.get('mannequin')

    def get_tent(self):
        with self._lock:
            return self._cloud.get('tent') or self._gd_best.get('tent')

    def clear(self):
        """Reset all in-memory best results (cloud-pulled and GD backup)."""
        with self._lock:
            self._cloud = {"tent": None, "mannequin": None}
            self._gd_best = {"tent": None, "mannequin": None}
        print_green("[result_store] Cleared all in-memory best results")

    def rebuild_from_disk(self, export_dir: Path):
        """Restore in-memory best-result state from previously exported meta files.

        Lets a restarted process recover 'current best' (used by both the
        dashboard and send_to_autopilot()) without waiting for a fresh
        detection. Replays every detection meta file, in original timestamp
        order, through update() so the same precedence rules apply as if the
        process had been running continuously.
        """
        entries = []
        for meta_path in export_dir.glob("meta_*.json"):
            if meta_path.name.startswith("meta_gs_"):
                continue  # raw GS pulls aren't detections
            try:
                with open(meta_path, "r") as f:
                    meta = _json.load(f)
                entries.append((meta.get("timestamp", 0), meta_path, meta))
            except Exception as e:
                print_yellow(f"[result_store] Skipping unreadable meta {meta_path.name}: {e}")

        entries.sort(key=lambda entry: entry[0])

        restored = 0
        for _, meta_path, meta in entries:
            try:
                bbox = meta.get("bbox") or []
                full_image_name = meta.get("full_image")
                if len(bbox) != 4 or not full_image_name:
                    continue
                full_path = export_dir / full_image_name
                if not full_path.exists():
                    continue

                x1, y1, x2, y2 = [int(v) for v in bbox]
                full_image = Image.open(full_path).convert("RGB")
                width, height = full_image.size
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))
                if x2 <= x1 or y2 <= y1:
                    continue

                roi = ROI(roi=full_image.crop((x1, y1, x2, y2)), top_left=(x1, y1), bottom_right=(x2, y2))
                label = _parse_label(meta.get("label"))
                classification = Classification(label=label, number_conf=float(meta.get("score", 0.0)))
                self.update(
                    label,
                    meta.get("assignment"),
                    roi,
                    classification,
                    meta.get("model_source", ""),
                    meta.get("gemini_reason"),
                    meta_path.name,
                )
                restored += 1
            except Exception as e:
                print_yellow(f"[result_store] Failed to restore {meta_path.name}: {e}")

        if restored:
            print_green(f"[result_store] Restored {restored} detection(s) from {export_dir}")


def _parse_label(raw_label) -> LabelType:
    if raw_label is None:
        return LabelType.UNKNOWN
    if isinstance(raw_label, str):
        return LabelType.__members__.get(raw_label.strip().upper(), LabelType.UNKNOWN)
    try:
        return LabelType(int(raw_label))
    except Exception:
        return LabelType.UNKNOWN


def _parse_result_payload(data: dict):
    """Parse a cloud-pushed result JSON payload into (label, assignment, ROI, Classification, model_source, gemini_reason).

    Flat payload shape (fields at top level):
    {
        "label": "tent" | "mannequin",
        "score": 0.95,
        "bbox": [x1, y1, x2, y2],
        "base64_image": "<base64>",
        "assignment": { ... } | null,
        "model_source": "grounding_dino",
        "gemini_reason": "..."
    }
    Returns (label, assignment, ROI, Classification, model_source, gemini_reason) or raises ValueError.
    """
    label = _parse_label(data.get("label"))
    score = float(data.get("score", 0.0))
    bbox = data.get("bbox") or []
    assignment = data.get("assignment")
    source_b64 = data.get("base64_image") or ""
    model_source = data.get("model_source") or ""
    gemini_reason = data.get("gemini_reason") or ""

    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values, got: {bbox}")

    payload = source_b64.split(",", 1)[1] if "," in source_b64 else source_b64
    if not payload:
        raise ValueError("Missing base64_image")

    full_image = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    width, height = full_image.size

    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate bbox after clipping: {[x1, y1, x2, y2]}")

    roi = ROI(roi=full_image.crop((x1, y1, x2, y2)), top_left=(x1, y1), bottom_right=(x2, y2))
    classification = Classification(label=label, number_conf=score)
    return label, assignment, roi, classification, model_source, gemini_reason


class MapCommandHandler(BaseHTTPRequestHandler):
    mapper = None
    result_store: ResultStore = None
    vision_client = None
    # e.g. "http://192.168.1.10:8000". None => capture controls disabled and no
    # socket is ever opened toward the aircraft.
    mps_base_url = None
    # id -> {"label": str, "kind": "file"|"proxy", "path": Path|None}
    # An allowlist. The client sends a key, never a path.
    log_sources: dict = {}

    def end_headers(self):
        # Allow browser frontends on different origins to call this API.
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        super().end_headers()

    def do_OPTIONS(self):
        # Handle CORS preflight requests.
        self.send_response(204)
        self.end_headers()
    
    def do_GET(self):
        """Handle mapping status, static files, and best-result API."""
        path = unquote(self.path.split('?',1)[0])
        try:
            # SSE stream endpoint for live updates
            if path == '/api/stream':
                # Send SSE headers and register client
                self.send_response(200)
                self.send_header('Content-type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'keep-alive')
                self.end_headers()
                with SSE_LOCK:
                    SSE_CLIENTS.append({'wfile': self.wfile})
                try:
                    # Keep connection alive; actual writes happen from notify_sse
                    while True:
                        time.sleep(60)
                except Exception:
                    with SSE_LOCK:
                        # remove if present
                        for c in list(SSE_CLIENTS):
                            if c.get('wfile') is self.wfile:
                                try: SSE_CLIENTS.remove(c)
                                except Exception: pass
                    return
            # Serve frontend index
            if path == '/' or path == '/index.html':
                index_file = FRONTEND_DIR / 'index.html'
                if index_file.exists():
                    hostname = (self.headers.get('Host') or '').split(':')[0] or '127.0.0.1'
                    content = render_frontend_index(hostname, self.server.server_address[1])
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(content)
                    return

            # Serve exported images and meta files
            if path.startswith('/export/'):
                rel = path[len('/export/'):]
                file_path = EXPORT_DIR / rel
                if file_path.exists() and file_path.is_file():
                    # Guess content type
                    if str(file_path).lower().endswith('.jpg') or str(file_path).lower().endswith('.jpeg'):
                        ctype = 'image/jpeg'
                    elif str(file_path).lower().endswith('.json'):
                        ctype = 'application/json'
                    else:
                        ctype = 'application/octet-stream'
                    self.send_response(200)
                    self.send_header('Content-type', ctype)
                    self.end_headers()
                    self.wfile.write(file_path.read_bytes())
                    return

            # API: latest best metadata for both labels
            if path == '/api/best':
                def load_meta_list(label_name: str, limit: int = 200):
                    metas = []
                    for mf in EXPORT_DIR.glob(f'meta_{label_name}_*.json'):
                        try:
                            with open(mf, 'r') as f:
                                m = _json.load(f)
                            # attach source meta filename for frontend identification
                            m['_meta_filename'] = mf.name
                            metas.append(m)
                        except Exception:
                            continue
                    metas.sort(key=lambda m: m.get('timestamp', 0), reverse=True)
                    return metas[:limit]

                # Build response with lists and indicate the current best meta filename per label
                mannequin_list = load_meta_list('mannequin')
                tent_list = load_meta_list('tent')
                # Also include raw GS pulls (meta_gs_*.json)
                gs_list = load_meta_list('gs')

                # Determine current best for mannequin: prefer result_store entry (meta filename stored),
                # otherwise fall back to latest gd_backup meta if available.
                current_best_mannequin = None
                try:
                    bm = self.result_store.get_mannequin() if self.result_store else None
                    if bm is not None and len(bm) >= 6 and bm[5]:
                        current_best_mannequin = bm[5]
                    else:
                        for m in mannequin_list:
                            if m.get('model_source') == 'gd_backup':
                                current_best_mannequin = m.get('_meta_filename')
                                break
                except Exception:
                    current_best_mannequin = None

                current_best_tent = None
                try:
                    bt = self.result_store.get_tent() if self.result_store else None
                    if bt is not None and len(bt) >= 6 and bt[5]:
                        current_best_tent = bt[5]
                    else:
                        for m in tent_list:
                            if m.get('model_source') == 'gd_backup':
                                current_best_tent = m.get('_meta_filename')
                                break
                except Exception:
                    current_best_tent = None

                # Also expose explicit "to_send" fields which indicate the meta
                # file that would be used when `VisionClient.send_result()` runs
                # (prefer cloud-pushed entry in result_store, else fall back to gd_backup).
                # Also include in-memory cloud results (if present) so the
                # frontend can show cloud-pulled bests even if no exported
                # meta file exists on disk.
                cloud_mannequin = None
                cloud_tent = None
                try:
                    bm = self.result_store.get_mannequin() if self.result_store else None
                    if bm is not None and len(bm) >= 3:
                        assign, roi, classification = bm[0], bm[1], bm[2]
                        bbox = list(roi.top_left) + list(roi.bottom_right) if roi is not None else []
                        score = float(classification.label[1]) if classification is not None else 0.0
                        cloud_mannequin = {
                            'assignment': assign,
                            'bbox': bbox,
                            'score': score,
                            'model_source': bm[3] if len(bm) > 3 else '',
                            'gemini_reason': bm[4] if len(bm) > 4 else '',
                            '_meta_filename': bm[5] if len(bm) > 5 else None,
                            'full_image': None,
                            'roi_image': None,
                        }
                        # If a meta filename was provided, try to read exported filenames
                        try:
                            mfname = cloud_mannequin.get('_meta_filename')
                            if mfname:
                                mfpath = EXPORT_DIR / mfname
                                if mfpath.exists():
                                    with open(mfpath, 'r') as _mf:
                                        try:
                                            _m = _json.load(_mf)
                                        except Exception:
                                            _m = {}
                                    # Determine expected filenames
                                    full_fname = _m.get('full_image')
                                    roi_fname = _m.get('roi_image')
                                    full_path = EXPORT_DIR / full_fname if full_fname else None
                                    roi_path = EXPORT_DIR / roi_fname if roi_fname else None

                                    # If full image file is missing, try to fetch from imaging GS
                                    if full_fname and (not full_path.exists()):
                                        try:
                                            assignment = cloud_mannequin.get('assignment') or {}
                                            img_endpoint = None
                                            if isinstance(assignment, dict):
                                                img_endpoint = (assignment.get('image') or {}).get('imageUrl') or (assignment.get('image') or {}).get('localImageUrl')
                                            if img_endpoint and hasattr(self, 'mapper') and getattr(self.mapper, 'work_client', None):
                                                try:
                                                    fetched = self.mapper.work_client.get_image(img_endpoint)
                                                    if fetched is not None:
                                                        try:
                                                            fetched.save(str(full_path), format='JPEG')
                                                        except Exception:
                                                            pass
                                                except Exception:
                                                    pass
                                        except Exception:
                                            pass

                                    # If roi image file is missing but full is present, create roi crop
                                    if roi_fname and (not roi_path.exists()) and full_fname and (EXPORT_DIR / full_fname).exists():
                                        try:
                                            from PIL import Image as _PILImage
                                            _full = _PILImage.open(str(EXPORT_DIR / full_fname)).convert('RGB')
                                            bbox = _m.get('bbox') or []
                                            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                                                x1, y1, x2, y2 = [int(v) for v in bbox]
                                                crop = _full.crop((x1, y1, x2, y2))
                                                try:
                                                    crop.save(str(roi_path), format='JPEG')
                                                except Exception:
                                                    pass
                                        except Exception:
                                            pass

                                    cloud_mannequin['full_image'] = full_fname if full_fname else None
                                    cloud_mannequin['roi_image'] = roi_fname if roi_fname else None
                                    cloud_mannequin['since_session_start_ms'] = _m.get('since_session_start_ms')
                                    # Attach base64 image data for frontend convenience
                                    try:
                                        if full_fname and (EXPORT_DIR / full_fname).exists():
                                            with open(EXPORT_DIR / full_fname, 'rb') as _fimg:
                                                import base64 as _b64
                                                cloud_mannequin['full_image_b64'] = 'data:image/jpeg;base64,' + _b64.b64encode(_fimg.read()).decode('utf-8')
                                        if roi_fname and (EXPORT_DIR / roi_fname).exists():
                                            with open(EXPORT_DIR / roi_fname, 'rb') as _fimg2:
                                                import base64 as _b642
                                                cloud_mannequin['roi_image_b64'] = 'data:image/jpeg;base64,' + _b642.b64encode(_fimg2.read()).decode('utf-8')
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                except Exception:
                    cloud_mannequin = None

                try:
                    bt = self.result_store.get_tent() if self.result_store else None
                    if bt is not None and len(bt) >= 3:
                        assign, roi, classification = bt[0], bt[1], bt[2]
                        bbox = list(roi.top_left) + list(roi.bottom_right) if roi is not None else []
                        score = float(classification.label[1]) if classification is not None else 0.0
                        cloud_tent = {
                            'assignment': assign,
                            'bbox': bbox,
                            'score': score,
                            'model_source': bt[3] if len(bt) > 3 else '',
                            'gemini_reason': bt[4] if len(bt) > 4 else '',
                            '_meta_filename': bt[5] if len(bt) > 5 else None,
                            'full_image': None,
                            'roi_image': None,
                        }
                        try:
                            mfname = cloud_tent.get('_meta_filename')
                            if mfname:
                                mfpath = EXPORT_DIR / mfname
                                if mfpath.exists():
                                    with open(mfpath, 'r') as _mf:
                                        try:
                                            _m = _json.load(_mf)
                                        except Exception:
                                            _m = {}
                                    full_fname = _m.get('full_image')
                                    roi_fname = _m.get('roi_image')
                                    full_path = EXPORT_DIR / full_fname if full_fname else None
                                    roi_path = EXPORT_DIR / roi_fname if roi_fname else None

                                    if full_fname and (not full_path.exists()):
                                        try:
                                            assignment = cloud_tent.get('assignment') or {}
                                            img_endpoint = None
                                            if isinstance(assignment, dict):
                                                img_endpoint = (assignment.get('image') or {}).get('imageUrl') or (assignment.get('image') or {}).get('localImageUrl')
                                            if img_endpoint and hasattr(self, 'mapper') and getattr(self.mapper, 'work_client', None):
                                                try:
                                                    fetched = self.mapper.work_client.get_image(img_endpoint)
                                                    if fetched is not None:
                                                        try:
                                                            fetched.save(str(full_path), format='JPEG')
                                                        except Exception:
                                                            pass
                                                except Exception:
                                                    pass
                                        except Exception:
                                            pass

                                    if roi_fname and (not roi_path.exists()) and full_fname and (EXPORT_DIR / full_fname).exists():
                                        try:
                                            from PIL import Image as _PILImage
                                            _full = _PILImage.open(str(EXPORT_DIR / full_fname)).convert('RGB')
                                            bbox = _m.get('bbox') or []
                                            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                                                x1, y1, x2, y2 = [int(v) for v in bbox]
                                                crop = _full.crop((x1, y1, x2, y2))
                                                try:
                                                    crop.save(str(roi_path), format='JPEG')
                                                except Exception:
                                                    pass
                                        except Exception:
                                            pass

                                    cloud_tent['full_image'] = full_fname if full_fname else None
                                    cloud_tent['roi_image'] = roi_fname if roi_fname else None
                                    cloud_tent['since_session_start_ms'] = _m.get('since_session_start_ms')
                                    # Attach base64 image data for frontend convenience
                                    try:
                                        if full_fname and (EXPORT_DIR / full_fname).exists():
                                            with open(EXPORT_DIR / full_fname, 'rb') as _fimg:
                                                import base64 as _b64
                                                cloud_tent['full_image_b64'] = 'data:image/jpeg;base64,' + _b64.b64encode(_fimg.read()).decode('utf-8')
                                        if roi_fname and (EXPORT_DIR / roi_fname).exists():
                                            with open(EXPORT_DIR / roi_fname, 'rb') as _fimg2:
                                                import base64 as _b642
                                                cloud_tent['roi_image_b64'] = 'data:image/jpeg;base64,' + _b642.b64encode(_fimg2.read()).decode('utf-8')
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                except Exception:
                    cloud_tent = None

                resp = {
                    'mapping_running': self.mapper.mapping_running,
                    'mapping_result': self.mapper.mapping_result,
                    'mannequin': mannequin_list,
                    'tent': tent_list,
                    'gs': gs_list,
                    'current_best_mannequin_meta': current_best_mannequin,
                    'current_best_tent_meta': current_best_tent,
                    'to_send_mannequin_meta': current_best_mannequin,
                    'to_send_tent_meta': current_best_tent,
                    'cloud_mannequin': cloud_mannequin,
                    'cloud_tent': cloud_tent,
                }
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(_json.dumps(resp).encode())
                return

            # Capture control status. Proxied to MPS, on its own endpoint so an
            # unreachable aircraft can never slow down /api/best.
            if path == '/api/capture/status':
                now_ms = int(time.time() * 1000)
                if self._mps_url('/pipeline/status') is None:
                    # Not configured: return without opening a socket.
                    self._json_response(200, {"configured": False, "checked_at": now_ms})
                    return
                ok, payload = self._mps_status()
                if ok:
                    self._json_response(200, {
                        "configured": True,
                        "reachable": True,
                        "mode": payload.get('mode'),
                        "pipeline_running": bool(payload.get('pipeline_running')),
                        "checked_at": now_ms,
                    })
                else:
                    self._json_response(200, {
                        "configured": True,
                        "reachable": False,
                        "error": str(payload),
                        "checked_at": now_ms,
                    })
                return

            # Available log sources. Drives the frontend tab strip, so adding a
            # source later needs no frontend change.
            if path == '/api/logs/sources':
                out = []
                for sid, entry in (self.log_sources or {}).items():
                    available, detail = True, None
                    if entry.get('kind') == 'file':
                        p = entry.get('path')
                        try:
                            available = p is not None and Path(p).is_file()
                        except Exception:
                            available = False
                        if not available:
                            detail = "log file not found (service may not have started yet)"
                    out.append({
                        "id": sid,
                        "label": entry.get('label', sid),
                        "kind": entry.get('kind', 'file'),
                        "available": available,
                        "detail": detail,
                    })
                self._json_response(200, {"sources": out})
                return

            # Incremental log tail. Query is parsed from the raw self.path:
            # line ~295 unquotes before splitting, which would misparse a %3F
            # inside a value, and we must not change that shared line.
            if path == '/api/logs':
                q = parse_qs(urlsplit(self.path).query)
                source_id = (q.get('source') or [''])[0]
                entry, err = self._resolve_log_source(source_id)
                if err is not None:
                    self._json_response(400, err)
                    return
                try:
                    offset = int((q.get('offset') or ['0'])[0])
                except ValueError:
                    offset = 0
                gen = (q.get('gen') or [''])[0]
                try:
                    max_bytes = int((q.get('max_bytes') or [str(LOG_MAX_BYTES_DEFAULT)])[0])
                except ValueError:
                    max_bytes = LOG_MAX_BYTES_DEFAULT

                if entry.get('kind') == 'file':
                    resolved = entry.get('path')
                    # Re-check containment at read time: the path was resolved
                    # once at startup, and a symlink could have been swapped in
                    # since to point somewhere else entirely.
                    try:
                        if resolved is None or Path(resolved).resolve() != Path(entry['path']).resolve():
                            result = {"ok": False, "reason": "error", "detail": "log path changed since startup"}
                        else:
                            result = read_log_chunk(Path(resolved), offset, gen, max_bytes)
                    except Exception as e:
                        result = {"ok": False, "reason": "error", "detail": str(e)}
                else:
                    result = self._proxy_log_chunk(offset, gen, max_bytes)
                result['source'] = source_id
                self._json_response(200, result)
                return

            # Default: mapping status (backwards compatibility)
            response = {
                "mapping_running": self.mapper.mapping_running,
                "mapping_result": self.mapper.mapping_result,
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(_json.dumps(response).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(_json.dumps({"status": "error", "message": str(e)}).encode())

    def _read_json_body(self):
        content_length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'
        return json.loads(raw)

    def _json_response(self, status: int, body: dict):
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(encoded)

    # --- MPS capture control helpers ---------------------------------------

    def _mps_url(self, path: str):
        """Absolute MPS URL for `path`, or None when capture control is off."""
        base = (self.mps_base_url or "").strip()
        if not base:
            return None
        if not base.startswith(("http://", "https://")):
            base = "http://" + base
        return base.rstrip('/') + path

    def _mps_status(self):
        """Read MPS pipeline status. Returns (ok, payload_or_error). Never raises.

        Deliberately does NOT use WorkClient._do_request_with_retries: that
        retries, and a retried start whose response was merely lost would issue
        a second start. Capture traffic is zero-retry throughout.
        """
        url = self._mps_url('/pipeline/status')
        if url is None:
            return False, "not configured"
        try:
            resp = requests.get(url, timeout=CAPTURE_STATUS_TIMEOUT)
        except Exception as e:
            return False, f"{type(e).__name__}"
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"
        try:
            return True, resp.json()
        except Exception:
            return False, "malformed status response"

    def _capture_command(self, action: str, interval=None):
        """Run capture_start / capture_stop against MPS. Returns a response dict."""
        if self._mps_url('/pipeline/status') is None:
            return {"status": "error", "message": "MPS address not configured (start core.py with --mps)"}

        if action == 'start':
            if interval is None or interval == '':
                interval = DEFAULT_CAPTURE_INTERVAL
            try:
                interval = float(interval)
            except (TypeError, ValueError):
                return {"status": "error", "message": f"Interval must be a number, got {interval!r}"}
            if not (MIN_CAPTURE_INTERVAL <= interval <= MAX_CAPTURE_INTERVAL):
                return {"status": "error",
                        "message": (f"Interval must be between {MIN_CAPTURE_INTERVAL} and "
                                    f"{MAX_CAPTURE_INTERVAL} seconds (got {interval})")}

        with CAPTURE_CMD_LOCK:
            if action == 'start':
                # Authoritative guard. /pipeline/start/cc stops and restarts the
                # pipeline, so starting while already running silently drops the
                # in-flight session. Verified here, not just in the UI, so a
                # stale tab or a raw curl cannot bypass it.
                ok, payload = self._mps_status()
                if not ok:
                    return {"status": "error",
                            "message": f"Cannot confirm MPS capture state ({payload}); not sending start."}
                mode = payload.get('mode')
                if payload.get('pipeline_running') or mode != 'idle':
                    # Anything not "idle" counts as busy: server.py's CamState
                    # comment says "distance_mode" but start_dm writes "dm", so
                    # matching exact strings is fragile.
                    return {"status": "error",
                            "message": f"MPS is busy (mode '{mode}'); stop capture before starting."}
                url = self._mps_url('/pipeline/start/cc')
                kwargs = {"params": {"interval": interval}}
            else:
                url = self._mps_url('/pipeline/stop')
                kwargs = {}

            try:
                resp = requests.post(url, timeout=CAPTURE_CMD_TIMEOUT, **kwargs)
            except Exception as e:
                return {"status": "error", "unknown_state": True,
                        "message": (f"MPS did not respond in time ({type(e).__name__}); "
                                    "capture state is UNKNOWN - verify on the MPS CLI before retrying.")}

            if 200 <= resp.status_code < 300:
                if action == 'start':
                    print_green(f"[capture] Started continuous capture (interval={interval}s)")
                    return {"status": "success", "message": f"Capture started ({interval}s interval)",
                            "interval": interval}
                print_green("[capture] Stopped continuous capture")
                return {"status": "success", "message": "Capture stopped"}

            body = (resp.text or "")[:300]
            print_red(f"[capture] MPS rejected {action} (status={resp.status_code})")
            return {"status": "error", "message": f"MPS returned {resp.status_code}: {body}"}

    # --- Log source helpers -------------------------------------------------

    def _resolve_log_source(self, source_id: str):
        """Look up an allowlisted log source. Returns (entry, error_dict)."""
        entry = (self.log_sources or {}).get(source_id)
        if entry is None:
            return None, {"ok": False, "reason": "unknown_source", "detail": source_id}
        return entry, None

    def _proxy_log_chunk(self, offset: int, gen: str, max_bytes: int) -> dict:
        """Fetch a log chunk from MPS, with a circuit breaker.

        The aircraft link carries image uploads, so a dead Pi must not be
        re-dialled on every poll for the rest of the flight.
        """
        url = self._mps_url('/logs')
        if url is None:
            return {"ok": False, "reason": "unreachable", "detail": "MPS address not configured"}

        now = time.time()
        with _LOG_PROXY_LOCK:
            if now < _LOG_PROXY_STATE["open_until"]:
                remaining = int(_LOG_PROXY_STATE["open_until"] - now)
                return {"ok": False, "reason": "unreachable",
                        "detail": f"aircraft not responding; retrying in {remaining}s"}

        try:
            with LOG_PROXY_SEMAPHORE:
                resp = requests.get(
                    url,
                    params={"offset": offset, "gen": gen, "max_bytes": max_bytes},
                    timeout=LOG_PROXY_TIMEOUT,
                )
        except Exception as e:
            with _LOG_PROXY_LOCK:
                _LOG_PROXY_STATE["fails"] += 1
                if _LOG_PROXY_STATE["fails"] >= LOG_PROXY_FAIL_THRESHOLD:
                    _LOG_PROXY_STATE["open_until"] = time.time() + LOG_PROXY_COOLDOWN_S
            return {"ok": False, "reason": "unreachable", "detail": type(e).__name__}

        if resp.status_code == 404:
            # Reached the aircraft, but it is running a build without /logs.
            with _LOG_PROXY_LOCK:
                _LOG_PROXY_STATE["fails"] = 0
            return {"ok": False, "reason": "unsupported",
                    "detail": "this MPS build has no /logs endpoint"}
        if resp.status_code != 200:
            return {"ok": False, "reason": "error", "detail": f"MPS returned {resp.status_code}"}

        try:
            payload = resp.json()
        except Exception:
            return {"ok": False, "reason": "error", "detail": "malformed response from MPS"}

        with _LOG_PROXY_LOCK:
            _LOG_PROXY_STATE["fails"] = 0
            _LOG_PROXY_STATE["open_until"] = 0.0
        return payload

    def _dm_presets(self) -> dict:
        """List distance-mode presets available on the aircraft."""
        url = self._mps_url('/pipeline/dm/presets')
        if url is None:
            return {"status": "error", "message": "MPS address not configured (start core.py with --mps)"}
        try:
            resp = requests.get(url, timeout=CAPTURE_STATUS_TIMEOUT)
        except Exception as e:
            return {"status": "error", "message": f"Could not reach MPS ({type(e).__name__})"}
        if resp.status_code == 404:
            return {"status": "error", "message": "This MPS build has no distance-mode presets"}
        if resp.status_code != 200:
            return {"status": "error", "message": f"MPS returned {resp.status_code}"}
        try:
            return {"status": "success", "presets": resp.json().get("presets", [])}
        except Exception:
            return {"status": "error", "message": "Malformed preset list from MPS"}

    def _dm_start(self, preset: str) -> dict:
        """Start distance mode from a named preset, with the same busy guard as CC."""
        if not preset:
            return {"status": "error", "message": "No distance-mode preset selected"}
        if self._mps_url('/pipeline/status') is None:
            return {"status": "error", "message": "MPS address not configured (start core.py with --mps)"}

        with CAPTURE_CMD_LOCK:
            ok, payload = self._mps_status()
            if not ok:
                return {"status": "error",
                        "message": f"Cannot confirm MPS state ({payload}); not starting distance mode."}
            mode = payload.get('mode')
            if payload.get('pipeline_running') or mode != 'idle':
                # start/dm cancels any running DM and stops the pipeline, so the
                # same anti-restart rule applies here as to continuous capture.
                return {"status": "error",
                        "message": f"MPS is busy (mode '{mode}'); stop it before starting distance mode."}

            url = self._mps_url('/pipeline/start/dm/preset/' + preset)
            try:
                resp = requests.post(url, timeout=CAPTURE_CMD_TIMEOUT)
            except Exception as e:
                return {"status": "error", "unknown_state": True,
                        "message": (f"MPS did not respond in time ({type(e).__name__}); "
                                    "state is UNKNOWN - verify on the MPS CLI.")}
            if resp.status_code == 404:
                return {"status": "error", "message": f"Unknown preset '{preset}' on the aircraft"}
            if not (200 <= resp.status_code < 300):
                return {"status": "error", "message": f"MPS returned {resp.status_code}: {(resp.text or '')[:300]}"}
            print_green(f"[capture] Started distance mode (preset={preset})")
            return {"status": "success", "message": f"Distance mode started ({preset})"}

    def _handle_result_push(self):
        """Handle POST /api/result — cloud server pushes a detection result here."""
        if self.result_store is None:
            self._json_response(503, {"status": "error", "message": "result_store not initialized"})
            return
        try:
            data = self._read_json_body()
            label, assignment, roi, classification, model_source, gemini_reason = _parse_result_payload(data)
            # Save the full image and ROI crop to export/ for inspection
            try:
                if not ensure_export_dir():
                    raise RuntimeError("export directory unavailable")
                ts = int(time.time() * 1000)
                label_name = str(label).lower()
                aid = assignment.get('id') if assignment else 'noid'
                full_fn = EXPORT_DIR / f"full_{label_name}_{aid}_{ts}.jpg"
                roi_fn = EXPORT_DIR / f"roi_{label_name}_{aid}_{ts}.jpg"
                # Save ROI crop
                try:
                    roi.image.save(str(roi_fn), format="JPEG")
                except Exception:
                    pass

                # Also attempt to save the original full image if provided in payload
                payload_b64 = data.get("base64_image") or ""
                if payload_b64:
                    import io as _io, base64 as _b64
                    payload = payload_b64.split(",", 1)[1] if "," in payload_b64 else payload_b64
                    try:
                        full_img = Image.open(_io.BytesIO(_b64.b64decode(payload))).convert("RGB")
                        full_img.save(str(full_fn), format="JPEG")
                    except Exception:
                        pass

                # Write metadata sidecar JSON for full and roi images
                try:
                        meta = {
                            "timestamp": ts,
                            "label": str(label).lower() if label is not None else None,
                            "assignment_id": assignment.get("id") if assignment else None,
                            "assignment": assignment,
                            "model_source": model_source,
                            "gemini_reason": gemini_reason,
                            "score": float(data.get("score", 0.0)),
                            "full_image": str(full_fn.name),
                            "roi_image": str(roi_fn.name),
                            "bbox": list(roi.top_left) + list(roi.bottom_right),
                            # Mark that this meta was created by a cloud push
                            "pushed": True,
                        }
                        meta_fn = EXPORT_DIR / f"meta_{label_name}_{aid}_{ts}.json"
                        with open(meta_fn, "w") as mf:
                            json.dump(meta, mf)
                        mf_name = meta_fn.name
                except Exception as e:
                    print_red(f"[result_push] Failed to write metadata sidecar: {e}")
            except Exception as e:
                print_red(f"[result_push] Failed to save exported images: {e}")

            # Update the in-memory result store and include the meta filename so the frontend
            # can identify which exported meta corresponds to the current best.
            try:
                self.result_store.update(label, assignment, roi, classification, model_source, gemini_reason, mf_name)
            except Exception:
                # Fallback to updating without meta filename
                try:
                    self.result_store.update(label, assignment, roi, classification, model_source, gemini_reason)
                except Exception:
                    pass
            self._json_response(200, {"status": "ok", "label": data.get("label", str(label))})
        except Exception as e:
            print_red(f"[result_push] Failed to parse result payload: {e}")
            self._json_response(400, {"status": "error", "message": str(e)})

    def do_POST(self):
        if self.path == "/api/result":
            self._handle_result_push()
            return
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'
            
            data = json.loads(post_data)
            command = data.get('command', '')
            
            if command == 'start':
                self.mapper.mapping = True
                response = {"status": "success", "message": "Mapping started"}
                print_green("Mapping started!")
            elif command == 'stop':
                self.mapper.mapping = False
                response = {"status": "success", "message": "Mapping stopped"}
                print_green("Mapping stopped!")
            elif command == 'generate':
                response = {"status": "error", "message": "Use 'trigger_mapping' command instead"}
            elif command == 'trigger_mapping':
                if self.mapper.mapping_running:
                    response = {"status": "error", "message": "Mapping pipeline already running"}
                else:
                    self.mapper.trigger_pipeline()
                    response = {"status": "success", "message": "Mapping triggered"}
            elif command == 'check_cloud':
                try:
                    work_client = getattr(self.mapper, 'work_client', None)
                    if work_client is None:
                        response = {"status": "error", "message": "work_client not available"}
                    else:
                        status = work_client.check_cloud_saved()

                        def _fmt(v):
                            if v is True:
                                return "saved"
                            if v is False:
                                return "empty"
                            return "unknown (request failed)"

                        message = f"Tent: {_fmt(status.get('tent'))} · Mannequin: {_fmt(status.get('mannequin'))}"
                        response = {
                            "status": "success",
                            "message": message,
                            "tent_saved": status.get('tent'),
                            "mannequin_saved": status.get('mannequin'),
                        }
                except Exception as e:
                    response = {"status": "error", "message": f"Failed to check cloud status: {e}"}
                    print_red(f"[api] check_cloud failed: {e}")
            elif command == 'clear_cloud':
                try:
                    work_client = getattr(self.mapper, 'work_client', None)
                    if work_client is None:
                        response = {"status": "error", "message": "work_client not available"}
                    else:
                        cloud_resp = work_client.clear_cloud()
                        if 200 <= cloud_resp.status_code < 300:
                            response = {"status": "success", "message": "Cloud server cleared"}
                        else:
                            response = {"status": "error", "message": f"Cloud server returned status {cloud_resp.status_code}"}
                except Exception as e:
                    response = {"status": "error", "message": f"Failed to clear cloud: {e}"}
                    print_red(f"[api] clear_cloud failed: {e}")
            elif command == 'send_to_autopilot':
                meta_filename = data.get('meta_filename')
                vision_client = getattr(self, 'vision_client', None)
                if vision_client is None:
                    response = {"status": "error", "message": "Vision client not ready yet"}
                elif not meta_filename:
                    response = {"status": "error", "message": "meta_filename is required"}
                else:
                    response = vision_client.send_meta_to_autopilot(meta_filename)
            elif command == 'clear_exports':
                try:
                    deleted = 0
                    if EXPORT_DIR.exists():
                        for f in EXPORT_DIR.iterdir():
                            if f.is_file():
                                try:
                                    f.unlink()
                                    deleted += 1
                                except Exception:
                                    pass
                    if self.result_store is not None:
                        self.result_store.clear()
                    print_green(f"[export] Cleared {deleted} local file(s) from {EXPORT_DIR}")
                    response = {"status": "success", "message": f"Deleted {deleted} local file(s)"}
                except Exception as e:
                    response = {"status": "error", "message": f"Failed to clear local exports: {e}"}
                    print_red(f"[api] clear_exports failed: {e}")
            elif command == 'capture_start':
                try:
                    response = self._capture_command('start', data.get('interval'))
                except Exception as e:
                    response = {"status": "error", "message": f"capture_start failed: {e}"}
                    print_red(f"[api] capture_start failed: {e}")
            elif command == 'capture_stop':
                try:
                    response = self._capture_command('stop')
                except Exception as e:
                    response = {"status": "error", "message": f"capture_stop failed: {e}"}
                    print_red(f"[api] capture_stop failed: {e}")
            elif command == 'dm_presets':
                try:
                    response = self._dm_presets()
                except Exception as e:
                    response = {"status": "error", "message": f"dm_presets failed: {e}"}
            elif command == 'dm_start':
                try:
                    response = self._dm_start(data.get('preset'))
                except Exception as e:
                    response = {"status": "error", "message": f"dm_start failed: {e}"}
                    print_red(f"[api] dm_start failed: {e}")
            else:
                response = {"status": "error", "message": f"Unknown command: {command}"}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            return response
        except Exception as e:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = {"status": "error", "message": str(e)}
            print_red(f"Error: {error_response}")
            self.wfile.write(json.dumps(error_response).encode())
            return error_response

    def do_DELETE(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'
            data = json.loads(post_data)
            
            image_id = data.get('image_id')
            if not image_id:
                raise ValueError("image_id is required for delete")
            
            image_path = os.path.join('images', f'{image_id}.jpg')
            if os.path.exists(image_path):
                os.remove(image_path)
                # Remove entry from CSV
                with open('images.csv', 'r') as f:
                    lines = f.readlines()
                with open('images.csv', 'w') as f:
                    for line in lines:
                        if not line.startswith(image_id + ','):
                            f.write(line)
                
                response = {"status": "success", "message": f"Image {image_id} deleted"}
                print_green(f"Image {image_id} deleted")

                self.send_response(204)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                return response 
            else:
                self.send_response(404)
                response = {"status": "error", "message": f"Image {image_id} not found"}
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                return response

        except Exception as e:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = {"status": "error", "message": str(e)}
            print_red(f"Error: {error_response}")
            self.wfile.write(json.dumps(error_response).encode())


class FrontendHandler(BaseHTTPRequestHandler):
    """Serves just the dashboard HTML, pointed at the API/command server.

    Lets the frontend run on its own port, separate from the API/command
    server (MapCommandHandler), which stays reachable at `server_port`.
    """
    server_port = 8080

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def do_GET(self):
        path = unquote(self.path.split('?', 1)[0])
        if path == '/' or path == '/index.html':
            index_file = FRONTEND_DIR / 'index.html'
            if index_file.exists():
                hostname = (self.headers.get('Host') or '').split(':')[0] or '127.0.0.1'
                content = render_frontend_index(hostname, self.server_port)
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(content)
                return
        self.send_response(404)
        self.end_headers()