from http.server import BaseHTTPRequestHandler
from urllib.parse import unquote
import json as _json
import json
import os
import io
import base64
import threading
from utils.helper import print_green, print_red, print_yellow
from constructs.roi import ROI
from constructs.classification import Classification, LabelType
from PIL import Image
import time
import json
from pathlib import Path

# Directory to save exported images pushed by cloud
EXPORT_DIR = Path(__file__).parent.parent / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Simple Server-Sent Events (SSE) support for notifying frontend of new GS pulls
SSE_CLIENTS = []
SSE_LOCK = threading.Lock()

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
                frontend_dir = Path(__file__).parent.parent / 'frontend'
                index_file = frontend_dir / 'index.html'
                if index_file.exists():
                    content = index_file.read_bytes()
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
                ts = int(time.time() * 1000)
                label_name = str(label).lower()
                aid = assignment.get('id') if assignment else 'noid'
                full_fn = EXPORT_DIR / f"full_{label_name}_{aid}_{ts}.jpg"
                roi_fn = EXPORT_DIR / f"roi_{label_name}_{aid}_{ts}.jpg"
                # The _parse_result_payload already created roi.roi (PIL Image) and we still have full image via roi
                # But we don't have direct full_image variable here; reconstruct from roi by pasting onto new image of roi size is not needed.
                # Save ROI crop
                try:
                    roi.roi.save(str(roi_fn), format="JPEG")
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
            return error_response