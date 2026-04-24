from http.server import BaseHTTPRequestHandler
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


class ResultStore:
    """Thread-safe store for detection results pushed by the cloud server."""

    def __init__(self):
        self._lock = threading.Lock()
        self.best_mannequin: tuple = None  # (assignment, ROI, Classification)
        self.best_tent: tuple = None       # (assignment, ROI, Classification)

    def update(self, label: LabelType, assignment: dict, roi: ROI, classification: Classification, model_source: str = "", gemini_reason: str = ""):
       print("POLLED!!!!")
       with self._lock:
            entry = (assignment, roi, classification, model_source, gemini_reason)
            if label == LabelType.MANNEQUIN:
                self.best_mannequin = entry
                print(f"[result_store] Updated mannequin (conf={classification.label[1]:.3f}, model={model_source})")
            elif label == LabelType.TENT:
                self.best_tent = entry
                print(f"[result_store] Updated tent (conf={classification.label[1]:.3f}, model={model_source})")
            else:
                print_yellow(f"[result_store] Received unknown label: {label}")

    def get_mannequin(self):
        with self._lock:
            return self.best_mannequin

    def get_tent(self):
        with self._lock:
            return self.best_tent


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
    
    def do_GET(self):
        """Return current mapping pipeline status (mirrors hawk-ai GET /api/mapping/status)."""
        try:
            response = {
                "mapping_running": self.mapper.mapping_running,
                "mapping_result": self.mapper.mapping_result,
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode())

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
                        "model_source": model_source,
                        "gemini_reason": gemini_reason,
                        "score": float(data.get("score", 0.0)),
                        "full_image": str(full_fn.name),
                        "roi_image": str(roi_fn.name),
                    }
                    meta_fn = EXPORT_DIR / f"meta_{label_name}_{aid}_{ts}.json"
                    with open(meta_fn, "w") as mf:
                        json.dump(meta, mf)
                except Exception as e:
                    print_red(f"[result_push] Failed to write metadata sidecar: {e}")
            except Exception as e:
                print_red(f"[result_push] Failed to save exported images: {e}")

            self.result_store.update(label, assignment, roi, classification, model_source, gemini_reason)
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