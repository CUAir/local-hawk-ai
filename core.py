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
import os
import csv
import shutil
import threading
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer
from communication.intsys_gs_api import MapCommandHandler
import requests
import io
import base64
from utils.helper import print_green, print_red, print_yellow
import time
import threading
from constructs.detection import GDDetection
from constructs.image_types import Base64Image, LabelTypes


# Keep color only for section headers; all other logs are plain text.
header = print_green


GD_model = GDDetection()

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
    def __init__(self, work_client : WorkClient, mapper : Mapper, result_interval_seconds: float = 10.0):
        header("\n[vision] Initializing Work Client")
        self.work_client = work_client
        # print("Getting target attributes")
        # self.target_attr = self.work_client.get_target_attributes()
        header("[vision] Initializing Mapper")
        self.mapper = mapper
        self.result_interval_seconds = max(1.0, float(result_interval_seconds))
        self._send_lock = threading.Lock()
        self._result_scheduler_thread = threading.Thread(
            target=self._result_scheduler_loop,
            daemon=True,
        )
        self._result_scheduler_thread.start()

        # GD backup: best detected candidate per label from the most recent image
        self._gd_best_mannequin: tuple = None  # (assignment, ROI, Classification)
        self._gd_best_tent: tuple = None        # (assignment, ROI, Classification)

    def _result_scheduler_loop(self):
        while True:
            remaining = int(self.result_interval_seconds)
            while remaining > 0:
                # Log every 5 seconds, then every second in the last 5 seconds.
                if remaining <= 5 or remaining % 5 == 0:
                    print(f"[scheduler] Next GS send in {remaining}s")
                time.sleep(1)
                remaining -= 1

            if not self._send_lock.acquire(blocking=False):
                print_yellow("[scheduler] send_result() already running; skipping this tick")
                continue
            try:
                print_green("\n[scheduler] Sending current results to GS")
                self.send_result()
            except Exception as e:
                print_red(f"[scheduler] send_result() failed: {e}")
            finally:
                self._send_lock.release()

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
        self.mapper.mark_image_received()

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
            candidates = GD_model.detect_candidates(
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
        else:
            print_yellow("[gd_backup] No mannequin candidate found in current image")

        if best_tent is not None:
            roi, clf = _candidate_to_roi_classification(best_tent, LabelType.TENT)
            self._gd_best_tent = (self.assignment, roi, clf)
            print(f"[gd_backup] Cached tent candidate (score={best_tent.score:.3f})")
        else:
            print_yellow("[gd_backup] No tent candidate found in current image")


    # Send result of detection and classification to GS
    def send_result(self):
        def _send_with_log(target_name: str, assignment, roi, classification, source: str):
            try:
                response = self.work_client.send_adlc_output(assignment, roi, classification)
                if 200 <= response.status_code < 300:
                    print(
                        f"[send_result] Sent {target_name} using {source} image (assignment_id={assignment.get('id')}, status={response.status_code})"
                    )
                else:
                    print_red(
                        f"[send_result] Failed to send {target_name} using {source} image (assignment_id={assignment.get('id')}, status={response.status_code})"
                    )
            except Exception as e:
                print_red(f"[send_result] Exception while sending {target_name} using {source} image: {e}")

        print("[send_result] Getting MANNEQUIN image")
        m_assignment, m_roi, m_classification = self.work_client.get_mannequin_image()
        print("[send_result] Getting TENT image")
        t_assignment, t_roi, t_classification = self.work_client.get_tent_image()
        

        # Mannequin: prefer cloud server result; fall back to GD backup on 204
        if m_assignment is not None and m_roi is not None and m_classification is not None:
            print_green("[send_result] mannequin source = server")
            _send_with_log("mannequin", m_assignment, m_roi, m_classification, "server")
        elif self._gd_best_mannequin is not None:
            print_yellow("[send_result] mannequin source = cached (GD backup)")
            gd_assign, gd_roi, gd_clf = self._gd_best_mannequin
            _send_with_log("mannequin", gd_assign, gd_roi, gd_clf, "cached")
        else:
            print_red("[send_result] No mannequin candidate available to send")

        # Tent: prefer cloud server result; fall back to GD backup on 204
        if t_assignment is not None and t_roi is not None and t_classification is not None:
            print_green("[send_result] tent source = server")
            _send_with_log("tent", t_assignment, t_roi, t_classification, "server")
        elif self._gd_best_tent is not None:
            print_yellow("[send_result] tent source = cached (GD backup)")
            gd_assign, gd_roi, gd_clf = self._gd_best_tent
            _send_with_log("tent", gd_assign, gd_roi, gd_clf, "cached")
        else:
            print_red("[send_result] No tent candidate available to send")
        print("[send_result] DONE SENDING RESULTS ========= ")
def start_mapping_server(mapper: Mapper, port=8000):
    header(f"\n[mapping] Map HTTP server started on port {port}")

    # Set the mapper in the handler class
    MapCommandHandler.mapper = mapper

    # Create and start the HTTP server
    server = HTTPServer(('0.0.0.0', port), MapCommandHandler)

    try:
        server.serve_forever()
    except Exception as e:
        print_red(f"Error in map HTTP server: {e}")

def worker_loop(work_client: WorkClient, mapper: Mapper, result_interval_seconds: float = 10.0):
    header("\n[worker] Starting worker loop")
    worker = VisionClient(work_client, mapper, result_interval_seconds)
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

def main(gs_ip_address: str, cs_ip_address: str, map_server_port: int = 8000, result_interval_seconds: float = 10.0, map_idle_timeout: float = IDLE_MAPPING_TIMEOUT_SECONDS):
    header(
        f"\n[startup] GS={gs_ip_address}, CS={cs_ip_address}, map_port={map_server_port}, "
        f"send_interval={result_interval_seconds}s, map_idle_timeout={map_idle_timeout}s"
    )
    # Create worker(s) with detector and classifier
    work_client = WorkClient(gs_ip_address, cs_ip_address)
    mapper = Mapper(work_client)

    # Initialize mapping session directory (wipes stale data, creates clean dirs)
    _reset_session()

    # Start mapping HTTP server in background daemon thread
    threading.Thread(target=start_mapping_server, args=(mapper, map_server_port), daemon=True).start()
    threading.Thread(target=idle_mapping_monitor_loop, args=(mapper, map_idle_timeout), daemon=True).start()

    # TODO: implement MP, not doing it right now to see errors clearly
    worker_loop(work_client, mapper, result_interval_seconds)
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
    parser.add_argument('--csip', type=str, default="34.106.113.118:8000", help="Specify cloud server custom IP address")
    parser.add_argument('--map-port', type=int, default=8000, help="Port for the map command HTTP server")
    parser.add_argument('--interval-seconds', type=float, default=20.0, help="Run send_result() every F seconds")
    parser.add_argument('--map-idle-timeout', type=float, default=IDLE_MAPPING_TIMEOUT_SECONDS,
                        help="Seconds of ingest idle time before mapping auto-triggers (0 to disable)")

    args = parser.parse_args()

    if args.local:
        gs_ip_address = "127.0.0.1:9000"
        cs_ip_address = "127.0.0.1:8000"
    else:
        gs_ip_address = args.gsip
        cs_ip_address = args.csip

    main(gs_ip_address, cs_ip_address, args.map_port, args.interval_seconds, args.map_idle_timeout)
