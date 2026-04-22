from communication.work_client import WorkClient
from models.classifiers import ImageNet
from models.detectors import MaskRCNN
from vision.detectors.abstract_detector import AbstractDetector
from vision.classifiers.abstract_classifier import AbstractClassifier
from constructs.classification import Classification, LabelType
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


GD_model = GDDetection()


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
        print_green(f"[mapping] Saved {img_filename} to session")
        return True
    except Exception as e:
        print_yellow(f"[mapping] Failed to save image {metadata.get('id')}: {e}")
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
        print_green(f"[mapping] Done → {final_out}")
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
        print_green("[mapping] Pipeline triggered in background thread")

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
        print_green(
            f"[mapping] Idle for {timeout_seconds}s with {n_images} images; auto-triggering pipeline"
        )
        self.last_auto_trigger_ts = time.time()
        self.trigger_pipeline()

class VisionClient:
    def __init__(self, work_client : WorkClient, mapper : Mapper, result_interval_minutes: float = 5.0):
        print("Initializing Work Client")
        self.work_client = work_client
        # print("Getting target attributes")
        # self.target_attr = self.work_client.get_target_attributes()
        print("Initializing Mapper")
        self.mapper = mapper
        self.result_interval_seconds = max(1.0, float(result_interval_minutes) * 60.0)
        self._send_lock = threading.Lock()
        self._result_scheduler_thread = threading.Thread(
            target=self._result_scheduler_loop,
            daemon=True,
        )
        self._result_scheduler_thread.start()

    def _result_scheduler_loop(self):
        while True:
            time.sleep(self.result_interval_seconds)
            if not self._send_lock.acquire(blocking=False):
                print("> send_result() already running, skipping this interval")
                continue
            try:
                print("> [scheduler] Sending result to GS")
                self.send_result()
            except Exception as e:
                print_red(f"Error in send_result scheduler: {e}")
            finally:
                self._send_lock.release()

    def run_task(self):
        print(">> Running task...")
        
        print("> Requesting image")
        self.request_image()
        time.sleep(1)
        print("> Running model on image")
        self.run_model()

        print("> Task finished")
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
        response = self.work_client.send_image(self.image, self.assignment)
        print(f"> GET Response: {response.status_code}")

    # Send result of detection and classification to GS
    def send_result(self):
        m_assignment, m_roi, m_classification = self.work_client.get_mannequin_image()
        t_assignment, t_roi, t_classification = self.work_client.get_tent_image()
        

        if m_assignment is not None and m_roi is not None and m_classification is not None:
            print("> Sending mannequin image")
            self.work_client.send_adlc_output(
                m_assignment, m_roi, m_classification
            )
        else:
            print("> No valid mannequin image to send")

        if t_assignment is not None and t_roi is not None and t_classification is not None:
            print("> Sending tent image")
            self.work_client.send_adlc_output(
                t_assignment, t_roi, t_classification
            )
        else:
            print("> No valid tent image to send")

def start_mapping_server(mapper: Mapper, port=8000):
    print_green(f"Map HTTP server started on port {port}!")

    # Set the mapper in the handler class
    MapCommandHandler.mapper = mapper

    # Create and start the HTTP server
    server = HTTPServer(('0.0.0.0', port), MapCommandHandler)

    try:
        server.serve_forever()
    except Exception as e:
        print_red(f"Error in map HTTP server: {e}")

def worker_loop(work_client: WorkClient, mapper: Mapper, result_interval_minutes: float = 5.0):
    print("Starting Worker process...")
    worker = VisionClient(work_client, mapper, result_interval_minutes)
    while True:
        try:
            worker.run_task()
        except Exception as e:
            print_red(f"Error in Worker process: {e}")


def idle_mapping_monitor_loop(mapper: Mapper, timeout_seconds: float):
    """Background loop for fallback auto-trigger when no images arrive."""
    if timeout_seconds <= 0:
        print_yellow("[mapping] Idle monitor disabled (--map-idle-timeout=0)")
        return
    print_green(
        f"[mapping] Idle monitor started (timeout={timeout_seconds}s)"
    )
    while True:
        try:
            mapper.maybe_trigger_pipeline_on_idle(timeout_seconds)
        except Exception as e:
            print_yellow(f"[mapping] Idle monitor error: {e}")
        time.sleep(IDLE_MAPPING_POLL_SECONDS)

def main(gs_ip_address: str, cs_ip_address: str, map_server_port: int = 8000, result_interval_minutes: float = 5.0, map_idle_timeout: float = IDLE_MAPPING_TIMEOUT_SECONDS):
    # Create worker(s) with detector and classifier
    work_client = WorkClient(gs_ip_address, cs_ip_address)
    mapper = Mapper(work_client)

    # Initialize mapping session directory (wipes stale data, creates clean dirs)
    _reset_session()

    # Start mapping HTTP server in background daemon thread
    threading.Thread(target=start_mapping_server, args=(mapper, map_server_port), daemon=True).start()
    threading.Thread(target=idle_mapping_monitor_loop, args=(mapper, map_idle_timeout), daemon=True).start()

    # TODO: implement MP, not doing it right now to see errors clearly
    worker_loop(work_client, mapper, result_interval_minutes)
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
    parser.add_argument('--csip', type=str, default="34.73.222.251:8000", help="Specify cloud server custom IP address")
    parser.add_argument('--map-port', type=int, default=8000, help="Port for the map command HTTP server")
    parser.add_argument('--interval-minutes', type=float, default=1.0, help="Run send_result() every F minutes")
    parser.add_argument('--map-idle-timeout', type=float, default=IDLE_MAPPING_TIMEOUT_SECONDS,
                        help="Seconds of ingest idle time before mapping auto-triggers (0 to disable)")

    args = parser.parse_args()

    if args.local:
        gs_ip_address = "127.0.0.1:9000"
        cs_ip_address = "127.0.0.1:8000"
    else:
        gs_ip_address = args.gsip
        cs_ip_address = args.csip

    main(gs_ip_address, cs_ip_address, args.map_port, args.interval_minutes, args.map_idle_timeout)
