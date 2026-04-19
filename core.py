from communication.work_client import WorkClient
from models.classifiers import ImageNet
from models.detectors import MaskRCNN
from vision.detectors.abstract_detector import AbstractDetector
from vision.classifiers.abstract_classifier import AbstractClassifier
from constructs.classification import Classification, LabelType
import argparse
import PIL.Image as Image
from multiprocessing import Process
import cv2
import os
from mapping.main import stitch_images, normalize_image, resize_image, rotate_image
from http.server import HTTPServer
from communication.intsys_gs_api import MapCommandHandler
import requests
import io
import base64
from utils.helper import print_green, print_red, print_yellow
import time
import threading

class Mapper:
    def __init__(self, work_client : WorkClient):
        self.work_client = work_client
        self.mapping = False
        if not os.path.exists('images.csv'):
            with open('images.csv', 'w') as f:
                f.write('Image,Latitude,Longitude,Altitude,Degrees_Clockwise_from_North\n')

        self.processed_images = []
    
    def add_image(self, image : Image.Image, metadata : dict):
        # Save image to images folder
        os.makedirs('images', exist_ok=True)
        if self.mapping == True:    
            image_path = os.path.join('images', f'{metadata["id"]}.jpg')
            image.save(image_path)
            print_green(f" Image {metadata['id']} saved to {image_path}")
            with open('images.csv', 'a') as f:
                longitude = metadata["telemetry"]["longitude"]
                latitude = metadata["telemetry"]["latitude"]
                altitude = metadata["telemetry"]["altitude"]
                yaw = metadata["telemetry"]["yaw"]
                f.write(f'{metadata["id"]},{latitude},{longitude},{altitude},{yaw}\n')

            # Send image to Ground Server server
            image_buffer = io.BytesIO()
            image.save(image_buffer, format='JPEG')
            image_data = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
            
            requests.post('http://localhost:8888/mapping/upload', json={
                'image_id': metadata["id"],
                'image_data': image_data,
                'timestamp': metadata['timestamp'],
                'lat': latitude,
                'lon': longitude, 
                'alt': altitude,
                'yaw': yaw
            })

            normalized_img = normalize_image(image)
            resized_img = resize_image(normalized_img, altitude)
            rotated_img = rotate_image(resized_img, float(yaw))
            
            self.processed_images.append({
                'image': rotated_img,
                'lat': latitude,
                'lon': longitude,
                'dimensions': rotated_img.shape[:2]  # (height, width)
            })
    
    def generate_map(self):
        print_green("Starting Mapping process...")
        result = stitch_images(self.processed_images)
        cv2.imwrite('map.jpg', result)
        print_green("Map saved to map.jpg")

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
        
        # Add image to mapping folder
        # self.mapper.add_image(image, metadata)

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

def main(gs_ip_address: str, cs_ip_address : str, map_server_port: int = 8000, result_interval_minutes: float = 5.0):
    # Create worker(s) with detector and classifier
    work_client = WorkClient(gs_ip_address, cs_ip_address)
    mapper = Mapper(work_client)

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

    args = parser.parse_args()

    if args.local:
        gs_ip_address = "127.0.0.1:9000"
        cs_ip_address = "127.0.0.1:8000"
    else:
        gs_ip_address = args.gsip
        cs_ip_address = args.csip

    main(gs_ip_address, cs_ip_address, args.map_port, args.interval_minutes)
    