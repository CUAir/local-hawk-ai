#!/bin/python

"""
This script will POST all images from DIR_NAME to the ground server running
at GS_URL.

With the -r/--random flag, it generates random images and telemetry data instead.
"""

from datetime import datetime
import os
import requests
import json
import argparse
import random
import time
import io
from PIL import Image
from PIL.ExifTags import TAGS
import piexif

parser = argparse.ArgumentParser()
parser.add_argument(
  "-d", "--directory",
  help="image directory", 
  default='images'
)
parser.add_argument(
  "-g", "--gs-host",
  help="ground server hostname and port", 
  default='127.0.0.1:9000'
)
parser.add_argument(
    "-m", "--mode",
    help="telemetry mode, can be 'dummy' or 'json'",
    default='json'
)
parser.add_argument(
    "-r", "--random",
    help="generate N random images with random telemetry",
    type=int,
    default=0
)
parser.add_argument(
    "-s", "--save-locally",
    help="save random images locally before uploading",
    action="store_true"
)
args = parser.parse_args()

DIR_NAME = args.directory
GS_HOST = args.gs_host
TELEM_MODE = args.mode
RANDOM_COUNT = args.random
SAVE_LOCALLY = args.save_locally

# Constants for default
IMAGE_LAT = 38.315946
IMAGE_LONG = -76.558576
IMAGE_ALT = 100

# Reasonable bounds for random telemetry generation
BOUNDS = {
    'latitude': (42.0, 43.0),      # Roughly in the region of the example data
    'longitude': (-77.0, -76.0),
    'altitude_msl': (250.0, 400.0),
    'altitude_rel': (0.0, 150.0),
    'roll': (-45.0, 45.0),          # degrees
    'pitch': (-45.0, 45.0),
    'yaw': (-180.0, 180.0),
    'gimbal_roll': (-10.0, 10.0),
    'gimbal_pitch': (-90.0, 0.0),   # typically pointing downward
    'gimbal_yaw': (-180.0, 180.0)
}

_last_generated_timestamp = 0

def generate_timestamp():
    """Generate a strictly increasing timestamp based on current wall-clock time.

    Uses milliseconds since Unix epoch to provide much finer granularity than
    seconds while remaining stateless across separate script runs.
    """
    global _last_generated_timestamp

    current_ms = time.time_ns() // 1_000_000
    if current_ms <= _last_generated_timestamp:
        current_ms = _last_generated_timestamp + 1

    _last_generated_timestamp = current_ms
    return current_ms

def generate_random_telemetry():
    """Generate random telemetry data within reasonable bounds"""
    return {
        "timestamp": generate_timestamp(),
        "imgMode": "fixed",
        "telemetry": {
            "planeYaw": random.uniform(*BOUNDS['yaw']),
            "altitude": random.uniform(*BOUNDS['altitude_msl']),
            "gps": {
                "latitude": random.uniform(*BOUNDS['latitude']),
                "longitude": random.uniform(*BOUNDS['longitude']),
            },
            "gimOrt": {
                "pitch": random.uniform(*BOUNDS['gimbal_pitch']),
                "roll": random.uniform(*BOUNDS['gimbal_roll'])
            },
        }
    }

def fetch_random_image():
    """Fetch a random image from picsum.photos"""
    response = requests.get("https://picsum.photos/5000/3000")
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to fetch random image: {response.status_code}")

def add_exif_data(image_data):
    """Add random EXIF data to an image"""
    # Open image with PIL
    image = Image.open(io.BytesIO(image_data))
    
    # Create EXIF data
    exif_dict = {
        "0th": {},
        "Exif": {},
        "GPS": {},
        "1st": {},
        "thumbnail": None
    }
    
    # Add basic camera info
    exif_dict["0th"][piexif.ImageIFD.Make] = "Canon"
    exif_dict["0th"][piexif.ImageIFD.Model] = "Canon EOS 5D Mark IV"
    exif_dict["0th"][piexif.ImageIFD.Software] = "Adobe Photoshop Lightroom"
    exif_dict["0th"][piexif.ImageIFD.DateTime] = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    
    # Add camera settings
    # Focal length in mm (e.g., 50mm lens) - stored as rational (numerator, denominator)
    focal_length_mm = random.randint(24, 200)
    exif_dict["Exif"][piexif.ExifIFD.FocalLength] = (focal_length_mm, 1)
    
    # F-number (aperture) - e.g., f/2.8
    f_numbers = [14, 18, 20, 22, 28, 35, 40, 45, 50, 56, 63, 71, 80]  # f/1.4, f/1.8, f/2.0, etc.
    f_number = random.choice(f_numbers)
    exif_dict["Exif"][piexif.ExifIFD.FNumber] = (f_number, 10)
    
    # ISO speed
    iso_values = [100, 200, 400, 800, 1600, 3200]
    exif_dict["Exif"][piexif.ExifIFD.ISOSpeedRatings] = random.choice(iso_values)
    
    # Shutter speed (exposure time) - e.g., 1/1000 second
    shutter_speeds = [(1, 1000), (1, 500), (1, 250), (1, 125), (1, 60), (1, 30)]
    exif_dict["Exif"][piexif.ExifIFD.ExposureTime] = random.choice(shutter_speeds)
    
    # Image dimensions
    exif_dict["Exif"][piexif.ExifIFD.PixelXDimension] = image.width
    exif_dict["Exif"][piexif.ExifIFD.PixelYDimension] = image.height
    
    # Encode EXIF data
    exif_bytes = piexif.dump(exif_dict)
    
    # Save image with EXIF data to BytesIO
    output = io.BytesIO()
    image.save(output, format='JPEG', exif=exif_bytes, quality=95)
    output.seek(0)
    
    return output.getvalue()

# Determine what to upload
if RANDOM_COUNT > 0:
    print(f'Generating {RANDOM_COUNT} random images with random telemetry...')
    upload_items = [(f"random_image_{i}", None) for i in range(RANDOM_COUNT)]
else:
    # Gather set of each image identifier stored on SD card
    ids = set()
    for filename in os.scandir(DIR_NAME):
        if filename.is_file():
            ids.add(filename.path.split(".")[0])
    
    print(f'found files: {", ".join(ids)}')
    upload_items = [(img_name, img_name) for img_name in ids]

# Post request for each image to gs
upload_log_file = open("save-data-upload.log", "w")

for display_name, img_name in upload_items:
    if RANDOM_COUNT > 0:
        # Generate random image and telemetry
        request_data = generate_random_telemetry()
        
        print(f'Fetching random image {display_name}... ', end='', flush=True)
        try:
            image_data = fetch_random_image()
            print('done')
            
            print(f'  Adding EXIF data... ', end='', flush=True)
            image_data = add_exif_data(image_data)
            print('done')
        except Exception as e:
            print(f'error: {e}')
            continue
        
        jpg_filename = f"{display_name}.JPG"
        
        # Save locally if requested
        if SAVE_LOCALLY:
            os.makedirs(DIR_NAME, exist_ok=True)
            local_path = os.path.join(DIR_NAME, jpg_filename)
            with open(local_path, 'wb') as f:
                f.write(image_data)
            print(f'  Saved to {local_path}')
            
            # Also save JSON
            json_path = os.path.join(DIR_NAME, f"{display_name}.json")
            with open(json_path, 'w') as f:
                json.dump(request_data, f, indent=2)
        
        image_file = io.BytesIO(image_data)
    else:
        # Use existing images and telemetry
        json_filename = img_name + ".json"
        jpg_filename = img_name + ".JPG"

        if TELEM_MODE == 'dummy':
            request_data = {
                "timestamp": generate_timestamp(),
                "imgMode": "fixed",
                "telemetry": {
                    "altitude": IMAGE_ALT,
                    "planeYaw": 0,
                    "gps": {"latitude": IMAGE_LAT, "longitude": IMAGE_LONG},
                    "gimOrt": {"pitch": 0, "roll": 0},
                },
            }
        else:
            with open(json_filename, "rb") as json_file:
                request_data = json.load(json_file)

                request_data = {
                                                                        "timestamp": generate_timestamp(),
                  "imgMode": "fixed",
                  "telemetry": {
                    "planeYaw": request_data['plane_attitude']['yaw'],
                    "altitude": request_data['position']['altitude_msl'],
                    "gps": {
                      "latitude": request_data['position']['latitude'],
                      "longitude": request_data['position']['longitude'],
                    },
                    "gimOrt": { 
                      "pitch": request_data['gimbal_attitude']['pitch'], 
                      "roll": request_data['gimbal_attitude']['roll']
                    },
                  }
                }

                print(request_data)

        image_file = open(jpg_filename, "rb")

    files = {"json": json.dumps(request_data), "files": image_file}

    print(f'uploading {jpg_filename} ... ', end = '', flush = True)

    try:
        response = requests.post(url=f"http://{GS_HOST}/api/v1/image", files=files)

        if response.status_code == 200:
            print('success')
            upload_log_file.write(
                f"file {jpg_filename} status {response.status_code} response '{response.content}'"
            )
        else:
            print(f'error ({response.status_code})')
            try:
                response_json = response.json()
                print(f"\tresponse: {response_json}")
            except:
                print(f"\tresponse: {response.text}")
            upload_log_file.write(
                f"file {jpg_filename} status {response.status_code} response '{response.content}'"
            )
    except requests.exceptions.ConnectionError as e:
        print(f'connection error - is ground server running at {GS_HOST}?')
        upload_log_file.write(
            f"file {jpg_filename} CONNECTION ERROR: {str(e)}"
        )
    except Exception as e:
        print(f'error: {str(e)}')
        upload_log_file.write(
            f"file {jpg_filename} ERROR: {str(e)}"
        )
    
    upload_log_file.write("\n")
    
    # Close the file handle
    if hasattr(image_file, 'close'):
        image_file.close()

print("full log written to 'save-data-upload.log'")
