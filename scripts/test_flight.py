#!/bin/python

"""
This script will POST N random images from a test-data flight folder to the
ground server running at GS_URL, using each image's paired telemetry
(<name>.JPG + <name>_gs.json).
"""

import os
import sys

# scripts/random.py shadows the stdlib `random` module for any script run
# from this directory (Python puts the script's own dir first on sys.path),
# which also breaks third-party libraries (e.g. requests/urllib3) that import
# the stdlib `random` internally. Drop it from sys.path before importing
# anything else so all imports below resolve the real stdlib module.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)

import json
import time
import random
import argparse
import requests

parser = argparse.ArgumentParser()
parser.add_argument(
    "-n", "--num",
    help="number of images to upload",
    type=int,
    default=2
)
parser.add_argument(
    "-r", "--random",
    help="select images randomly instead of the first N by timestamp order",
    action="store_true"
)
parser.add_argument(
    "-p", "--path",
    help="path to the test data folder (each <name>.JPG must have a paired <name>_gs.json)",
    type=str,
    required=False,
    default="./test_data/flight1/"
)
parser.add_argument(
    "-g", "--gs-host",
    help="ground server hostname and port",
    default='127.0.0.1:9000'
)
args = parser.parse_args()

NUM_IMAGES = args.num
USE_RANDOM = args.random
DIR_PATH = args.path
GS_HOST = args.gs_host


def find_pairs(dir_path):
    """Find <name>.JPG/.jpg files in dir_path that have a matching <name>_gs.json sibling.

    Returns a list of (img_path, json_path, timestamp) tuples, where timestamp
    is read from the paired _gs.json so callers can sort chronologically.
    """
    pairs = []
    for filename in os.scandir(dir_path):
        if not filename.is_file():
            continue
        if not filename.name.lower().endswith(('.jpg', '.jpeg')):
            continue
        stem = filename.name.rsplit('.', 1)[0]
        json_path = os.path.join(dir_path, f"{stem}_gs.json")
        if os.path.isfile(json_path):
            with open(json_path, "r") as f:
                timestamp = json.load(f)["timestamp"]
            pairs.append((filename.path, json_path, timestamp))
    return pairs


_last_generated_timestamp = 0


def generate_timestamp():
    """Generate a strictly increasing timestamp (ms since epoch), mirroring
    random.py's generate_timestamp(). The GS keys uploaded files by this
    timestamp, so re-uploading a file with its original recorded timestamp
    (e.g. on a repeat run) is rejected as a duplicate — always mint a fresh
    one instead."""
    global _last_generated_timestamp
    current_ms = time.time_ns() // 1_000_000
    if current_ms <= _last_generated_timestamp:
        current_ms = _last_generated_timestamp + 1
    _last_generated_timestamp = current_ms
    return current_ms


def build_telemetry_from_file(data: dict) -> dict:
    """Build a request payload with the same shape as random.py's generate_random_telemetry(),
    filled in with real recorded values instead of random ones (timestamp excepted --
    see generate_timestamp())."""
    t = data["telemetry"]
    return {
        "timestamp": generate_timestamp(),
        "imgMode": data.get("imgMode", "fixed"),
        "telemetry": {
            "planeYaw": t["planeYaw"],
            "altitude": t["altitude"],
            "gps": {
                "latitude": t["gps"]["latitude"],
                "longitude": t["gps"]["longitude"],
            },
            "gimOrt": {
                "pitch": t["gimOrt"]["pitch"],
                "roll": t["gimOrt"]["roll"],
            },
        },
    }


pairs = find_pairs(DIR_PATH)
if len(pairs) < NUM_IMAGES:
    raise SystemExit(
        f"Only found {len(pairs)} image/telemetry pair(s) in {DIR_PATH}, "
        f"but {NUM_IMAGES} were requested"
    )

if USE_RANDOM:
    selected = random.sample(pairs, k=NUM_IMAGES)
else:
    selected = sorted(pairs, key=lambda pair: pair[2])[:NUM_IMAGES]

upload_log_file = open("test-flight-upload.log", "w")

for img_path, json_path, _timestamp in selected:
    with open(json_path, "r") as f:
        raw_data = json.load(f)
    request_data = build_telemetry_from_file(raw_data)

    jpg_filename = os.path.basename(img_path)
    print(f'uploading {jpg_filename} ... ', end='', flush=True)

    image_file = open(img_path, "rb")
    files = {"json": json.dumps(request_data), "files": image_file}

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
            except Exception:
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
    image_file.close()

print("full log written to 'test-flight-upload.log'")
