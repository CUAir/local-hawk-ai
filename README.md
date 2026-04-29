# HawkAI
The complete intelligent systems system for Computer Vision tasks like object detection and mapping. 
It integrates with the Intelligent Systems Ground System (GS) and Imaging System's Ground Server for real time communication.
In addition to its light-weight process for mission-critical ML tasks, it also hosts some other important
functionality, including a script to transform annotated data 
into an easily-processable CSV for ML training.


## Running the local worker directly (`core.py`)

You can also run the local worker without Docker:

```bash
python core.py [options]
```

### Command-line arguments

The local worker (`core.py`) accepts the following command-line arguments:

- `--local` (flag)
  - Use localhost defaults for GS/CS addresses. When set, the worker will use:
    - `gs = 127.0.0.1:9000`
    - `cs = 127.0.0.1:8000`

- `--gsip <host:port>`
  - Ground Station API address (used when `--local` is not set).
  - Default: `127.0.0.1:9000`

- `--csip <host:port>`
  - Cloud Server API address (used when `--local` is not set).
  - Default: `34.106.160.143:8000`

- `--map-port <int>`
  - Port for the map-command HTTP server (serves the frontend UI).
  - Default: `8080`

- `--interval-seconds <float>`
  - Scheduler interval (seconds) between cloud polls and autopilot sends.
  - Default: `20.0`

- `--autopilot-ip <host>`
  - Autopilot host/IP to which target payloads are POSTed. If omitted, autopilot posting is disabled.
  - Default: `None`

- `--autopilot-port <int>`
  - Autopilot port to POST target payloads to.
  - Default: `8001`

- `--map-idle-timeout <float>`
  - Seconds of ingest idle time before the mapping pipeline auto-triggers. Set to `0` to disable auto-trigger.
  - Default: `20`

Example usage:

```bash
python core.py --gsip 192.168.1.2:9000 --csip 10.0.0.2:8000 --map-port 8080 --interval-seconds 20 --autopilot-ip 127.0.0.1 --autopilot-port 8001 --map-idle-timeout 20
```

### To Generate Map, use the Intsys GS

### Download model weights

Download the model weights from the Box folder and place them in the `model_weights` directory. The required weight files are:

# local-hawk-ai

Current local runtime for GS ingestion, cloud upload/pull, GroundingDINO fallback detection, mapping, and a small results dashboard.

## What this repo currently does

- Pulls image assignments from GS and fetches image bytes.
- Uploads full images to the cloud inference service.
- Polls cloud best-image endpoints (tent + mannequin), deduplicates repeated detections, and stores current best results.
- Runs a local GroundingDINO fallback proposal step (`GDDetection`) for person/tent candidates when cloud results are unavailable.
- Projects selected targets to lat/lon and optionally posts to an autopilot endpoint.
- Runs a local HTTP server for:
  - result dashboard UI,
  - SSE updates,
  - map trigger commands,
  - cloud push result ingestion.
- Runs a GPS+SIFT stitching pipeline to generate map outputs from session telemetry/images.

Primary entrypoint: [core.py](core.py)

## Main components

- Worker loop and scheduler: [core.py](core.py)
- GS/cloud HTTP client: [communication/work_client.py](communication/work_client.py)
- HTTP API + dashboard server: [communication/intsys_gs_api.py](communication/intsys_gs_api.py)
- Frontend UI: [frontend/index.html](frontend/index.html)
- Mapping pipeline: [mapping/main_gps_sift.py](mapping/main_gps_sift.py)
- GroundingDINO wrapper used by worker: [constructs/detection.py](constructs/detection.py)
- Lightweight websocket uploader/listener client: [client.py](client.py)

## Quick start (local Python)

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Run the worker:

```bash
python core.py --local
```

3. Open dashboard:

- http://127.0.0.1:8080

## `core.py` CLI arguments (current)

- `--local`
  - Force GS/CS defaults:
    - GS: `127.0.0.1:9000`
    - CS: `127.0.0.1:8000`
- `--gsip <host:port>` (default `127.0.0.1:9000`)
- `--csip <host:port>` (default `34.106.160.143:8000`)
- `--map-port <int>` (default `8080`)
- `--interval-seconds <float>` (default `20.0`)
- `--autopilot-ip <host>` (default `None`)
- `--autopilot-port <int>` (default `8001`)
- `--map-idle-timeout <float>` (default `20`, use `0` to disable)

Example:

```bash
python core.py --gsip 192.168.1.2:9000 --csip 10.0.0.2:8000 --map-port 8080 --interval-seconds 20 --autopilot-ip 127.0.0.1 --autopilot-port 8001 --map-idle-timeout 20
```

## Local HTTP server behavior

Served by `MapCommandHandler` from [communication/intsys_gs_api.py](communication/intsys_gs_api.py).

- `GET /` → dashboard HTML
- `GET /api/stream` → SSE stream
- `GET /api/best` → latest metadata/results for tent/mannequin + GS pulls
- `GET /export/<file>` → exported images and metadata JSON
- `POST /api/result` → cloud pushes a labeled detection payload
- `POST /` with command JSON:
  - `start`
  - `stop`
  - `trigger_mapping`

## Mapping pipeline notes

- Session ingest dir: [mapping/current_session](mapping/current_session)
- Session CSV: [mapping/current_session/metadata.csv](mapping/current_session/metadata.csv)
- Generated outputs: [mapping](mapping)
- Standalone mapping script: [mapping/main_gps_sift.py](mapping/main_gps_sift.py)

The worker auto-triggers mapping after configurable ingest idle time (`--map-idle-timeout`) when enough images are present.

## Optional websocket helper client

Run [client.py](client.py) if you want a simple websocket listener + uploader utility:

```bash
python client.py --server ws://CLOUD_HOST:8001/ws --upload-url http://CLOUD_HOST:8001/upload-image --client-id my_local_id --watch-folder ./incoming
```

## Docker state (current)

- Dockerfile exists: [Dockerfile](Dockerfile)
- Compose exists: [docker-compose.yml](docker-compose.yml)
- Entrypoint runs `python core.py`

Use:

```bash
docker-compose up --build
```

## Known issues in current state

- Legacy test script [test.py](test.py) still references old MaskRCNN/ImageNet classes and is not aligned with the current `GDDetection`-first flow.
- Makefile targets in [Makefile](Makefile) are placeholders (`not implemented`).

## Data and artifacts

- Exported images/metadata for dashboard: [export](export)
- Runtime logs: [logs](logs)
- Model files folder: [model_weights](model_weights)
- GroundingDINO package and weights:
  - [GroundingDINO](GroundingDINO)
  - [GroundingDINO/weights](GroundingDINO/weights)
