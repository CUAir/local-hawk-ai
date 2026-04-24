Local component (local-hawk-ai)

This script demonstrates connecting to the cloud WebSocket and uploading images to the cloud server.

Quick start (inside `local-hawk-ai` virtualenv):

1) Install deps:

```bash
python -m pip install -r requirements.txt
```

2) Run the client, pointing at your cloud host:

```bash
python client.py --server ws://CLOUD_HOST:8001/ws --upload-url http://CLOUD_HOST:8001/upload-image --client-id my_local_id --watch-folder ./incoming
```

Behavior:
- The client opens a WebSocket connection to the cloud and registers with `client_id`.
- When an image is uploaded to the cloud and becomes the new best, the cloud sends a `save` message containing the image; the client saves it into `export/`.
- The cloud also periodically (every 10s) sends current best-images; the client saves those to `export/` as well.

Notes:
- The client includes a `--watch-folder` option to auto-upload any new `*.jpg` files found in that folder. Integration with your GS pipeline can call the same HTTP upload endpoint.
