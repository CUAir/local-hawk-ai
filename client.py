"""Local client to upload images to cloud and listen for save/best-image commands.

Usage:
  python client.py --server ws://<CLOUD_HOST>:8001/ws --upload-url http://<CLOUD_HOST>:8001/upload-image --client-id my_local_id

This script demonstrates connecting to the cloud websocket, registering, uploading images, and saving requested images to `export/`.
"""
import argparse
import asyncio
import base64
import os
import time
from pathlib import Path

import requests
import websockets

EXPORT_DIR = Path(__file__).parent / "export"
EXPORT_DIR.mkdir(exist_ok=True)


async def listen_ws(uri: str, client_id: str):
    async with websockets.connect(uri) as ws:
        # send register
        await ws.send('{"type":"register","client_id":"%s"}' % client_id)
        print("Connected and registered to cloud websocket")
        async for msg in ws:
            try:
                import json

                data = json.loads(msg)
            except Exception:
                continue
            if data.get("type") == "save":
                fn = data.get("filename") or f"cloud_{int(time.time()*1000)}.jpg"
                b64 = data.get("image_b64")
                if b64:
                    path = EXPORT_DIR / fn
                    with open(path, "wb") as f:
                        f.write(base64.b64decode(b64))
                    print("Saved requested image:", path)
            elif data.get("type") == "best_images":
                items = data.get("items", [])
                for it in items:
                    fn = it.get("filename") or f"best_{it.get('client_id')}_{int(time.time()*1000)}.jpg"
                    b64 = it.get("image_b64")
                    if b64:
                        path = EXPORT_DIR / fn
                        with open(path, "wb") as f:
                            f.write(base64.b64decode(b64))
                        print("Saved best image from cloud:", path)


def upload_image(upload_url: str, client_id: str, file_path: str):
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "image/jpeg")}
        data = {"client_id": client_id}
        r = requests.post(upload_url, files=files, data=data)
    try:
        print("Upload response:", r.json())
    except Exception:
        print("Upload status:", r.status_code, r.text)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="WebSocket server ws://host:port/ws")
    parser.add_argument("--upload-url", required=True, help="HTTP upload URL http://host:port/upload-image")
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--watch-folder", help="Optional folder to watch and auto-upload images")
    args = parser.parse_args()

    # start websocket listener
    ws_task = asyncio.create_task(listen_ws(args.server, args.client_id))

    # optionally watch a folder for images to upload
    if args.watch_folder:
        folder = Path(args.watch_folder)
        seen = set()
        while True:
            for p in folder.glob("*.jpg"):
                if str(p) in seen:
                    continue
                try:
                    upload_image(args.upload_url, args.client_id, str(p))
                    seen.add(str(p))
                except Exception as e:
                    print("Upload error:", e)
            await asyncio.sleep(1.0)
    else:
        await ws_task


if __name__ == "__main__":
    asyncio.run(main())
