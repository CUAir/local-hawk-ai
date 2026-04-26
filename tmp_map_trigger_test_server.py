#!/usr/bin/env python3
from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from communication.intsys_gs_api import MapCommandHandler, ResultStore
from mapping.main_gps_sift import GpsSiftPipeline


ROOT = Path(__file__).resolve().parent
SESSION_DIR = ROOT / "mapping" / "current_session"
SESSION_IMAGES = SESSION_DIR / "images"
SESSION_CSV = SESSION_DIR / "metadata.csv"
OUTPUT_DIR = ROOT / "mapping"


class TestMapper:
    def __init__(self) -> None:
        self.mapping_running = False
        self.mapping_result = None
        self.mapping = False

    def trigger_pipeline(self) -> None:
        if self.mapping_running:
            print("[test-mapper] trigger ignored: already running")
            return
        threading.Thread(target=self._run_pipeline, daemon=True).start()
        print("[test-mapper] trigger accepted")

    def _run_pipeline(self) -> None:
        self.mapping_running = True
        print("[test-mapper] pipeline start")
        try:
            pipeline = GpsSiftPipeline(output_dir=str(OUTPUT_DIR), verbose=False)
            out = pipeline.run(str(SESSION_IMAGES), str(SESSION_CSV))
            self.mapping_result = out
            print(f"[test-mapper] pipeline done -> {out}")
        except Exception as exc:
            print(f"[test-mapper] pipeline error: {exc}")
        finally:
            self.mapping_running = False
            print("[test-mapper] pipeline stopped")


def main() -> None:
    mapper = TestMapper()
    MapCommandHandler.mapper = mapper
    MapCommandHandler.result_store = ResultStore()
    server = ThreadingHTTPServer(("0.0.0.0", 8080), MapCommandHandler)
    print("[test-mapper] listening on http://0.0.0.0:8080")
    server.serve_forever()


if __name__ == "__main__":
    main()
