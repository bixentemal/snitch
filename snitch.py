"""snitch — it watches your camera and tells on everything it sees.

Live YOLO object detection on a Hikvision RTSP stream. Pulls H.264/H.265
video from the camera over RTSP, runs an Ultralytics YOLO model on each
frame, and either shows an annotated preview window or rats out every
detection to stdout.

Usage:
    snitch                      # defaults from .env
    snitch --model yolov8s.pt   # bigger/more accurate model
    snitch --stream 1           # main (full-res) stream
    snitch --conf 0.4           # confidence threshold
    snitch --classes 0          # only detect 'person' (COCO id 0)

Region of interest — gate detection to a sub-window of the frame:
    snitch                                  # press 'r' to drag a box, 'x' to clear
    snitch --roi 0.25 0 0.75 1              # fractions of the frame (overrides file)
    # The interactive box is saved to roi.json and reloaded on next launch.
    # Press 'v' to toggle a preview of just the ROI region (what snitch sees).

Text / data-export mode (no window — for CLI, SSH, logging, piping):
    snitch --headless                       # human-readable summary
    snitch --headless --only-detections     # skip empty frames
    snitch --headless --every 1             # at most one line/sec
    snitch --headless --format json         # NDJSON per frame
    snitch --headless --format json | jq .  # pipe into jq
    snitch --headless --format json > log.ndjson   # log to file
"""

import argparse
import json
import os
import threading
import time
from collections import Counter
from urllib.parse import quote

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()


def build_rtsp_url(stream: int) -> str:
    """Construct the Hikvision RTSP URL from environment credentials."""
    host = os.getenv("CAM_HOST", "192.168.1.100")
    user = os.getenv("CAM_USER", "admin")
    password = os.getenv("CAM_PASS", "")
    port = os.getenv("CAM_RTSP_PORT", "554")
    if not password:
        raise SystemExit(
            "No CAM_PASS set. Copy .env.example to .env and fill in your "
            "camera credentials."
        )
    # Channel 1, stream 01 = main; stream 02 = sub.
    channel = f"10{stream}"
    # URL-encode credentials so special characters in the password work.
    return (
        f"rtsp://{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:{port}/Streaming/Channels/{channel}"
    )


class RTSPReader:
    """Background thread that always holds the most recent decoded frame.

    Decoding RTSP is slower than the camera produces frames, so a naive
    read loop falls behind and shows stale video. This grabber keeps only
    the latest frame, so YOLO always works on near-live data.
    """

    def __init__(self, url: str):
        self.url = url
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def _open(self) -> bool:
        # Force TCP transport — far more reliable than UDP over wifi.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp"
        )
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return self.cap.isOpened()

    def start(self):
        if not self._open():
            raise SystemExit(
                "Could not open RTSP stream. Check credentials, IP, and "
                "that the camera is reachable."
            )
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        fail = 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                fail += 1
                if fail > 50:  # stream died — try to reconnect
                    print("Stream dropped, reconnecting...")
                    self.cap.release()
                    time.sleep(2)
                    self._open()
                    fail = 0
                else:
                    time.sleep(0.02)
                continue
            fail = 0
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()


def emit_detections(r, ts, fps, fmt, only_detections) -> bool:
    """Print one frame's detections to stdout. Returns True if it emitted.

    fmt='json'   -> one NDJSON object per line (pipe into jq, a file, etc.)
    fmt='pretty' -> human-readable per-class counts with a clock + fps.
    """
    names = r.names
    boxes = r.boxes
    n = len(boxes)
    if only_detections and n == 0:
        return False

    objs = []
    if n:
        for c, cf, box in zip(
            boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
        ):
            objs.append(
                {
                    "cls": names[int(c)],
                    "conf": round(float(cf), 3),
                    "box": [int(v) for v in box],  # [x1, y1, x2, y2]
                }
            )

    if fmt == "json":
        record = {
            "t": round(ts, 3),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
            "fps": round(fps, 1),
            "count": n,
            "objects": objs,
        }
        print(json.dumps(record), flush=True)
    else:  # pretty
        clock = time.strftime("%H:%M:%S", time.localtime(ts))
        if objs:
            counts = Counter(o["cls"] for o in objs)
            summary = "  ".join(f"{k}×{v}" for k, v in sorted(counts.items()))
        else:
            summary = "(nothing)"
        print(f"{clock}  {summary:<40} | {fps:4.1f} fps", flush=True)
    return True


def load_roi(path: str):
    """Load a normalized ROI [x1, y1, x2, y2] (fractions 0-1) from a JSON file.

    Returns None when the file is missing or doesn't hold a valid ROI.
    """
    try:
        with open(path) as f:
            roi = json.load(f).get("roi")
        if roi and len(roi) == 4:
            return [float(v) for v in roi]
    except (FileNotFoundError, ValueError, TypeError):
        pass
    return None


def save_roi(path: str, roi) -> None:
    """Persist a normalized ROI to a JSON file (roi=None clears it)."""
    with open(path, "w") as f:
        json.dump({"roi": roi}, f)


def roi_pixels(roi, w: int, h: int):
    """Convert a normalized ROI to integer pixel corners for a w×h frame."""
    if roi is None:
        return None
    x1, y1, x2, y2 = roi
    return [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)]


def filter_to_roi(r, roi_px):
    """Keep only detections whose box center falls inside the ROI rectangle."""
    if roi_px is None:
        return r
    x1, y1, x2, y2 = roi_px
    keep = []
    for i, (bx1, by1, bx2, by2) in enumerate(r.boxes.xyxy.tolist()):
        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            keep.append(i)
    return r[keep]


def select_roi(window: str, frame):
    """Drag a box in the preview window. Returns a normalized ROI, or None.

    ENTER/SPACE confirms the selection; 'c' (or an empty box) cancels.
    """
    h, w = frame.shape[:2]
    x, y, bw, bh = cv2.selectROI(window, frame, showCrosshair=True)
    if bw == 0 or bh == 0:
        return None
    return [x / w, y / h, (x + bw) / w, (y + bh) / h]


def is_edgetpu_model(model_path: str) -> bool:
    """Coral Edge TPU models are compiled TFLite files."""
    return model_path.endswith("_edgetpu.tflite")


def pick_device(backend: str) -> str:
    """Choose the inference device for the active backend.

    The Edge TPU is driven through the TFLite delegate, so the host device
    is just 'cpu' — the Coral does the actual compute. For the torch
    backend we prefer Apple MPS or CUDA when available.
    """
    if backend == "edgetpu":
        return "cpu"
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(
        prog="snitch",
        description="snitch — watches a Hikvision camera and tells on what it sees",
    )
    p.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO weights: a .pt for torch, or *_edgetpu.tflite for Coral",
    )
    p.add_argument(
        "--backend",
        choices=["auto", "torch", "edgetpu"],
        default="auto",
        help="auto picks edgetpu when --model is *_edgetpu.tflite, else torch",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="inference size; for Coral, match the size used at export",
    )
    p.add_argument(
        "--stream",
        type=int,
        default=int(os.getenv("CAM_STREAM", "2")),
        choices=[1, 2],
        help="1=main (full-res), 2=sub (lighter, smoother)",
    )
    p.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    p.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="restrict to COCO class ids (e.g. 0 for person)",
    )
    p.add_argument(
        "--device", default=None, help="mps / cpu / 0 (auto if unset; torch only)"
    )
    p.add_argument(
        "--roi",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="gate detection to a sub-window, as fractions 0-1 "
        "(e.g. 0.25 0 0.75 1). Overrides --roi-file.",
    )
    p.add_argument(
        "--roi-file",
        default="roi.json",
        help="JSON file to load/save the interactive ROI (default: roi.json)",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="no preview window; export detections to stdout",
    )
    p.add_argument(
        "--format",
        choices=["pretty", "json"],
        default="pretty",
        help="text-mode output format (with --headless)",
    )
    p.add_argument(
        "--every",
        type=float,
        default=0.0,
        help="text mode: min seconds between emits (0 = every frame)",
    )
    p.add_argument(
        "--only-detections",
        action="store_true",
        help="text mode: skip frames where nothing is detected",
    )
    return p.parse_args()


def main():
    args = parse_args()
    backend = args.backend
    if backend == "auto":
        backend = "edgetpu" if is_edgetpu_model(args.model) else "torch"

    device = args.device or pick_device(backend)
    print(f"Loading {args.model} (backend={backend}, device={device})...")
    model = YOLO(args.model, task="detect")

    roi = args.roi if args.roi else load_roi(args.roi_file)
    if roi:
        print(f"ROI active: {[round(v, 3) for v in roi]} (fractions of frame)")

    url = build_rtsp_url(args.stream)
    safe_url = url.split("@")[-1]
    print(f"Connecting to rtsp://***@{safe_url}")
    reader = RTSPReader(url).start()

    # Wait for the first frame.
    print("Waiting for first frame...")
    t0 = time.time()
    while reader.read() is None:
        if time.time() - t0 > 15:
            reader.stop()
            raise SystemExit("Timed out waiting for video.")
        time.sleep(0.1)
    if args.headless:
        print("Streaming. Press Ctrl+C to quit.")
    else:
        print(
            "Streaming. In the window: 'r' select ROI, 'v' toggle ROI view, "
            "'x' clear ROI, 'q' quit (or Ctrl+C)."
        )

    prev = time.time()
    fps = 0.0
    last_emit = 0.0
    crop_view = False  # toggle: show only the ROI region, filling the window
    try:
        while True:
            frame = reader.read()
            if frame is None:
                time.sleep(0.01)
                continue

            predict_kwargs = dict(
                conf=args.conf,
                classes=args.classes,
                device=device,
                verbose=False,
            )
            if args.imgsz:
                predict_kwargs["imgsz"] = args.imgsz
            results = model.predict(frame, **predict_kwargs)
            roi_px = roi_pixels(roi, frame.shape[1], frame.shape[0])
            r = filter_to_roi(results[0], roi_px)

            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            if args.headless:
                if now - last_emit >= args.every:
                    if emit_detections(
                        r, now, fps, args.format, args.only_detections
                    ):
                        last_emit = now
            else:
                annotated = r.plot()
                if crop_view and roi_px:
                    x1, y1, x2, y2 = roi_px
                    crop = annotated[y1:y2, x1:x2]
                    if crop.size == 0:  # degenerate ROI — show the full frame
                        display = annotated
                    else:  # zoom the ROI to fill the original window size
                        h, w = annotated.shape[:2]
                        display = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
                    tag = "  [ROI view]"
                else:
                    display = annotated
                    if roi_px:
                        cv2.rectangle(
                            display,
                            (roi_px[0], roi_px[1]),
                            (roi_px[2], roi_px[3]),
                            (0, 200, 255),
                            2,
                        )
                    tag = ""
                cv2.putText(
                    display,
                    f"{fps:.1f} FPS  {len(r.boxes)} obj{tag}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("snitch 🐀", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    sel = select_roi("snitch 🐀", frame)
                    if sel:
                        roi = sel
                        save_roi(args.roi_file, roi)
                        print(
                            f"ROI saved to {args.roi_file}: "
                            f"{[round(v, 3) for v in roi]}"
                        )
                elif key == ord("x") and roi is not None:
                    roi = None
                    save_roi(args.roi_file, None)
                    print(f"ROI cleared ({args.roi_file})")
                elif key == ord("v"):
                    crop_view = not crop_view
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
