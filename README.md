# snitch 🐀

*It watches your camera and tells on everything it sees.*

Live YOLO object detection on a Hikvision IP camera (RTSP), with **two
interchangeable backends**:

| Backend | Where | Runs on |
|---------|-------|---------|
| `torch`   | dev / desktop (this Mac) | Apple MPS / NVIDIA CUDA / CPU |
| `edgetpu` | deployment appliance     | Ubuntu x86_64 + USB Coral Edge TPU |

The RTSP capture pipeline is identical on both; only the model/detector swaps.

---

## Configure the camera

```bash
cp .env.example .env      # then set CAM_PASS (and CAM_USER if not 'admin')
```

`.env` is gitignored, so your password stays local. RTSP pulls from
`Streaming/Channels/101` (main) or `102` (sub) on the camera.

---

## A. Run on macOS / desktop (torch backend)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .        # installs deps + the `snitch` command

snitch                 # live window, sub-stream, auto GPU (MPS)
snitch --stream 1      # full-res main stream
snitch --classes 0     # people only
snitch --headless      # no window; print detections
```

First run downloads `yolov8n.pt` (~6 MB). Press `q` to quit.

### Modes

| Mode | Flag | Output |
|------|------|--------|
| Live window | *(default)* | annotated preview with boxes, labels, FPS |
| Text / data export | `--headless` | detections to stdout (no GUI needed) |

### Region of interest (ROI)

Gate detection to a sub-window of the frame — anything whose box center falls
outside the region is ignored. Inference still runs on the full frame, so
accuracy is unchanged; the ROI just filters what gets reported and drawn.

```bash
# Interactive: in the live window, press 'r' to drag a box, 'x' to clear.
snitch                          # the box is saved to roi.json, reloaded next launch
snitch --roi 0.25 0 0.75 1      # set it from the CLI (fractions of the frame)
snitch --roi-file front.json    # use a different ROI file
```

Press `v` in the window to toggle a preview that zooms the ROI region to fill
the whole window — exactly the pixels snitch is judging, scaled up. Press `v`
again to return to the full frame with the ROI outline. It's a view-only toggle:
it doesn't change detection, `roi.json`, or headless output.

ROIs are stored as fractions of the frame, so the same region works whether you
run the main or the sub stream. `--roi` overrides whatever is in the file. The
ROI is applied in `--headless` mode too (using the CLI value or the saved file).

Text mode is meant for CLI use, SSH, logging, and piping into other tools.
It reports timestamp, per-class counts, confidence, and bounding-box coords:

```bash
snitch --headless                     # human-readable summary
#  22:14:56  chair×1  person×2                  | 14.2 fps

snitch --headless --only-detections   # skip frames with nothing
snitch --headless --every 1           # throttle to ~1 line/sec

snitch --headless --format json        # NDJSON, one object per frame
#  {"t":1749500096.1,"iso":"2026-06-09T22:14:56","fps":14.2,"count":2,
#   "objects":[{"cls":"person","conf":0.91,"box":[10,20,110,300]}, ...]}

snitch --headless --format json | jq '.objects[].cls'   # pipe
snitch --headless --format json > detections.ndjson     # log to file
```

JSON mode works identically on the Coral backend, so the appliance can stream
structured detections over a pipe/socket with no display attached.

---

## B. Deploy on Ubuntu x86_64 + USB Coral (edgetpu backend)

The Coral runs **int8 TFLite models compiled for the Edge TPU**. The flow is:
export `.pt` → compile to `_edgetpu.tflite` → run.

```bash
# 1. Install runtime, compiler, and Python deps
bash setup_coral_ubuntu.sh
# then UNPLUG/REPLUG the Coral, and re-login if newly added to 'plugdev'

source .venv-coral/bin/activate
cp .env.example .env             # set CAM_PASS

# 2. Compile YOLO for the Edge TPU (x86_64 only)
python export_edgetpu.py --model yolov8n.pt --imgsz 256

# 3. Run live detection on the Coral
snitch \
  --model yolov8n_saved_model/yolov8n_full_integer_quant_edgetpu.tflite \
  --imgsz 256
```

`--backend` is `auto`: a `*_edgetpu.tflite` model selects the Coral path
automatically.

### Tuning the Coral

- **Input size** (`--imgsz`): smaller is faster. Try `192` / `224` / `256`.
  Inference size must match the value used at export.
- **Clock**: `setup_coral_ubuntu.sh` installs `libedgetpu1-std`. Swap to
  `libedgetpu1-max` for higher clock (faster, hotter, more power).
- **tflite-runtime wheel**: if pip has no wheel for your Python version, grab a
  matching one from <https://github.com/feranick/TFlite-builds/releases>
  (the script prints this hint and installs everything else).

### ARM Ubuntu (Jetson / Pi)

The runtime works, but `edgetpu-compiler` is **x86_64-Linux only** — compile
the model on an x86 box (or Docker/Colab) and copy the `_edgetpu.tflite` over.

---

## Files

| File | Purpose |
|------|---------|
| `snitch.py` | RTSP reader + YOLO inference; `torch`/`edgetpu` backends |
| `pyproject.toml` | packaging; provides the `snitch` console command |
| `export_edgetpu.py` | Compile a `.pt` to an Edge TPU TFLite (run on x86 Ubuntu) |
| `setup_coral_ubuntu.sh` | Install Coral runtime + compiler + venv on Ubuntu |
| `requirements.txt` | torch/desktop deps |
| `requirements-coral.txt` | Coral deployment deps |
| `.env.example` | camera config template |
| `roi.json` | saved region of interest (created on first ROI select; gitignored) |
