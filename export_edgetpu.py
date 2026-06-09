"""Export a YOLO model to a Coral Edge TPU TFLite file.

Run this ON THE x86_64 Ubuntu box (the edgetpu_compiler is x86_64-Linux
only). It produces a compiled model you then point snitch at:

    python export_edgetpu.py --model yolov8n.pt --imgsz 256
    # -> yolov8n_full_integer_quant_edgetpu.tflite (inside yolov8n_saved_model/)

Then:

    snitch --model yolov8n_saved_model/yolov8n_full_integer_quant_edgetpu.tflite --imgsz 256

Notes:
- Smaller --imgsz (192/224/256) = faster on the Coral. 256 is a good start.
- Export runs an int8 quantization calibration pass; the first run downloads
  a small calibration dataset automatically.
- Requires the edgetpu_compiler to be installed (see setup_coral_ubuntu.sh).
"""

import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Export YOLO -> Edge TPU TFLite")
    p.add_argument("--model", default="yolov8n.pt", help="source .pt weights")
    p.add_argument(
        "--imgsz",
        type=int,
        default=256,
        help="square input size baked into the model (192/224/256...)",
    )
    args = p.parse_args()

    model = YOLO(args.model)
    path = model.export(format="edgetpu", imgsz=args.imgsz)
    print("\nExported Edge TPU model:")
    print(f"  {path}")
    print("\nRun the live stream with it:")
    print(f"  snitch --model {path} --imgsz {args.imgsz}")


if __name__ == "__main__":
    main()
