#!/usr/bin/env bash
#
# Set up Ubuntu x86_64 to run YOLO on a USB Coral Edge TPU.
#
# Installs:
#   - Coral apt repo + GPG key
#   - libedgetpu1-std        (Edge TPU runtime + udev rules)
#   - edgetpu-compiler       (x86_64 only; needed by export_edgetpu.py)
#   - a Python venv with requirements-coral.txt
#
# After running, UNPLUG and REPLUG the Coral so the new udev rules apply,
# then test with:  python export_edgetpu.py  &&  snitch ...
#
# Run:  bash setup_coral_ubuntu.sh
set -euo pipefail

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "WARNING: this box is $(uname -m), not x86_64."
  echo "The edgetpu-compiler is x86_64-Linux only. The runtime (libedgetpu)"
  echo "still works on ARM, but you must compile the model on an x86 machine."
  read -rp "Continue with runtime-only install? [y/N] " ans
  [[ "${ans:-N}" =~ ^[Yy]$ ]] || exit 1
  SKIP_COMPILER=1
else
  SKIP_COMPILER=0
fi

echo "==> Adding Coral apt repository"
sudo install -d -m 0755 /usr/share/keyrings
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg
echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list >/dev/null

sudo apt-get update

echo "==> Installing Edge TPU runtime (std clock)"
# Use libedgetpu1-max instead for higher clock = faster but hotter/more power.
sudo apt-get install -y libedgetpu1-std

if [[ "$SKIP_COMPILER" -eq 0 ]]; then
  echo "==> Installing edgetpu-compiler"
  sudo apt-get install -y edgetpu-compiler
fi

# The Coral is a USB device; non-root access needs the plugdev group.
echo "==> Ensuring $USER is in the plugdev group"
sudo groupadd -f plugdev
sudo usermod -aG plugdev "$USER" || true

echo "==> Creating Python venv (.venv-coral) and installing deps"
sudo apt-get install -y python3-venv python3-pip >/dev/null
python3 -m venv .venv-coral
# shellcheck disable=SC1091
source .venv-coral/bin/activate
pip install --upgrade pip

# tflite-runtime wheels are not published for every Python version. Try the
# normal install first; if it fails, fall back to feranick's maintained index
# (https://github.com/feranick/TFlite-builds) which has builds for newer Python.
if ! pip install -r requirements-coral.txt; then
  echo
  echo "tflite-runtime install failed for $(python3 --version)."
  echo "Installing other deps, then fetch a matching tflite-runtime wheel from:"
  echo "  https://github.com/feranick/TFlite-builds/releases"
  echo "and:  pip install <downloaded-wheel>.whl"
  grep -v '^tflite-runtime' requirements-coral.txt | pip install -r /dev/stdin
fi

# Register the `snitch` console command (deps already installed above).
pip install -e . --no-deps

echo
echo "================================================================"
echo "Done. Next steps:"
echo "  1) UNPLUG and REPLUG the Coral USB Accelerator now."
echo "  2) If you were just added to 'plugdev', log out/in (or reboot)."
echo "  3) source .venv-coral/bin/activate"
echo "  4) cp .env.example .env   # and set your camera password"
echo "  5) python export_edgetpu.py --model yolov8n.pt --imgsz 256"
echo "  6) snitch \\"
echo "       --model yolov8n_saved_model/yolov8n_full_integer_quant_edgetpu.tflite \\"
echo "       --imgsz 256"
echo "================================================================"
