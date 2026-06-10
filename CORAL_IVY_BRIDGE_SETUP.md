# Coral USB Accelerator on Ivy Bridge — Setup Notes

This machine runs an Intel Core i7-3720QM (Ivy Bridge, 3rd-gen). Ivy Bridge supports
AVX but **not AVX2**. All official Google Coral packages and most community builds are
compiled with AVX2 and will crash with `Illegal instruction (SIGILL)` on this CPU.

Everything below had to be compiled from source with `-march=ivybridge -mno-avx2`.

---

## 1. OS and hardware

| | |
|---|---|
| CPU | Intel Core i7-3720QM (Ivy Bridge) |
| OS | Ubuntu 24.04 |
| Device | Google Coral USB Accelerator |
| USB ID (boot) | `1a6e:089a` (pre-firmware) |
| USB ID (running) | `18d1:9302` (post-firmware, firmware is uploaded by libedgetpu on first use) |

---

## 2. Standard apt packages (no recompile needed)

```bash
# Add Coral apt repo
echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg

sudo apt update
sudo apt install -y libedgetpu1-std edgetpu-compiler libusb-1.0-0-dev python3.12-dev
```

`libedgetpu1-std` installs the runtime (`.so`) and udev rules. We immediately replace
the `.so` with a custom build (see §4). The udev rules it drops at
`/lib/udev/rules.d/60-libedgetpu1-std.rules` are fine and stay in place.

---

## 3. Python virtualenv

```bash
cd ~/dev/snitch
python3 -m venv .venv-coral
source .venv-coral/bin/activate
pip install -r requirements-coral.txt
```

`requirements-coral.txt` pulls CPU-only PyTorch (saves ~2 GB) and leaves
`tflite-runtime` to be installed manually from the custom wheel (see §5).

---

## 4. Build libedgetpu from source (no AVX2)

The system `libedgetpu1-std` is compiled with AVX2. We replace it.

### 4a. Install Bazel 6.5.0

The libedgetpu build requires Bazel **6.x**. Bazel 7+ removed rules that libedgetpu
still uses.

```bash
sudo mkdir -p /opt/bazel6
wget -O /tmp/bazel-6.5.0 \
  https://github.com/bazelbuild/bazel/releases/download/6.5.0/bazel-6.5.0-linux-x86_64
chmod +x /tmp/bazel-6.5.0
sudo mv /tmp/bazel-6.5.0 /opt/bazel6/bin/bazel
```

### 4b. Clone libedgetpu

```bash
sudo mkdir -p /opt/libedgetpu-build
sudo chown $USER:$USER /opt/libedgetpu-build
git clone https://github.com/google-coral/libedgetpu /opt/libedgetpu-build
```

### 4c. Patch .bazelrc to disable AVX2

Append to `/opt/libedgetpu-build/.bazelrc`:

```
startup --output_user_root=/opt/bazel-cache
build:linux --copt=-march=ivybridge
build:linux --copt=-mno-avx2
```

Also create the Bazel cache dir:

```bash
sudo mkdir -p /opt/bazel-cache
sudo chown $USER:$USER /opt/bazel-cache
```

### 4d. Build

```bash
cd /opt/libedgetpu-build
make BAZEL=/opt/bazel6/bin/bazel libedgetpu-direct
```

This takes a while (downloads TF source, compiles everything).
Output: `out/direct/k8/libedgetpu.so.1.0`

### 4e. Install

```bash
sudo cp /opt/libedgetpu-build/out/direct/k8/libedgetpu.so.1.0 \
        /usr/lib/x86_64-linux-gnu/libedgetpu.so.1.0
sudo ln -sfn libedgetpu.so.1.0 /usr/lib/x86_64-linux-gnu/libedgetpu.so.1
```

> **Important**: after any `apt upgrade` that touches `libedgetpu1-std`, re-run the
> `cp` and `ln` commands above — apt will restore the AVX2 binary and the symlink.
> The symptom is "EdgeTpuDelegateForCustomOp failed to invoke" (no SIGILL, just a
> RuntimeError) because the failing inference code path happens to not be the first
> AVX2 instruction hit.

---

## 5. Build tflite-runtime from source (no AVX2)

The community Python 3.12 wheels for `tflite-runtime` (feranick builds and PyPI) are
compiled with AVX2. We build from TensorFlow source using CMake.

TF source is already on disk after the libedgetpu build (Bazel downloaded it).
Find it:

```bash
find /opt/bazel-cache -name "lite" -path "*/tensorflow/lite" -type d 2>/dev/null | head -3
# → something like /opt/bazel-cache/.../external/org_tensorflow
```

Note the path (call it `$TF_SRC`).

### 5a. Build dependencies

```bash
sudo apt install -y cmake patchelf
pip install wheel
```

### 5b. Build

```bash
TF_SRC=/opt/bazel-cache/<hash>/external/org_tensorflow   # fill in actual path

BUILD_FLAGS="-march=ivybridge -mno-avx2 -I${TF_SRC} -I${TF_SRC}/tensorflow/lite/tools/pip_package"

cd ${TF_SRC}/tensorflow/lite/tools/pip_package
BUILD_NUM_JOBS=4 \
  CUSTOM_BAZEL_FLAGS="" \
  CFLAGS="$BUILD_FLAGS" CXXFLAGS="$BUILD_FLAGS" \
  bash build_pip_package_with_cmake.sh native
```

The wheel lands in a `gen/tflite_pip/` subdirectory.

### 5c. Install

```bash
source ~/dev/snitch/.venv-coral/bin/activate
pip install ${TF_SRC}/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/dist/tflite_runtime-*.whl
```

---

## 6. Export the YOLOv8 model for Edge TPU

```bash
cd ~/dev/snitch
source .venv-coral/bin/activate
python export_edgetpu.py --imgsz 256
```

Output: `yolov8n_saved_model/yolov8n_full_integer_quant_edgetpu.tflite`
(3.8 MB, 256 ops, all mapped to Edge TPU)

---

## 7. Run

```bash
source .venv-coral/bin/activate
snitch --model yolov8n_saved_model/yolov8n_full_integer_quant_edgetpu.tflite \
       --imgsz 256 --headless --only-detections --format json
```

---

## Troubleshooting quick-reference

| Symptom | Cause | Fix |
|---|---|---|
| `Illegal instruction` on `import tflite_runtime` | tflite-runtime wheel has AVX2 | Rebuild from source (§5) |
| `Illegal instruction` on `load_delegate` | libedgetpu has AVX2 | Rebuild from source (§4) |
| `EdgeTpuDelegateForCustomOp failed to invoke` | libedgetpu.so.1 symlink points to old AVX2 backup | `sudo ln -sfn libedgetpu.so.1.0 /usr/lib/x86_64-linux-gnu/libedgetpu.so.1` |
| Device shows `1a6e:089a` in lsusb | Firmware not yet uploaded | Normal on cold boot; uploading happens on first `load_delegate` call |
| Device shows `18d1:9302` in lsusb | Firmware loaded, device ready | Normal operating state |
