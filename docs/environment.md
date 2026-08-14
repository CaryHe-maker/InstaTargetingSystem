# Runtime Environment

本项目当前按 GPU-only 方式运行官方 HiT，不提供 CPU 兼容路径。以下版本已在本机环境中安装并确认 CUDA 可用。

## Hardware and Python

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM
- NVIDIA driver: 566.07
- Driver-supported CUDA: 12.7
- Python: 3.12.13
- Virtual environment: `.venv`

## Runtime Dependencies

| Package | Installed version | Purpose |
|---|---:|---|
| `torch` | `2.6.0+cu124` | HiT GPU inference and training |
| `torchvision` | `0.21.0+cu124` | PyTorch vision operators |
| `timm` | `0.5.4` | HiT backbone compatibility |
| `yacs` | `0.1.8` | Official HiT experiment configuration |
| `easydict` | `1.13` | Official HiT configuration objects |
| `h5py` | `3.16.0` | AirSim360 depth HDF5 reader |
| `scipy` | `1.18.0` | Official HiT utility dependency |
| `pandas` | `3.0.5` | Official HiT data and result utilities |
| `numpy` | `2.5.1` | Image, geometry, and depth processing |
| `PyYAML` | `6.0.3` | Project and HiT YAML loading |
| `gdown` | `6.1.0` | Official checkpoint download helper |
| `tensorboard` | `2.21.0` | Official checkpoint/training metadata imports |

The NVIDIA driver may support a newer CUDA version than the PyTorch runtime. This is expected: the installed CUDA 12.4 PyTorch wheel runs on the CUDA 12.7-capable driver.

## Installation

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install `
  "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" `
  --index-url https://download.pytorch.org/whl/cu124

.\.venv\Scripts\python.exe -m pip install `
  "timm==0.5.4" "yacs>=0.1.8" "easydict>=1.13" `
  "h5py>=3" "scipy>=1" "pandas>=2" "gdown>=5"
```

For development tests, also install the project development extra:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Official HiT Source

The official repository is checked out at `third_party/HiT`:

```powershell
git clone --depth 1 https://github.com/kangben258/HiT.git third_party/HiT
```

Both `configs/RGBonly.yaml` and `configs/RGBD.yaml` use:

```yaml
model:
  backend: pytorch
  variant: HiT_Small
  source: ../third_party/HiT
  weights: ../models/hit_small.pth
  device: cuda
```

RGB-only and RGBD both use CUDA FP32 with the official checkpoint. RGBD sends the
depth-edge-enhanced RGB image to the same HiT network. The adapter supports CUDA autocast,
but the official HiT_Small checkpoint is kept on FP32 because its corner head is numerically
unstable in FP16 and can return non-finite boxes.

## Official Checkpoint

`models/hit_small.pth` is the official `HiT_Small/VT_ep1500.pth.tar` checkpoint from the model directory linked by the HiT authors:

- Model directory: `https://drive.google.com/drive/folders/15VTIJnUtJTdU6TcmGOixSEcErYV-h_xL`
- File ID: `14I3VyqrRre6KThBBAqS2c27EkHmXM544`
- Local size: 137,177,837 bytes

Download command:

```powershell
.\.venv\Scripts\python.exe -m gdown `
  14I3VyqrRre6KThBBAqS2c27EkHmXM544 `
  -O models\hit_small.pth
```

The checkpoint contains official training-statistics objects. PyTorch 2.6 changed `torch.load()` to default to `weights_only=True`, so the project adapter explicitly uses `weights_only=False` for this configured, trusted official checkpoint.

## Verification

Confirm CUDA before running tracking or training:

```powershell
.\.venv\Scripts\python.exe -c `
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected core output on the current machine:

```text
2.6.0+cu124 12.4 True
NVIDIA GeForce RTX 4060 Laptop GPU
```
