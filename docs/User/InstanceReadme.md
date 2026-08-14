# AirSim360 instance 编号查找与使用

## 1. instance ID 是什么

AirSim360 的 `instance\panorama_N.png` 是实例分割图。同一物体在图中的所有像素使用同一个 instance ID，不同物体即使属于同一语义类别，也有不同的 ID。

本项目对 RGBA instance PNG 的解码规则是：

```text
instanceId = R | (G << 8) | (B << 16)
```

也就是 `instanceId = R + 256 * G + 65536 * B`。alpha 通道不参与 instance ID。`semanticId` 表示“汽车、建筑、路锥”等类别，不能替代 `instanceId`。

## 2. 列出训练序列首帧的所有物体

在仓库根目录执行：

```powershell
& ".venv\Scripts\python.exe" "tools\run_airsim360_dataset.py" `
  --dataset-root "data\airsim360\nyc_sample" `
  --config "configs\RGBonly.yaml" `
  --list-instances
```

该命令只读取首帧并输出 JSON，不运行跟踪。为了方便检索，可以保存清单：

```powershell
& ".venv\Scripts\python.exe" "tools\run_airsim360_dataset.py" `
  --dataset-root "data\airsim360\nyc_sample" `
  --config "configs\RGBonly.yaml" `
  --list-instances | Set-Content -Encoding utf8 "artifacts\nyc_instances.json"
```

每条记录包含：

| 字段 | 含义 |
|---|---|
| `instanceId` | 跟踪和训练实际使用的整数 ID |
| `semanticId` / `semanticName` | 该实例主要像素对应的语义类别 |
| `pixels` | 首帧中该实例占用的像素数，清单按此值从大到小排序 |
| `bbox` | 首帧 ERP 图上的 `xPx, yPx, widthPx, heightPx` |
| `frameFraction` | 该实例占整张首帧图像的比例 |

## 3. 找到“某一个具体物体”的 ID

先按语义类别缩小范围。例如查找 `car` 类，类别名称以本序列的 `semantic_lists.txt` 为准：

```powershell
$instances = Get-Content -Raw "artifacts\nyc_instances.json" | ConvertFrom-Json
$instances | Where-Object semanticName -eq "car" |
  Select-Object instanceId, pixels, semanticName, bbox |
  Format-Table -AutoSize
```

然后在 `raw\panorama_0.png` 中查看目标位置，并与候选记录的 `bbox` 对照。`bbox.xPx/yPx` 是左上角，`widthPx/heightPx` 是宽高。对于跨越全景图左右接缝的物体，`xPx + widthPx` 可能越过图像右边界，应把超出部分循环到图像左侧理解。

如果你已经知道目标像素坐标 `(x, y)`，可以直接读取 instance 图在该点的 RGB 并解码。下面示例查询首帧坐标 `(1200, 480)`：

```powershell
@'
from pathlib import Path
import numpy as np
from PIL import Image

path = Path(r"data\airsim360\nyc_sample\instance\panorama_0.png")
x, y = 1200, 480
pixel = np.asarray(Image.open(path))[y, x]
instance_id = int(pixel[0]) | (int(pixel[1]) << 8) | (int(pixel[2]) << 16)
print({"x": x, "y": y, "rgb": pixel[:3].tolist(), "instanceId": instance_id})
'@ | & ".venv\Scripts\python.exe" -
```

坐标必须落在物体内部，不要选轮廓边缘。背景通常解码为 `0`，项目的实例清单会忽略它。

选择训练目标时，优先选择首帧 `pixels` 较多、bbox 尺寸合理、没有被严重遮挡的实例。只有 1 到几十个像素的实例通常不适合作为首帧模板。

## 4. 在跟踪中使用 instance ID

把查到的整数传给 `--target-instance`：

```powershell
& ".venv\Scripts\python.exe" "tools\run_airsim360_dataset.py" `
  --dataset-root "data\airsim360\nyc_sample" `
  --config "configs\RGBonly.yaml" `
  --target-instance 14211313 `
  --max-frames 1
```

该 ID 用于从首帧 instance mask 生成初始化框；后续帧由 tracker 持续跟踪，不会在每一帧重新按 instance mask 选择目标。传入的 ID 必须在首帧中可见，否则无法建立初始框。

## 5. 在训练数据集中使用 instance ID

Python 数据集接口：

```python
from instatarget.training import AirSim360TrainingDataset

dataset = AirSim360TrainingDataset(
    "data/airsim360/nyc_sample",
    targetInstanceId=14211313,
)

for sample in dataset:
    rgb = sample.rgb
    box = sample.targetBox
    visible = sample.visible
```

`targetInstanceId` 指定整条序列关注的实例。每个 `TrainingSample` 提供 RGB、当前帧目标框和可见性；实例在某帧被遮挡或离开画面时，`visible` 为 `False`。

## 6. 多序列数据集

当 `--dataset-root` 指向多个序列的父目录时，用 `--sequence` 指定序列：

```powershell
& ".venv\Scripts\python.exe" "tools\run_airsim360_dataset.py" `
  --dataset-root "data\airsim360" `
  --sequence "nyc_sample" `
  --config "configs\RGBonly.yaml" `
  --list-instances
```

instance ID 的稳定范围由数据集生成方式决定，不要假设不同序列中的同一个整数一定代表同一个真实物体。应对每条序列分别列出并确认目标 ID。
# Instance ID Commands

安装项目后，可以用短命令读取 AirSim360 第一帧出现的实例 ID：

```powershell
getInstanceID /data/airsim360/nyc_sample /artifacts/airsim360/nyc_sample/InstanceID.txt
```

以 `/data/...` 和 `/artifacts/...` 开头的路径会相对仓库根目录解析。也可以传入普通相对路径或 Windows 绝对路径。

输出按第一帧的语义类别分组，每行格式为：

```text
<semantic-name> <class-ordinal> <instance-id>
```

例如：

```text
concreteblock 1 2497023
```

可选参数：

- `--sequence NAME`：数据根目录下存在多个序列时指定一个序列。
