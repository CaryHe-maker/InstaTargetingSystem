# InstanceID.txt 说明

`InstanceID.txt` 是 AirSim360 序列第 0 帧的实例清单。它只包含按 RGB 文件末尾编号排序后第一张图中实际可见的 instance ID，因此其中每个 ID 都满足当前跟踪器的首帧初始化要求。

## 生成命令

在仓库根目录运行：

```powershell
& ".venv\Scripts\python.exe" "tools\generate_instance_ids.py" `
  --dataset-root "data\airsim360\nyc_sample"
```

默认输出：

```text
artifacts/airsim360/nyc_sample/InstanceID.txt
```

从包含多个序列的父目录选择一组数据：

```powershell
& ".venv\Scripts\python.exe" "tools\generate_instance_ids.py" `
  --dataset-root "data\airsim360" `
  --sequence "nyc_sample"
```

自定义输出文件：

```powershell
& ".venv\Scripts\python.exe" "tools\generate_instance_ids.py" `
  --dataset-root "data\airsim360\nyc_sample" `
  --output "artifacts\custom\InstanceID.txt"
```

## 文件格式

每行包含 semantic 名称、该类别内从 1 开始的序号和实际 instance ID：

```text
concreteblock 1 2497023
concreteblock 2 4138724

streetprops 1 123456
```

类别顺序遵循数据集的 `semantic_lists.txt`；类别之间保留一个空行，类内 instance ID 按数值升序排列。背景 ID `0` 不写入文件。

## 用于跟踪

选择第三列的 instance ID：

```powershell
& ".venv\Scripts\python.exe" "tools\run_airsim360_dataset.py" `
  --dataset-root "data\airsim360\nyc_sample" `
  --config "configs\RGBonly.yaml" `
  --target-instance 14211313
```

清单保证 ID 在第 0 帧可见，但不保证目标足够大。建议优先选择面积较大、边界清晰且遮挡较少的目标，避免只有少量像素的实例。

## 程序接口

```python
from instatarget.visualization import (
    collectInstanceIdGroups,
    formatInstanceIdDocument,
    writeInstanceIdDocument,
)

groups = collectInstanceIdGroups(frame0)
text = formatInstanceIdDocument(groups)
path = writeInstanceIdDocument("artifacts/InstanceID.txt", groups)
```

`collectInstanceIdGroups` 要求传入 `frameIndex=0` 的 `FramePacket`，以防调用者误把后续帧中无法用于初始化的 ID 写入清单。

## IoU 汇总约定

`result/iou.json` 的 `frames` 仍保留第 0 帧 IoU，便于检查初始化框；但第 0 帧是给定的初始化结果，不是 tracker 的预测，因此它的 `includedInSummary` 为 `false`。

`summary.meanIoU`、`summary.successRate@0.5` 和 `summary.auc` 只统计第 1 帧及后续可见帧的预测结果。
