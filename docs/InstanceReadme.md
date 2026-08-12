# Instance ID 清单生成与使用

## 1. 生成第 0 帧的 InstanceID.txt

在仓库根目录执行：

```powershell
& ".venv\Scripts\python.exe" "tools\generate_instance_ids.py" `
  --dataset-root "data\airsim360\nyc_sample"
```

命令只读取按数据文件编号排序后的第一帧，也就是项目内部的 `frameIndex=0`。对于当前示例，默认输出为：

```text
artifacts/airsim360/nyc_sample/InstanceID.txt
```

输出文件的类别顺序沿用数据集中的 `semantic_lists.txt`。清单只包含第 0 帧实际出现的 instance ID，类内按 ID 从小到大排序并从 1 开始编号。不同类别之间保留一个空行：

```text
concreteblock 1 810871
concreteblock 2 2471892

streetprops 1 123456
streetprops 2 789012
```

每行三列依次是：

```text
<semantic 名称> <该类别内的序号> <instance ID>
```

如果同一 instance 的像素在第 0 帧中对应多个 semantic ID，生成器会把它归到像素数最多的 semantic 类别。instance ID `0` 作为背景忽略。

## 2. 可选参数

查看命令帮助：

```powershell
& ".venv\Scripts\python.exe" "tools\generate_instance_ids.py" --help
```

| 参数 | 是否必需 | 说明 |
|---|---:|---|
| `--dataset-root <目录>` | 是 | 单个 AirSim360 序列目录，或包含多个序列的父目录 |
| `--sequence <名称>` | 否 | 当 `--dataset-root` 是父目录时，选择其中一个序列 |
| `--output <文件>` | 否 | 自定义输出文件；不传时写入 `artifacts/<data 相对路径>/InstanceID.txt` |

从多序列父目录选择 `nyc_sample`：

```powershell
& ".venv\Scripts\python.exe" "tools\generate_instance_ids.py" `
  --dataset-root "data\airsim360" `
  --sequence "nyc_sample"
```

自定义输出位置：

```powershell
& ".venv\Scripts\python.exe" "tools\generate_instance_ids.py" `
  --dataset-root "data\airsim360\nyc_sample" `
  --output "artifacts\custom\MyInstanceID.txt"
```

## 3. 在代码中调用

清单逻辑位于 `instatarget.visualization`，接收一个 `frameIndex=0` 的 `FramePacket`：

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

- `collectInstanceIdGroups(frame0)`：读取第 0 帧并返回按 semantic 分组的 ID；传入其他帧会报错。
- `formatInstanceIdDocument(groups)`：生成上述带类别空行的文本。
- `writeInstanceIdDocument(path, groups)`：以 UTF-8 写入文件并返回绝对 `Path`。

## 4. 使用选出的 instance ID

`InstanceID.txt` 中的每个 ID 都在第 0 帧出现，因此可用于当前从第 0 帧开始的初始化。仍应优先选择目标像素较多、轮廓完整且没有严重遮挡的实例；极小实例虽然可见，但不适合作为可靠模板。

运行跟踪：

```powershell
& ".venv\Scripts\python.exe" "tools\run_airsim360_dataset.py" `
  --dataset-root "data\airsim360\nyc_sample" `
  --config "configs\RGBonly.yaml" `
  --target-instance 14211313
```

训练数据集：

```python
from instatarget.training import AirSim360TrainingDataset

dataset = AirSim360TrainingDataset(
    "data/airsim360/nyc_sample",
    targetInstanceId=14211313,
)
```

AirSim360 RGBA instance PNG 的解码规则仍是：

```text
instanceId = R | (G << 8) | (B << 16)
```
