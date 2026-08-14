# 实例 ID 规范

AirSim360 的实例掩码是 `int32` 二维数组，背景值为 `0`，其他正整数表示可选目标。RGBA 实例图将 RGB 三通道按 `R | (G << 8) | (B << 16)` 组合为整数 ID；标量图像和 NumPy 文件保留原值。

## `InstanceID.txt`

`tools/generate_instance_ids.py` 只读取第 0 帧，忽略背景，并按照语义类别分组写入文本：

```text
<semantic name> <class-local index> <instance ID>
```

同一语义类别内的序号按该帧像素数量从大到小排列。文件头包含数据集、序列和帧索引信息时，仍以每行最后一列的整数作为传入 tracker 的 `--target-instance`。

## 目标框

`PseudoTrackBuilder.buildInitialBox()` 依据实例掩码像素生成初始 ERP 框，水平坐标使用最小循环区间，因此可表达跨越 0/宽度边界的目标。第 1 帧起的伪真值使用同一实例 ID；实例不可见时，IoU 记录为 `0`，该帧不进入可见帧汇总。

## 命令

```powershell
& ".venv\Scripts\python.exe" "tools/generate_instance_ids.py" --dataset-root "data\airsim360\nyc_sample" --output "artifacts\easy_user\nyc_sample\InstanceID.txt"
```

完整选择和跟踪流程见 [User/InstanceReadme.md](User/InstanceReadme.md)。
