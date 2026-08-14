# 实例 ID 使用说明

AirSim360 的实例 ID 是实例掩码中的正整数。语义类别 ID 只用于分组显示，不能替代实例 ID。背景值 `0` 不会写入清单。

推荐命令：

```powershell
& ".venv\Scripts\python.exe" "tools\generate_instance_ids.py" --dataset-root "data\airsim360\nyc_sample" --output "artifacts\easy_user\nyc_sample\InstanceID.txt"
```

命令仅读取第 0 帧，并按语义名称、类内序号和实例 ID 输出 `InstanceID.txt`。将选定的第三列整数传给 `--target-instance`，即可由 `PseudoTrackBuilder` 生成初始化框。

目标跨越 ERP 水平接缝时，初始化框使用循环区间表示。目标在第 1 帧起不可见时，评估记录该帧的可见性和零 IoU，但不会用错误框替代真实掩码信息。
