# 实例选择与初始化

项目的 AirSim360 跟踪入口要求显式提供一个整数 `target instance ID`。该 ID 来自第 0 帧实例掩码，不是语义类别编号，也不是文件名序号。

推荐先执行：

```powershell
& ".venv\Scripts\python.exe" "tools\generate_instance_ids.py" --dataset-root "data\airsim360\nyc_sample" --output "artifacts\easy_user\nyc_sample\InstanceID.txt"
```

选择目标后，将实例 ID 传给 `tools/run_airsim360_dataset.py` 的 `--target-instance`。初始化框由第 0 帧掩码计算；跟踪器随后从同一视频的第 0 帧开始建立模板，保证初始化和推理使用相同的帧序。

实例图缺失、目标 ID 不存在、掩码与 RGB 尺寸不一致或目标面积为空时，入口返回数据解码或协议错误，不生成可被误读的结果文件。
