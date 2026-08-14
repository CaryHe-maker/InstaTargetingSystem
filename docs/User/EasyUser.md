# 快速使用

以下命令在仓库根目录的 PowerShell 中执行。命令使用项目虚拟环境、真实 HiT-Small 和现有 `nyc_sample` 数据；输出目录可直接作为实验归档目录。

安装项目命令入口后，也可用以下简写直接执行跟踪或实例扫描：

```powershell
run -RGB_only /data/airsim360/nyc_sample /artifacts/easy_user/nyc_sample/RGB_only 14211313
run -RGBD /data/airsim360/nyc_sample /artifacts/easy_user/nyc_sample/RGBD 14211313
getInstanceID /data/airsim360/nyc_sample /artifacts/easy_user/nyc_sample/InstanceID.txt
```

`run` 写入逐帧跟踪结果和可视化；本页下方的 `tools/run_airsim360_dataset.py` 命令还会生成 IoU 与运行清单。

## RGB-only 一行命令

```powershell
& ".venv\Scripts\python.exe" "tools\run_airsim360_dataset.py" --dataset-root "data\airsim360\nyc_sample" --config "configs\RGBonly.yaml" --target-instance 14211313 --output-dir "artifacts\easy_user\nyc_sample\RGB_only"
```

该命令逐帧运行 RGB-only 路线，将结果写入指定目录的 `result` 子目录，并生成 `tracking.txt`、`iou.json`、`manifest.json`、`visualResult` 和 `midVisual`。

## RGB-D 一行命令

```powershell
& ".venv\Scripts\python.exe" "tools\run_airsim360_dataset.py" --dataset-root "data\airsim360\nyc_sample" --config "configs\RGBD.yaml" --target-instance 14211313 --output-dir "artifacts\easy_user\nyc_sample\RGBD"
```

RGB-D 命令使用 RGB HiT 和深度伪彩色 HiT 两个独立会话。输出结构与 RGB-only 一致，便于逐项比较。

## 第 0 帧实例清单一行命令

```powershell
& ".venv\Scripts\python.exe" "tools\generate_instance_ids.py" --dataset-root "data\airsim360\nyc_sample" --output "artifacts\easy_user\nyc_sample\InstanceID.txt"
```

该命令只读取第 0 帧，不启动 tracker，也不加载 HiT 权重。生成的每行格式为：

```text
<semantic name> <class-local index> <instance ID>
```

## 快速检查

测试读取链路时可在跟踪命令末尾添加 `--max-frames 2`。第 0 帧用于初始化，第 1 帧开始执行模型推理。详细输入格式、结果字段和指标定义见 [RuntimeReadme.md](RuntimeReadme.md) 与 [VisualReadme.md](VisualReadme.md)。
