# 快速使用

以下命令在仓库根目录的 PowerShell 中执行。命令使用项目虚拟环境、真实 HiT-Small 和现有 `nyc_sample` 数据；输出目录可直接作为实验归档目录。

安装项目命令入口后，也可用以下简写直接执行跟踪或实例扫描：

```powershell
run /data/airsim360/nyc_sample /artifacts/easy_user/nyc_sample/RGB_only 14211313
getInstanceID /data/airsim360/nyc_sample /artifacts/easy_user/nyc_sample/InstanceID.txt
```

`run` 写入逐帧跟踪结果和可视化；可追加 `--no-mid-visual` 或 `--no-result-visual` 关闭对应图像产物。本页下方的 `tools/run_airsim360_dataset.py` 命令还会生成 IoU 与运行清单，并始终启用中间和最终可视化。

## 真实训练数据：可视化（指定序列）

以下命令使用 `E:\NewDownload\train\manifest.jsonl` 的 `train` split。把 `--sequence` 后的值替换为 manifest 中的一个或多个 `sequenceId`；输出的 `midVisual` 只包含 `backend_box` 和 `geometry_box`，最终 ERP 图写入 `resultVisual`。

```powershell
& ".venv\Scripts\python.exe" "tools\run_manifest_sequences.py" --manifest "E:\NewDownload\train\manifest.jsonl" --dataset-root "E:\NewDownload\train" --config "configs\RGBonly.yaml" --weights "models\hit_small_stage3.pth" --split train --sequence "train_sim/seq_0001" "train_real/seq_0001" --output-dir "artifacts\easy_user\train_visual_selected" --visualize
```

## 真实训练数据：可视化（全部序列）

```powershell
& ".venv\Scripts\python.exe" "tools\run_manifest_sequences.py" --manifest "E:\NewDownload\train\manifest.jsonl" --dataset-root "E:\NewDownload\train" --config "configs\RGBonly.yaml" --weights "models\hit_small_stage3.pth" --split all --all --allow-holdout --output-dir "artifacts\easy_user\train_visual_all" --visualize
```

## 比赛级测试（指定序列，无可视化）

每个序列会写入 `evaluations`，聚合结果写入 artifact 根目录的 `aggregate.json` 和 `result.txt`。`result.txt` 包含最终循环 ERP IoU、AUC、Success Rate@0.5、tracking loss rate、丢失帧数以及每帧运行时间均值/P50/P95/P99。

```powershell
& ".venv\Scripts\python.exe" "tools\run_manifest_sequences.py" --manifest "E:\NewDownload\train\manifest.jsonl" --dataset-root "E:\NewDownload\train" --config "configs\RGBonly.yaml" --weights "models\hit_small_stage3.pth" --split train --sequence "train_sim/seq_0001" "train_real/seq_0001" --output-dir "artifacts\easy_user\train_competition_selected"
```

## 比赛级测试（全部序列，无可视化）

```powershell
& ".venv\Scripts\python.exe" "tools\run_manifest_sequences.py" --manifest "E:\NewDownload\train\manifest.jsonl" --dataset-root "E:\NewDownload\train" --config "configs\RGBonly.yaml" --weights "models\hit_small_stage3.pth" --split all --all --allow-holdout --output-dir "artifacts\easy_user\train_competition_all"
```

`--split` 默认为 `train`；指定 `--split all --all --allow-holdout` 会覆盖 manifest 中的全部 130 个序列（包括 holdout），必须在模型冻结后执行。测试命令不传 `--visualize`，因此不会生成 `midVisual` 或最终可视化图。

## RGB-only 一行命令

```powershell
& ".venv\Scripts\python.exe" "tools\run_airsim360_dataset.py" --dataset-root "data\airsim360\nyc_sample" --config "configs\RGBonly.yaml" --target-instance 14211313 --output-dir "artifacts\easy_user\nyc_sample\RGB_only"
```

该命令逐帧运行 RGB-only 路线，将结果写入指定目录的 `result` 子目录，并生成 `tracking.txt`、`iou.json`、`manifest.json`、`time.json`、`visualResult` 和 `midVisual`。

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
