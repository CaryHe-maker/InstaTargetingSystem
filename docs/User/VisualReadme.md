# PowerShell 可视化使用说明

以下命令默认从仓库根目录 `D:\19810\Documents\GitHub\InstaTargetingSystem` 执行，并使用项目虚拟环境：

```powershell
Set-Location "D:\19810\Documents\GitHub\InstaTargetingSystem"
```

## 注意：instanceID说明见InstanceReadMe.md

## 1. 推荐：运行数据集并生成全部结果

`run_airsim360_dataset.py` 会同时运行跟踪、计算 IoU、生成最终框图片和中间过程图片：

```powershell
& ".venv\Scripts\python.exe" "tools\run_airsim360_dataset.py" `
  --dataset-root "data\airsim360\nyc_sample" `
  --config "configs\RGBonly.yaml" `
  --target-instance 15849574
```

常用调试方式是先只跑一帧，并固定输出目录：

```powershell
& ".venv\Scripts\python.exe" "tools\run_airsim360_dataset.py" `
  --dataset-root "data\airsim360\nyc_sample" `
  --config "configs\RGBonly.yaml" `
  --target-instance 14211313 `
  --max-frames 1 `
  --output-dir "artifacts\visual_smoke"
```

参数如下：

| 参数 | 是否必需 | 说明 |
|---|---:|---|
| `--dataset-root <目录>` | 是 | 单个 AirSim360 序列目录，或包含多个序列的父目录 |
| `--config <YAML>` | 是 | 运行配置；RGB 使用 `configs\RGBonly.yaml`，RGB-D 使用 `configs\RGBD.yaml` |
| `--target-instance <整数>` | 跟踪时是 | 首帧目标的 instance ID；查找方法见 `InstanceReadme.md` |
| `--sequence <名称>` | 否 | 当 `--dataset-root` 是多序列父目录时选择一个子目录 |
| `--max-frames <整数>` | 否 | 只读取前 N 帧，适合快速检查；不传则运行完整序列 |
| `--output-dir <目录>` | 否 | 固定本次输出目录；不传则自动使用 `artifacts\airsim360\...\output_N` |
| `--list-instances` | 否 | 只列出首帧实例并退出，不运行跟踪；此时无需传 `--target-instance` |
| `-h` / `--help` | 否 | 显示当前版本的完整参数帮助 |

一次成功运行的主要输出为：

```text
<output-dir>/
  result/
    tracking.txt
    iou.json
    manifest.json
    visualResult/frame_000000.png
  midVisual/<sequence>/frame_000000/
    local_rgb/
    depth_rgb/
    backend_box/
    geometry_box/
```

- `result\visualResult`：每帧最终提交框，使用荧光绿色绘制。
- `midVisual`：局部视图、深度 RGB、backend 候选框和 ERP 回投影框。
- `result\iou.json`：逐帧 IoU 和汇总指标。
- `result\manifest.json`：输入、目标、输出路径和运行退出码。跟踪失败时不会再误报缺少 `tracking.txt`，应先看控制台中的原始错误和这里的 `trackingExitCode`。

## 2. 直接调用底层跟踪器

需要自行指定每一种输出路径时，可以直接调用模块：

```powershell
& ".venv\Scripts\python.exe" -m instatarget.track_airsim360 `
  --dataset-root "data\airsim360\nyc_sample" `
  --target-instance 14211313 `
  --config "configs\RGBonly.yaml" `
  --output "artifacts\direct_run\tracking.txt" `
  --mid-visual-root "artifacts\direct_run\midVisual" `
  --result-visual-root "artifacts\direct_run\visualResult"
```

底层命令参数：

| 参数 | 是否必需 | 说明 |
|---|---:|---|
| `--dataset-root` | 是 | 数据集或序列目录 |
| `--sequence` | 否 | 从父目录选择序列 |
| `--target-instance` | 是 | 目标 instance ID |
| `--config` | 是 | YAML 配置文件 |
| `--output` | 是 | OTB 格式 `tracking.txt` 的路径 |
| `--mid-visual-root` | 否 | 开启中间可视化并指定根目录；别名是 `--visualization-root` |
| `--result-visual-root` | 否 | 最终逐帧绿色框图片目录 |
| `--max-frames` | 否 | 最多处理 N 帧 |

不传 `--mid-visual-root` 时，中间可视化由 YAML 的 `visualization` 段决定。不传 `--result-visual-root` 时不生成最终绿色框图片。

## 3. YAML 中间可视化选项

```yaml
visualization:
  enabled: true
  outputRoot: ../outputs/visualization
  stages:
    - local_rgb
    - depth_rgb
    - backend_box
    - geometry_box
```

`outputRoot` 的相对路径以 YAML 文件所在目录为基准。`stages` 可选值为：

| stage | 内容 |
|---|---|
| `local_rgb` | backend 实际送入 HiT 的局部图；RGBD 为边缘增强 RGB |
| `depth_rgb` | 黑底白线的深度边缘预测图；RGB-only 不产生有效深度图 |
| `backend_box` | 在同一张 HiT 输入图上的候选框及 `fuseScore` |
| `geometry_box` | 候选框回投影到 ERP 全景图后的结果及 `fuseScore` |

只保留需要检查的 stage 可减少 PNG 编码时间和磁盘占用。例如只检查回投影：

```yaml
visualization:
  enabled: true
  outputRoot: ../outputs/geometry_only
  stages:
    - geometry_box
```

注意：`run_airsim360_dataset.py` 总会传入 `--mid-visual-root`，因此它会开启上述四个中间阶段，并把输出放在本次运行目录中；要精确选择 stage，请直接调用底层跟踪器并只使用 YAML 配置。

## 4. 只预览一张深度图

```powershell
& ".venv\Scripts\python.exe" "tools\preview_airsim360_depth.py" `
  "data\airsim360\nyc_sample\depth\Depth_0.h5" `
  --output "artifacts\depth_preview\depth_0.png"
```

`input_h5` 是必需的位置参数；`--output` 可省略，默认写入 `artifacts\depth_preview\depth_color.png`。命令还会在同目录生成 `_contrast.png` 和 `_edges.png` 两个诊断图。

## 5. PowerShell 注意事项

- PowerShell 的续行符是反引号 `` ` ``，它必须是该行最后一个字符，后面不能有空格。
- 路径建议使用双引号包围，并通过调用运算符 `&` 启动虚拟环境中的 Python。
- 要查看某条命令的实际参数，以 `--help` 输出为准，例如：

```powershell
& ".venv\Scripts\python.exe" "tools\run_airsim360_dataset.py" --help
```
# Short Tracking Commands

安装项目后，可用统一的 `run` 命令运行完整 AirSim360 跟踪：

```powershell
run -RGB_only /data/airsim360/nyc_sample /artifacts/airsim360/nyc_sample/test_rgb 2497023
run -RGBD /data/airsim360/nyc_sample /artifacts/airsim360/nyc_sample/test_rgbd 2497023
```

位置参数依次为数据目录、输出目录和 instance ID。默认配置分别为 `configs/RGBonly.yaml` 与 `configs/RGBD.yaml`。

可选参数：

- `--config PATH`：覆盖模式默认配置。
- `--sequence NAME`：选择数据根目录中的序列。
- `--max-frames N`：只运行前 N 帧。
- `--no-mid-visual`：不生成中间可视化。
- `--no-result-visual`：不生成最终 ERP 框图。

默认输出结构：

```text
<output>/
  result/tracking.txt
  result/visualResult/frame_*.png
  midVisual/<sequence>/frame_*/
```

RGB-only 的 `local_rgb` 和 `backend_box` 使用原始局部 RGB。RGBD 的这两类图使用真正送入 HiT 的深度边缘增强 RGB；非边缘像素保持原图。`depth_rgb` 是黑底白线的深度边缘预测图。
