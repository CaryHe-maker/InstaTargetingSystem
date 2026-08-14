# 验证记录

## 自动化检查

```powershell
& ".venv\Scripts\python.exe" -m pytest
& ".venv\Scripts\python.exe" -m pip check
git diff --check
```

仓库验证结果为 61 个 pytest 用例通过，依赖一致性检查和差异空白检查通过。全仓库 Ruff 包含与本文档整理无关的既有风格告警；涉及运行时的修改已单独检查。

## AirSim360 示例

RGB-only：

```powershell
& ".venv\Scripts\python.exe" "tools\run_airsim360_dataset.py" --dataset-root "data\airsim360\nyc_sample" --config "configs\RGBonly.yaml" --target-instance 14211313 --output-dir "artifacts\easy_user\nyc_sample\RGB_only"
```

RGB-D：

```powershell
& ".venv\Scripts\python.exe" "tools\run_airsim360_dataset.py" --dataset-root "data\airsim360\nyc_sample" --config "configs\RGBD.yaml" --target-instance 14211313 --output-dir "artifacts\easy_user\nyc_sample\RGBD"
```

两条线路均完成 11/11 帧处理并生成 11 张最终结果图。示例汇总指标如下：

| 线路 | mean IoU | AUC | success@0.5 |
|---|---:|---:|---:|
| RGB-only | 0.10765976999056637 | 0.1075 | 0 |
| RGB-D | 0.13089978897828028 | 0.1225 | 0 |

## 比赛路径

真实 MP4 比赛路径已验证可顺序处理视频帧并写出每帧 BFoV 行；官方配置校验拒绝启用深度的配置。Docker CUDA 路径已验证能够加载 HiT-Small 并完成前向推理。

## 产物

示例产物位于：

```text
artifacts/easy_user/nyc_sample/InstanceID.txt
artifacts/easy_user/nyc_sample/RGB_only/
artifacts/easy_user/nyc_sample/RGBD/
```

`result/manifest.json` 记录数据根目录、目标实例、帧数和输出目录；`result/tracking.txt` 为逐帧像素框；`result/iou.json` 为逐帧和汇总 IoU；`result/time.json` 记录本次项目从入口开始到结果收尾的运行时间；`result/visualResult` 与 `midVisual` 为诊断图像。
