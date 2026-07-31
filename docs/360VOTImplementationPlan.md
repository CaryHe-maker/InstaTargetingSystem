# 360VOT 全景跟踪实施方案

## 1. 结论

360VOT **不是一个可直接替代 LightTrack 或 HiT 的跟踪模型**，而是一个更贴近本项目需求的全景跟踪数据集、工具链、表示方法和评测框架。

它对本项目非常重要，因为它专门处理：

1. 等距柱状投影全景图。
2. 目标跨越左右边界。
3. 大畸变区域。
4. 球面上的目标表示和评测。
5. BFoV、rBFoV 等更适合全景图的标注格式。

结论：**360VOT 应作为全景几何、数据处理和评测参考框架，而不是 tracker 核心。**

## 2. 参考依据

官方仓库说明：

- 360VOT 是 ICCV 2023 的 omnidirectional visual object tracking benchmark。
- 数据集包含 120 个序列，最高约 113K 高分辨率 equirectangular 帧。
- 覆盖 crossing border、large distortion、stitching artifact 等全景跟踪挑战。
- 提供 BBox、rBBox、BFoV、rBFoV 四类 ground truth 表示。
- 提供球面角度精度、dual success rate 等全景评测指标。
- 工具库提供 `crop_bfov`、`plot_bfov`、`crop_bbox`、`localBbox2Bfov`、`localBbox2Bbox` 等全景图处理函数。

参考来源：

- https://github.com/HuajianUP/360VOT
- https://360vot.hkustvgd.com/
- https://arxiv.org/abs/2308.15504

## 3. 适配思路

本项目应把 360VOT 当成“全景适配层”的参考实现。

建议结构：

1. **360VOT 几何层**  
   使用或参考其 BFoV 裁剪、球面坐标和局部框回投影逻辑。

2. **外部 tracker 层**  
   接入 HiT、LightTrack、NanoTrack 或 MixFormerV2。

3. **结果表示层**  
   内部优先使用 BFoV/rBFoV，最终按比赛要求输出普通 bbox。

4. **评测层**  
   使用 360VOT 的指标和可视化方法验证全景表现。

## 4. 技术栈

### 4.1 语言

- **Python**：360VOT 工具链主语言，适合快速集成和实验。
- **Python + PyTorch**：如果使用其 PyTorch 相关跟踪框架。
- **C++（可选）**：后期将几何计算和推理封装成 CMake 工程。
- **Shell**：评测脚本、数据整理、Docker 入口。

### 4.2 框架与库

- **360VOT Toolkit**：全景图裁剪、转换、评测和可视化。
- **OpenCV**：视频处理、图像转换、可视化。
- **NumPy**：球面几何和坐标转换。
- **Rotated_IoU**：rBBox 评测依赖。
- **PyTorch**：外部 tracker 推理和训练。
- **Docker**：复现评测环境。
- **CMake**：后期 C++ 化和部署工程组织。

## 5. 实施步骤

### 阶段 1：跑通 360VOT 工具链

目标：先理解数据格式和评测流程。

工作项：

1. 拉取 360VOT toolkit。
2. 安装 `requirements.txt`。
3. 下载或准备小规模 360VOT 数据。
4. 跑通 `scripts/eval_360VOT.py`。
5. 跑通 `scripts/vis_result_360VOT.py`。

验收：

- 能读取 360VOT 数据。
- 能评测 benchmark 格式结果。
- 能可视化 BFoV 或 bbox。

### 阶段 2：抽取全景几何能力

目标：把 360VOT 的几何能力变成本项目模块。

工作项：

1. 封装 `crop_bfov` 类似能力。
2. 封装局部 bbox 到 BFoV 的转换。
3. 封装局部 bbox 到全景 bbox 的转换。
4. 封装 yaw/pitch 角度连续化。
5. 做单元测试覆盖赤道、高纬度、跨边界场景。

验收：

- 局部裁剪和回投影可逆性足够稳定。
- 跨边界结果不会出现断裂。

### 阶段 3：接入外部 tracker

目标：用 360VOT 几何层驱动 HiT/LightTrack 跟踪。

工作项：

1. 首帧 bbox 转成 BFoV。
2. 每帧从 BFoV 裁剪局部视图。
3. tracker 在局部视图输出 bbox。
4. 将局部 bbox 回投影成 BFoV 和全景 bbox。
5. 输出比赛需要的格式。

验收：

- HiT 或 LightTrack 可以在 360 视频上跑完整序列。

### 阶段 4：全景评测与误差分析

目标：比普通 IoU 更准确地分析全景失败原因。

工作项：

1. 使用 360VOT 属性评测 CB、LD、SA 等场景。
2. 分别统计普通 bbox 指标和 BFoV 球面指标。
3. 可视化失败帧。
4. 根据失败类型调整搜索窗口和恢复策略。

验收：

- 可以说清楚模型在哪些全景属性上失败。

### 阶段 5：比赛接口适配

目标：把 360VOT 风格能力转成比赛提交格式。

工作项：

1. 内部使用 BFoV 表示。
2. 输出时转成比赛要求 bbox。
3. Docker 中只保留必要工具和运行依赖。
4. 数据集和结果大文件不提交 Git。

## 6. 推荐目录结构

```text
InstaTargetingSystem/
  src/
    geometry/
      bfov.py
      projection.py
    tracker/
    recovery/
  tools/
    eval_360/
    visualize_360/
  third_party/
    360VOT/
  docs/
```

## 7. 风险点

1. **360VOT 不是模型**  
   需要配合 HiT、LightTrack 或其他 tracker。

2. **比赛评测可能仍用 bbox IoU**  
   内部 BFoV 有用，但输出必须兼容官方格式。

3. **工具链依赖可能偏研究化**  
   Docker 封装时要精简依赖。

4. **rBBox/Rotated_IoU 增加复杂度**  
   如果比赛只评 bbox，可以先不做 rBBox 主线。

## 8. 最终建议

360VOT 最适合承担：

1. 全景坐标表示参考。
2. BFoV 裁剪和回投影参考。
3. 全景专用评测和可视化参考。
4. 失败场景分析工具。

项目主线建议是：

**360VOT 几何与评测 + HiT/DyHiT 跟踪核心 + 自研丢失恢复。**

