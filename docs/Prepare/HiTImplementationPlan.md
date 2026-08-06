# HiT/DyHiT 全景跟踪实施方案

## 1. 结论

HiT/DyHiT **比 LightTrack 更适合作为本项目的主力跟踪核心**。

原因：

1. HiT 是轻量级层级 Vision Transformer 跟踪器，结构比传统 Siamese 跟踪器更现代。
2. DyHiT 可以通过阈值在速度和精度之间动态切换，适合全景视频中“简单帧快跑、困难帧加强”的策略。
3. 官方提供 PyTorch 训练、测试、视频 demo、ONNX 转换和性能测试脚本，工程落地条件较好。
4. 它仍然是普通平面视频 tracker，不原生解决 360 全景畸变和经线跨越，所以必须接入全景几何层。

结论：**推荐作为第一优先级模型路线。**

## 2. 参考依据

官方仓库说明：

- HiT 的主体包含 lightweight hierarchical vision transformer、bridge module、prediction head。
- DyHiT 通过场景划分策略实现速度和精度平衡。
- DyHiT 可作为插件加速其他 base tracker，并且不需要额外训练。
- 官方支持 PyTorch、ONNX 转换、benchmark 测试和 video demo。
- 官方性能表显示 HiT/DyHiT 在 GPU、CPU、Jetson AGX/NX 上都有较高 FPS。

参考来源：

- https://github.com/kangben258/HiT
- https://arxiv.org/abs/2308.06904
- https://doi.org/10.1007/s11263-025-02500-9

## 3. 适配思路

不要让 HiT/DyHiT 直接处理整张 equirectangular 全景图，而是采用：

1. **全景几何层**  
   把全景图按目标预测方向裁剪成局部透视图。

2. **HiT/DyHiT 跟踪层**  
   在局部透视图中做单目标跟踪。

3. **球面状态层**  
   把局部框转换回 yaw/pitch 或 BFoV 表示，并处理左右边界连续性。

4. **恢复层**  
   在低置信度时触发多候选搜索或更大视场搜索。

## 4. 技术栈

### 4.1 语言

- **Python**：训练、推理、评测、数据处理主语言。
- **CUDA**：GPU 推理和训练加速。
- **Shell**：环境安装、实验启动、Docker 入口。
- **C++（可选）**：后期做 CMake 原生推理程序或端侧部署。

### 4.2 框架与库

- **PyTorch**：HiT/DyHiT 官方训练和推理框架。
- **OpenCV**：视频读写、图像裁剪、可视化、基础几何操作。
- **NumPy**：球面坐标、框转换、角度连续化。
- **ONNX / ONNX Runtime**：模型导出和轻量部署。
- **TensorRT（可选）**：GPU 决赛部署优化。
- **CMake**：组织 C++/CUDA/ONNX Runtime 版本的推理工程。
- **Docker**：比赛提交环境封装。

## 5. 实施步骤

### 阶段 1：普通视频基线

目标：先跑通 HiT/DyHiT 原生流程。

工作项：

1. 拉取官方代码。
2. 安装 Python 环境和依赖。
3. 下载官方权重。
4. 运行 `tracking/video_demo.py` 或 benchmark 测试。
5. 封装统一接口：`init(frame, bbox)`、`track(frame)`。

验收：

- 普通视频可稳定输出目标框。
- 能输出置信度或用于判断跟踪质量的分数。

### 阶段 2：全景局部视图适配

目标：让 HiT/DyHiT 只看低畸变局部图。

工作项：

1. 把初始框中心转换成 yaw/pitch。
2. 根据上一帧目标方向生成 BFoV。
3. 从 equirectangular 图中裁剪透视视图。
4. 把局部预测框映射回全景坐标。
5. 对跨 0/360 经线的情况做角度展开。

验收：

- 目标跨左右边界时，内部状态连续。
- 高纬度区域比直接平面跟踪更稳定。

### 阶段 3：动态速度/精度策略

目标：利用 DyHiT 的动态能力适应不同难度帧。

工作项：

1. 简单帧使用快速路由或较小搜索窗口。
2. 低置信帧使用更高精度路由或更大搜索窗口。
3. 根据目标速度、尺度变化、响应分数调整 DyHiT threshold。
4. 记录不同阈值下的 FPS 和 AUC。

验收：

- 简单视频保持高 FPS。
- 困难片段的漂移减少。

### 阶段 4：丢失恢复

目标：补上全景长时跟踪能力。

工作项：

1. 低置信时暂停模板更新。
2. 在上一球面位置附近做多尺度 BFoV 搜索。
3. 长时间丢失后做全景粗搜索。
4. 找回后重置 tracker 状态。

验收：

- 遮挡、离开局部视图、相似目标干扰后可恢复。

### 阶段 5：部署优化

目标：为比赛封装和端侧性能做准备。

工作项：

1. 导出 HiT ONNX。
2. 测试 ONNX Runtime CPU/GPU 性能。
3. 可选接入 TensorRT。
4. Docker 内固定依赖版本。
5. 保留 PyTorch 版作为调试后端。

## 6. 推荐目录结构

```text
InstaTargetingSystem/
  src/
    geometry/
    tracker/
      hit_backend/
    recovery/
  tools/
    eval/
    export/
  models/
    hit/
  docker/
  docs/
```

## 7. 风险点

1. **官方模型不是全景模型**  
   必须加局部透视裁剪和球面坐标映射。

2. **Transformer 输入尺寸固定**  
   需要稳定管理搜索图尺寸和目标尺度。

3. **DyHiT 阈值影响精度**  
   要做 ablation，不能只凭直觉选 threshold。

4. **ONNX 导出兼容性**  
   需要提前验证动态维度和算子支持。

## 8. 最终建议

本项目建议优先采用：

1. **HiT-Small 或 HiT-Tiny** 做实时基线。
2. **HiT-Base** 做高精度对照。
3. **DyHiT** 做决赛速度/精度动态版本。

如果时间有限，先接 HiT-Small；如果评测更看 FPS，优先试 HiT-Tiny 或 DyHiT。

