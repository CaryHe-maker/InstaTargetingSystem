# InstaTargetingSystem 参考来源

> 本文档汇总本项目实际借鉴或直接参考的 GitHub 项目与技术路线。  
> 统一格式：项目名 / 借鉴技术 / 技术简要介绍 / GitHub 地址。

---

1. 项目名：360VOT

   借鉴技术：BFoV、rBFoV、球面坐标、局部透视裁剪、局部框回投影、全景评测。

   技术简要介绍：全景跟踪 benchmark 与 toolkit，提供 ERP 到局部视图的裁剪、回投影和球面评测方法，是本项目几何层的主参考。

   GitHub 仓库地址：[https://github.com/HuajianUP/360VOT](https://github.com/HuajianUP/360VOT)

2. 项目名：HiT / DyHiT

   借鉴技术：轻量层级 Vision Transformer 跟踪、bridge module、prediction head、动态速度切换。

   技术简要介绍：主力局部跟踪器参考。项目保留 HiT 作为 RGB 主干，DyHiT 作为速度/精度动态切换参考，不把整张 ERP 直接送入主干。

   GitHub 仓库地址：[https://github.com/kangben258/HiT](https://github.com/kangben258/HiT)

3. 项目名：LightTrack

   借鉴技术：轻量单目标跟踪、快速基线、简洁训练/测试流程。

   技术简要介绍：轻量 tracker 参考，用于建立低成本 baseline 和速度对照，不作为主线模型。

   GitHub 仓库地址：[https://github.com/researchmm/LightTrack](https://github.com/researchmm/LightTrack)

4. 项目名：DepthTrack / DeT

   借鉴技术：RGB-D 跟踪、深度边界判断、遮挡判断、恢复搜索、深度摘要融合。

   技术简要介绍：RGB-D 跟踪的核心参考。它说明深度不是旁路，而是可直接参与候选筛选、遮挡判断和恢复的有效信号。

   GitHub 仓库地址：[https://github.com/xiaozai/DeT](https://github.com/xiaozai/DeT)

5. 项目名：SMART-TRACK

   借鉴技术：预测-测量分离、Kalman Filter、深度到 3D 测量、ROI 反向搜索。

   技术简要介绍：运动预测与恢复逻辑参考。它适合映射到 DTC 的多帧预测、深度门控和找回窗口生成。

   GitHub 仓库地址：[https://github.com/mzahana/SMART-TRACK](https://github.com/mzahana/SMART-TRACK)

6. 项目名：RGB-D 跟踪路线索引

   借鉴技术：RGB 与 Depth 并行传递、特征级/分数级融合、模态缺失退化。

   技术简要介绍：多模态对象跟踪路线索引，提供 RGB-D 跟踪的代表性项目集合，用来确认本项目的融合方向和退化策略。

   GitHub 仓库地址：[https://github.com/983632847/Awesome-Multimodal-Object-Tracking](https://github.com/983632847/Awesome-Multimodal-Object-Tracking)

7. 项目名：Depth-Anything-V2

   借鉴技术：单目深度预训练初始化、轻量深度编码器、深度特征 warm start。

   技术简要介绍：本项目深度分支的初始化来源。用于把深度图先映射到可训练的深度编码空间，再接入融合头。

   GitHub 仓库地址：[https://github.com/DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)

8. 项目名：space-stream

   借鉴技术：深度 codec、`Linear` / `UniformHue` / `InverseHue` 编码、深度流伪彩色表达。

   技术简要介绍：用于参考“把深度转成更容易被视觉主干消费的颜色图”的做法。它更接近本项目采用的浮雕式深度颜色化路线，而不是单纯给深度图染色。

   GitHub 仓库地址：[https://github.com/cansik/space-stream](https://github.com/cansik/space-stream)

9. 项目名：Intel RealSense / librealsense examples

   借鉴技术：`align-depth2color`、深度与彩色对齐、深度伪彩色显示。

   技术简要介绍：用于参考深度与彩色对齐、深度图可视化和伪彩色转换的基础做法，适合作为深度转颜色前处理的工程参照。

   GitHub 仓库地址：[https://github.com/IntelRealSense/librealsense](https://github.com/IntelRealSense/librealsense)

---

## 说明

- 本文档只记录实际用到或明确采用的技术来源。
- 若后续新增 `ZoeDepth`、`SUTrack` 或其他外部技术，应在本文件继续追加。
