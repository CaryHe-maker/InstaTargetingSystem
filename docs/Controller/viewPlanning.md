# 分状态视域规划

实现位于 `controller/recovery_planner.py`。

## 固定最大视域

所有搜索 ViewSpec 的水平和垂直 FOV 都等于 `geometry.maxFovDeg=120`，不按状态或轮次缩放。`minFovDeg` 只参与运动回退包络，不改变实际搜索视图。

## 四角布局

给定中心 c，Planner 在其局部相机坐标系向左上、右上、左下、右下各偏移水平/垂直 40 度。中心方向通过 forward/right/up 基向量合成后归一化，因此靠近极点时仍保持局部四角语义。

120 度视域配合 80 度中心间距，使水平或垂直相邻视图重合 40 度，即各自宽度的 1/3；四视图共同覆盖中心的 40×40 度区域，即二维面积比例约 1/9。c 是共同重合区中心，不是四张图中的一个额外视图。

## Cubemap

全景轮固定读取 front/right/back/left/up/down 六个方向，每面仍为 120 度。比标准 90 度 cubemap 多出的边缘重合用于减少接缝附近漏检。

## 状态路线

| 状态 | Round 1 | Round 2 | Round 3 |
|---|---|---|---|
| TRACKING | c1 四角 | Round 1 最佳中心四角，最终轮 | 无 |
| UNCERTAIN | c1 四角 | Round 1 最佳中心四角 | 六面全景，最终轮 |
| RECOVERING | c1 四角 | Round 1 最佳中心四角 | 六面全景，最终轮 |
| LOST | 六面全景 | Round 1 最佳中心四角，最终轮 | 无 |

Round 2 的中心取 Round 1 最高置信局部框或融合框的球面中心。若继续到 Round 3，搜索种子取截至 Round 2 的累计最高置信局部框或融合框中心；即使该候选不具备输出资格，它仍可作为概率最大搜索种子。

## 预算

三轮最多 4+4+6=14 张，TRACKING 最多 8 张，LOST 最多 10 张。`maxViewsPerFrameTotal` 是事务总预算；不足以完整容纳下一轮时应直接报协议错误，而不是只生成部分 cubemap。

