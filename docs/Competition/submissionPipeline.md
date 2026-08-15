# 比赛序列运行

## 序列发现

`listSequences()` 优先读取数据根的 `seqlist.txt`，保持官方顺序；没有列表时扫描包含合法视频和 init.txt 的子目录并稳定排序。每个序列独立建立 runtime，避免模板、运动历史和状态跨序列泄漏。

## 初始化

`loadInitialBfov()` 读取 init.txt 前四个字段 `clon,clat,fov_h,fov_v`，由度转换为内部弧度 BFoV。首帧视频尺寸确定后，Geometry 再把 BFoV 转成初始化 ERP bbox。

## 视频线程

`OpenCvVideoSource` 持久保持一个 VideoCapture，顺序读取 BGR 帧并显式转为 RGB FramePacket。frameIndex 连续，时间戳依据视频 FPS 或稳定回退生成。

## 跟踪与输出

`trackOneSequence()` 复用 Runtime 的 runTracking。第 0 帧由比赛 sink 写给定初始 BFoV，后续帧写 Controller 结果。结束时校验结果行数与视频帧数一致。

## RGB-only 约束

`requireRgbOnlyConfig()` 拒绝 depth.enabled=true 的配置。比赛视频没有对齐深度输入，静默启用 RGB-D 会让本地和提交环境行为不一致。

