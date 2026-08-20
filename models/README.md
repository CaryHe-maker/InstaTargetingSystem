# Stage 3 模型产物

生产运行只使用以下配对产物：

- `hit_small_stage3.pth`：Stage 3 checkpoint，要求非空 `model` state。
- `hit_small_stage3.calibration.json`：外观校准、SingleScore 权重与 Controller 工作点。

checkpoint SHA-256 为 `23f7e6e5981eb29e2f4bc8027f2728a4600438efc7a61daefdc8587b492db73c`。校准文件记录同一哈希与 `E:\NewDownload\train\manifest.jsonl` 的 SHA-256；Runtime 默认逐字节核对 checkpoint，并拒绝阈值与 `configs/RGBonly.yaml` 不一致的产物。

维护者替换 Stage 3 原始权重时运行 `docker/compact_checkpoint.py`，生成 `hit_small_stage3_inference.pth` 与 `hit_small_stage3_inference.calibration.json`。脚本只保留同一个 `model` state 并逐张量验证，再把容器校准副本绑定到压缩文件的新 SHA-256。这两个压缩产物纳入 Git，保证远端服务器从 GitHub clone 后可以直接构建；原始训练 checkpoint、旧 `net` checkpoint和本机 engine 继续忽略。
