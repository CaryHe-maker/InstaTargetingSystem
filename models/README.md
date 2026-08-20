# Stage 3 模型产物

生产运行只使用以下已纳入 Git 的配对产物：

- `hit_small_stage3_inference.pth`：紧凑 Stage 3 checkpoint，只保留非空 `model` state。
- `hit_small_stage3_inference.calibration.json`：外观校准、SingleScore 权重与 Controller 工作点。

生产 checkpoint SHA-256 为 `f9ee8e946f29a813ee359d5e417245651b622863759145abce82690e3fc12c66`。校准文件记录同一哈希与 `E:\NewDownload\train\manifest.jsonl` 的 SHA-256；Runtime 默认逐字节核对 checkpoint，并拒绝阈值与 `configs/RGBonly.yaml` 不一致的产物。

维护者替换 Stage 3 原始权重时，在持有本地 `hit_small_stage3.pth` 与配对校准文件的机器上运行 `docker/compact_checkpoint.py`。脚本逐张量确认紧凑文件保留完全相同的 `model` state，并把校准副本绑定到新 SHA-256。原始训练 checkpoint、旧 `net` checkpoint 和本机 engine 继续忽略；远端构建服务器不得重新生成或替换上述生产配对产物。
