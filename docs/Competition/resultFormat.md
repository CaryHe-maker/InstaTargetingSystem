# BFoV 提交格式

## 每行字段

结果每帧一行：

```text
clon,clat,fov_h,fov_v
```

四项单位都是度，固定三位小数。内部 TrackResult 使用弧度，`formatBfov()` 在输出边界统一转换，避免算法内部混用单位。

## 第 0 帧

第 0 行必须是 init.txt 给定 BFoV，而不是重新运行模型得到的框。`BfovResultSink` 接收 initialBfov 并在第一次 write 时保证这一约定。

## 无效帧

当 TrackResult.valid=false 时，官方输出写 `0.000,0.000,0.000,0.000`。Controller 内部仍可能持有运动预测 bbox/BFoV，但比赛格式用四零明确表示该帧没有可靠测量。

## 原子与帧数

比赛 sink 先写临时文件，finalize 时检查行数，再原子替换 `<sequence>.txt`。序列中不能跳过帧，也不能因 LOST 而提前结束文件。

## 与本地格式区别

本地通用 sink 可能输出 ERP bbox 或附加状态字段；Competition adapter 是格式边界。修改比赛格式不应改变 TrackResult 或 Controller。

