# TestHit$Geometry

这个分支只做一条很薄的 smoke pipeline：

1. 读取一组图片文件
2. 用 `geometry` 把首帧初始框投成局部视图
3. 把局部视图交给 `tracker`
4. 把 tracker 的局部输出再投回 ERP
5. 用现有 `visualization` 把中间结果写成 PNG

## 结构

- `DirectoryFrameSource`
  - 读取单张图片或目录中的图片序列
  - 默认按文件名排序
- `SimpleGeometryTrackController`
  - 负责 `ERP -> geometry -> tracker -> ERP`
  - 每帧都会产出一张局部图和一张 ERP 标注图
- `VisualizationRecorder`
  - 记录 `local_rgb`
  - 记录 `backend_box`
  - 记录 `geometry_box`

## 输入

- `input_path`：图片文件或图片目录
- `--initial-box X Y W H`：首帧 ERP 初始框
- `--output-root`：可视化输出目录

## 运行

```bash
python -m instatarget.app.track <input_path> --initial-box X Y W H --output-root artifacts/smoke-track
```

## 输出

每一帧都会在输出目录下生成：

- `local_rgb`
- `backend_box`
- `geometry_box`

这个分支的目标不是做复杂控制，而是确认现有模块能连起来、能读图、能跑通、能看结果。
