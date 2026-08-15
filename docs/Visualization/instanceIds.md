# Instance ID 分组算法

## 收集

`collectInstanceIdGroups()` 遍历第一帧 instance mask 的唯一非零 ID。对每个实例，在对应像素中统计 semantic ID 频率，以占比最高的 semantic ID 作为类别；找不到类别名称时使用稳定的 fallback 名称。

## 排序与编号

分组先按类别名排序，组内按 instance ID 排序，并产生人类可读 ordinal。输出文本让用户从类别和序号定位真正的整数 instance ID。

## 命令路径

`getInstanceID` 由 `app/commands.py` 提供短命令；`tools/generate_instance_ids.py` 提供对应开发工具。它们只读取数据，不建立 HiT runtime。

## 注意事项

instance ID 是一段序列内的对象标识，不能假定跨序列稳定。semantic 多数投票会在严重遮挡或掩码污染时选错类别名，但整数 instance ID 仍保持原值；跟踪初始化实际使用整数 ID。
