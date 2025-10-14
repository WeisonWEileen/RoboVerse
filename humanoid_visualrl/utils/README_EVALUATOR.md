# See Flag Evaluator

这个evaluator用于实时追踪和可视化机器人在执行任务时的`see_flag`指标（即机器人是否能看到目标物体）。

## 功能特性

1. **实时追踪**: 记录每一步的see_flag状态
2. **滚动平均**: 计算滑动窗口内的平均see_flag
3. **可视化**: 定期生成图表并保存为PNG（不显示窗口）
4. **数据保存**: 自动保存评估数据和最终图表
5. **多环境支持**: 支持追踪多个并行环境的see_flag
6. **无窗口模式**: 使用非交互式后端，适合服务器环境运行

## 使用方法

### 在play.py中使用（已集成）

evaluator已经集成到`humanoid_visualrl/scripts/play.py`中，运行play.py时会自动：
- 创建evaluator实例
- 实时更新see_flag数据
- 显示可视化图表
- 保存结果到模型目录下的`evaluation/`文件夹

```bash
python humanoid_visualrl/scripts/play.py --task <task_name> --num_envs 1
```

### 独立使用

```python
from humanoid_visualrl.utils.evaluator import SeeFlagEvaluator
import torch

# 初始化evaluator
evaluator = SeeFlagEvaluator(
    num_envs=1,              # 环境数量
    window_size=100,         # 滚动平均窗口大小
    save_dir="./results",    # 保存目录
    plot_interval=5,         # 每5步更新一次图表
    enable_plot=True,        # 启用实时绘图
)

# 在训练/评估循环中
for step in range(1000):
    # ... 执行环境步骤 ...
    see_flag = env_wrapper.see_flag  # shape: (num_envs,)
    
    # 更新evaluator
    evaluator.update(see_flag)

# 关闭并保存结果
evaluator.close()
```

## 输出文件

evaluator会在指定的`save_dir`目录下生成以下文件：

1. **see_flag_evaluation.png**: 实时更新的图表（每次更新时覆盖）
2. **see_flag_evaluation_final.png**: 最终的高分辨率图表（300 DPI）
3. **see_flag_data.npz**: 保存的原始数据，包含：
   - `see_flag_history`: 所有步骤的see_flag平均值
   - `env_averages`: 每个环境的平均see_flag
   - `env_counts`: 每个环境看到物体的次数
   - `env_steps`: 每个环境的总步数
   - `overall_average`: 总体平均see_flag
   - `total_steps`: 总步数

## 可视化说明

生成的图表包含两个子图：

### 上图：See Flag Over Time
- **蓝色半透明线**: 每一步的即时see_flag值（0或1的平均值）
- **红色粗线**: 滚动平均值（平滑后的趋势）
- **绿色虚线**: 整体平均值（从开始到当前的总平均）

### 下图：
- **单环境**: 显示统计信息（总步数、平均值等）
- **多环境**: 显示每个环境的see_flag平均值柱状图

## 参数说明

- `num_envs` (int): 环境数量，默认为1
- `window_size` (int): 滚动平均的窗口大小，默认为100
- `save_dir` (str | None): 保存目录路径，如果为None则不保存
- `plot_interval` (int): 更新图表的间隔步数，默认为10
- `enable_plot` (bool): 是否启用实时绘图，默认为True

## 示例输出

```
============================================================
See Flag Evaluation Summary
============================================================
Total Steps: 10000
Overall Average See Flag: 0.7234
Rolling Average See Flag (last 100 steps): 0.7450
Total Object Visible Count: 7234/10000
============================================================
```

## 加载保存的数据

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = np.load('results/see_flag_data.npz')
see_flag_history = data['see_flag_history']
overall_average = data['overall_average']

# 自定义绘图
plt.figure(figsize=(10, 5))
plt.plot(see_flag_history)
plt.axhline(y=overall_average, color='r', linestyle='--')
plt.xlabel('Step')
plt.ylabel('See Flag Rate')
plt.title('See Flag Analysis')
plt.show()
```

## 注意事项

1. **无窗口显示**: 使用非交互式后端（Agg），图表直接保存为PNG文件，不会弹出窗口
2. 绘图会轻微影响性能，如果需要最快速度可以设置`enable_plot=False`
3. 建议`plot_interval`根据总步数调整，步数越多可以设置越大（例如：10000步可以设置为50-100）
4. 每次更新都会覆盖`see_flag_evaluation.png`，最终图表保存为`see_flag_evaluation_final.png`
5. 数据会在`evaluator.close()`时自动保存，确保在程序结束前调用
6. 适合在无显示器的服务器上运行，不需要X11转发

## 扩展使用

evaluator设计为可扩展的，你可以：
- 修改`plot()`方法来自定义可视化
- 添加其他指标的追踪
- 实现自己的统计方法
- 导出数据到其他格式（CSV、JSON等）

