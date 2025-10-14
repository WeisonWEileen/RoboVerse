# SeeFlagEvaluator 更新说明

## 新增功能：Episode Reward 追踪

evaluator现在可以追踪和可视化每个reward组件的平均episode reward。

### 新增方法

#### `update_episode_reward(episode_sums)`

在每个episode结束时调用，记录各个reward组件的值。

**参数：**
- `episode_sums`: `dict[str, torch.Tensor]` - reward名称到值的字典

**示例：**
```python
# 在episode结束时（reset_buf > 0）
if env_wrapper.reset_buf[0] > 0:
    evaluator.update_episode_reward(env_wrapper.episode_sums)
```

`episode_sums` 格式示例：
```python
{
    'see_object': tensor([0.2200], device='cuda:0'),
    'fuse_wrist_close_to_object_and_grasp': tensor([0.0405], device='cuda:0'),
    'tracking_lin_vel': tensor([0.1500], device='cuda:0'),
    # ... 其他reward组件
}
```

#### `get_episode_reward_averages()`

获取所有reward组件的平均值。

**返回：**
- `dict[str, float]` - reward名称到平均值的字典

### 可视化变化

现在生成的图表包含**3个子图**（原来是2个）：

1. **子图1：See Flag Over Time** （保持不变）
   - 显示see_flag随时间的变化
   - 包含滚动平均和总体平均

2. **子图2：Per-Environment Statistics** （保持不变）
   - 单环境：显示统计信息（增加了episode数量）
   - 多环境：显示各环境的see_flag柱状图

3. **子图3：Average Episode Rewards** （新增）✨
   - 柱状图显示各reward组件的平均episode reward
   - 正值用绿色，负值用红色
   - 按reward值从高到低排序
   - 每个柱子上方显示具体数值
   - 标题显示episode数量

### 数据保存

#### NPZ文件

`see_flag_data.npz` 现在包含：
- `see_flag_history`: 所有步骤的see_flag
- `env_averages`, `env_counts`, `env_steps`: 环境统计
- `overall_average`: 总体平均see_flag
- `total_steps`: 总步数
- `episode_count`: episode总数 ✨
- `episode_rewards_{reward_name}`: 每个reward组件的所有episode值 ✨

**读取示例：**
```python
import numpy as np

data = np.load('evaluation/see_flag_data.npz')
episode_count = data['episode_count']
see_object_rewards = data['episode_rewards_see_object']
grasp_rewards = data['episode_rewards_fuse_wrist_close_to_object_and_grasp']

print(f"Total episodes: {episode_count}")
print(f"See object rewards: mean={see_object_rewards.mean():.4f}, std={see_object_rewards.std():.4f}")
```

#### TXT文件

`episode_reward_averages.txt` - 可读的reward平均值总结 ✨

**示例内容：**
```
Episode Reward Averages (100 episodes):
============================================================
see_object                              :     0.2200
fuse_wrist_close_to_object_and_grasp    :     0.0405
tracking_lin_vel                        :     0.1500
...
```

### 输出总结

`print_summary()` 现在包含：
- Total Episodes数量 ✨
- 各reward组件的平均值（按值从高到低排序）✨

**示例输出：**
```
============================================================
See Flag Evaluation Summary
============================================================
Total Steps: 10000
Total Episodes: 133
Overall Average See Flag: 0.7234
Rolling Average See Flag (last 100 steps): 0.7450
Total Object Visible Count: 7234/10000

Average Episode Rewards (133 episodes):
  see_object                              :     0.2200
  fuse_wrist_close_to_object_and_grasp    :     0.0405
  tracking_lin_vel                        :     0.1500
============================================================
```

### 使用示例

在 `play.py` 中已集成：

```python
# 初始化evaluator（自动处理）
evaluator = SeeFlagEvaluator(...)

# 主循环
for i in range(10000):
    # ... 执行动作 ...
    
    # Episode结束时更新reward统计
    if env_wrapper.reset_buf[0] > 0:
        evaluator.update_episode_reward(env_wrapper.episode_sums)
    
    # 每步更新see_flag（原有功能）
    evaluator.update(env_wrapper.see_flag)

# 关闭并保存
evaluator.close()
```

### 配置参数

图表尺寸已调整为 `(12, 12)` 以容纳3个子图。

### 注意事项

1. **柱状图颜色**：正值（绿色）表示奖励，负值（红色）表示惩罚
2. **排序**：reward按平均值从高到低排序，便于快速识别主要reward来源
3. **标签旋转**：x轴标签旋转45度，避免重叠
4. **自动缩放**：y轴自动适应reward值范围
5. **零线**：图中有一条黑色水平线标记零值

### 性能影响

- Episode reward追踪几乎无性能开销（仅在episode结束时）
- 绘图第3个子图会略微增加渲染时间
- 如果reward组件很多（>10个），建议增大 `plot_interval`

