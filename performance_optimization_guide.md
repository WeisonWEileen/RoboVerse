# 数据收集性能优化指南

## 问题分析
当前训练中数据收集时间过长（每次迭代约12秒），主要瓶颈包括：

1. **相机渲染开销**: 128x96分辨率的相机渲染耗时
2. **ResNet特征提取**: 每步都进行ResNet-18特征提取
3. **GPU并行度不足**: 环境数量相对较少
4. **Tensor操作效率**: 观察计算和奖励计算中的非必要操作

## 已实施的优化

### 1. 相机分辨率优化
- **修改文件**: `humanoid_visualrl/cfg/humanoidFixedGazingCfg.py`
- **优化内容**: 将相机分辨率从128x96降低到64x48
- **预期收益**: 渲染时间减少约75%

```python
# 修改前
width=128, height=96

# 修改后  
width=64, height=48
```

### 2. ResNet特征提取优化
- **修改文件**: `humanoid_visualrl/wrapper/reset_18_extractor.py`
- **优化内容**: 
  - 启用ImageNet标准化以提高特征质量
  - 缓存normalization tensors避免重复创建
  - 优化tensor操作

```python
# 添加了高效的ImageNet归一化
if not hasattr(self, '_imagenet_mean'):
    self._imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=rgb_images.device).view(1, 3, 1, 1)
    self._imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=rgb_images.device).view(1, 3, 1, 1)
```

### 3. 训练参数优化
- **修改文件**: `humanoid_visualrl/cfg/humanoidFixedGazingCfg.py`
- **优化内容**:
  - 增加mini_batch数量从4到8，提高GPU利用率
  - 降低decimation从10到4，增加控制频率

### 4. 观察计算优化
- **修改文件**: `humanoid_visualrl/wrapper/fixed_active_vision_wrapper.py`
- **优化内容**:
  - 移除不必要的tensor clone操作
  - 优化历史缓存管理
  - 缓存视觉奖励计算中的坐标网格

### 5. 奖励计算优化
- **修改文件**: `humanoid_visualrl/wrapper/fixed_active_vision_wrapper.py`
- **优化内容**:
  - 缓存坐标网格避免重复计算
  - 优化tensor操作，减少中间变量
  - 使用更高效的距离计算

## 预期性能提升

基于以上优化，预期性能提升如下：

1. **相机渲染**: 75%速度提升（分辨率降低4倍）
2. **特征提取**: 10-15%速度提升（缓存和优化）
3. **整体训练**: 20-30%速度提升（增加batch size和控制频率）
4. **总体**: **预期数据收集时间从12秒降低到6-8秒**

## 进一步优化建议

### 1. 增加环境数量
```bash
# 在训练脚本中增加环境数量
--num_envs 2048  # 从当前的约1024增加到2048
```

### 2. 使用更快的渲染模式
如果质量要求不高，可以考虑：
- 切换到rasterization渲染模式
- 降低渲染质量设置

### 3. 特征提取优化
- 考虑使用更轻量的特征提取器（如MobileNet）
- 实施特征提取的频率控制（不是每步都提取）

### 4. 数据类型优化
- 只在需要时获取instance_id_seg数据
- 考虑使用half precision (FP16) 训练

## 运行优化后的训练

```bash
# 使用优化后的配置运行训练
cd /home/panwei/RoboVerse
python humanoid_visualrl/scripts/train.py \
    --robot g1_static \
    --task fixed_gazing \
    --sim isaacsim \
    --num_envs 2048 \
    --headless \
    --use_fixed_gazing
```

## 监控性能

训练时关注以下指标：
- **Collection time**: 应该从12秒降低到6-8秒
- **GPU utilization**: 应该更高，特别是在数据收集阶段
- **Memory usage**: 确保没有内存泄漏
- **Training stability**: 确保优化不影响训练稳定性

## 故障排除

如果遇到问题：

1. **内存不足**: 减少环境数量或batch size
2. **渲染问题**: 检查相机配置是否正确
3. **特征提取错误**: 确保ResNet模型加载正确
4. **训练不稳定**: 调整学习率或恢复原始参数

## 性能基准测试

建议运行几个iteration后对比：
- 优化前: ~12秒/iteration
- 优化后: 目标6-8秒/iteration
- 提升幅度: 33-50%

记录实际性能数据以验证优化效果。
