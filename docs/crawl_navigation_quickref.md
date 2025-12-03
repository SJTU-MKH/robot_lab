# Crawl Navigation System - Quick Reference

## 系统架构

### 1. 核心组件
- **CrawlNavigationEvent** - 导航事件管理器
- **crawl_observations.py** - 导航观察函数  
- **crawl_rewards.py** - 导航奖励函数
- **crawl_events.py** - 导航事件函数

### 2. 添加的观察 (observations)
```python
- target_direction: (2,) - 到目标的归一化xy方向
- target_distance: (1,) - 到目标的距离
- target_yaw_relative: (1,) - 相对于机器人朝向的目标角度
```

### 3. 添加的奖励 (rewards)
```python
- heading_to_target (权重 1.0) - 朝向目标奖励
- progress_to_target (权重 0.5) - 接近目标奖励  
- reached_target (权重 10.0) - 到达目标大奖励
- velocity_to_target (权重 0.8) - 朝目标移动奖励
- next_goal_awareness (权重 0.3) - 下一目标感知奖励
```

### 4. 添加的事件 (events)
```python
- initialize_navigation (startup) - 初始化导航系统
- update_navigation (interval, 每step) - 更新导航目标
- reset_navigation (reset) - 重置导航状态
```

## 导航点布局 (5个点)

```
起点 (0) ──→ 隧道1 (1) ──→ 隧道2 (2) ──→ 隧道3 (3) ──→ 终点 (4)
  |              |              |              |              |
靠近前墙      第一分隔墙     第二分隔墙       可能不存在      靠近后墙
中心线上      隧道中心       隧道中心       隧道中心        中心线上
```

## 工作流程

1. **Startup**: `initialize_navigation_system()`
   - 创建CrawlNavigationEvent实例
   - 加载地形导航点
   - 初始化目标索引为0（起点）

2. **Every Step**: `update_navigation_targets()`
   - 检查是否到达当前目标（距离 < 0.5m）
   - 切换到下一个目标（延迟0.5秒）
   - 更新方向和距离

3. **On Reset**: `reset_navigation_on_termination()`
   - 重置目标索引到0
   - 更新地形对应的导航点

## 训练命令

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task RobotLab-Isaac-Velocity-Crawl-Unitree-Go2-v0 \
    --num_envs 4096 \
    --max_iterations 3000
```

## 关键参数

- **next_goal_threshold**: 0.5米 - 到达目标的距离阈值
- **reach_goal_delay**: 0.5秒 - 到达后等待时间再切换
- **num_goals**: 5 - 固定的导航点数量

## 观察空间变化

添加导航后，观察维度增加了：
- 原始观察: ~48维
- 新增导航: 4维 (direction:2 + distance:1 + yaw_rel:1)
- 总计: ~52维

## 与Parkour的区别

| 特性 | Parkour | Crawl |
|------|---------|-------|
| 导航点数量 | 可变 | 固定5个 |
| 移动方式 | 跳跃/跑步 | 爬行 |
| 地形类型 | 多种障碍 | 三隧道房间 |
| 姿态要求 | 直立 | 低姿态 |
| 速度范围 | 较快 | 较慢 |

## 调试技巧

1. 检查导航点加载:
```python
env.navigation_event.terrain_goals  # 所有地形的导航点
env.navigation_event.env_goals      # 每个环境的导航点
env.navigation_event.cur_goal_idx   # 当前目标索引
```

2. 查看当前目标:
```python
env.navigation_event.cur_goals      # 当前目标位置
env.navigation_event.target_distance  # 到目标距离
env.navigation_event.target_yaw     # 目标方向角
```

3. 监控切换:
```python
env.navigation_event.reach_goal_timer  # 到达计时器
```

## 未来改进方向

- [ ] 添加导航点可视化标记
- [ ] 动态调整目标阈值
- [ ] 添加路径规划
- [ ] 考虑障碍物避让
- [ ] 多目标优化策略
