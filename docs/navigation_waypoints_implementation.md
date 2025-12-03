# 导航点系统实现文档

## 概述
为爬行环境(Crawl Environment)实现了导航点系统，参考了Parkour训练中的导航机制。

## 实现内容

### 1. 导航点布局
三隧道房间地形包含5个导航点：
- **起点** (索引0): 靠近前墙的中心线位置 `[width/2, wall_thickness + 0.3, 0.15]`
- **隧道1中心** (索引1): 第一个隧道的中心位置
- **隧道2中心** (索引2): 第二个隧道的中心位置  
- **隧道3中心** (索引3): 第三个隧道的中心位置（如果存在第三个隧道）
- **终点** (索引4): 靠近后墙的中心线位置 `[width/2, length - wall_thickness - 0.3, 0.15]`

所有起点和终点都位于房间的中心线上，隧道点则位于各自隧道的中心。

### 2. 新增文件

#### 2.1 CrawlTerrainGenerator (`crawl_terrain_generator.py`)
- 自定义地形生成器类，扩展自`TerrainGenerator`
- 添加了`num_goals`、`goals`属性用于存储导航点
- 实现了`_generate_random_terrains()`和`_generate_curriculum_terrains()`方法
- `_get_terrain_mesh()`方法处理地形生成并提取导航点

#### 2.2 CrawlTerrainImporter (`crawl_terrain_importer.py`)
- 自定义地形导入器类，扩展自`TerrainImporter`
- 使用`CrawlTerrainGenerator`来生成包含导航点的地形
- 通过`terrain_generator_class`属性暴露地形生成器实例

#### 2.3 CrawlTerrainGeneratorCfg
- 地形生成器配置类，扩展自`TerrainGeneratorCfg`
- 添加了`num_goals: int = 5`属性

### 3. 修改文件

#### 3.1 `three_tunnel_chamber_generator()` 函数
- 函数签名更新为: `def three_tunnel_chamber_generator(difficulty, cfg, num_goals=5)`
- 返回值从`(meshes, origin)`改为`(meshes, origin, goals)`
- 在生成隧道时记录隧道中心位置
- 最后生成包含5个导航点的numpy数组

#### 3.2 `crawl_terrain_cfg.py`
- 导入`TerrainGeneratorCfg`
- 添加`CrawlTerrainGeneratorCfg`类

#### 3.3 `crawl_env_cfg.py`
- 导入`TerrainImporterCfg`和`CrawlTerrainImporter`
- 在`__post_init__()`中使用自定义的`CrawlTerrainImporter`
- 设置`class_type=CrawlTerrainImporter`

### 4. 导航点数据结构

```python
# 地形生成器中的导航点存储
self.goals = np.zeros((num_rows, num_cols, num_goals, 3))
# 形状: (地形行数, 地形列数, 导航点数量, xyz坐标)

# 每个地形的导航点数组
goals = np.zeros((num_goals, 3))
goals[0] = [width/2, wall_thickness + 0.3, 0.15]  # 起点
goals[1] = tunnel_centers[0]  # 隧道1中心
goals[2] = tunnel_centers[1]  # 隧道2中心  
goals[3] = tunnel_centers[2] if len(tunnel_centers) > 2 else tunnel_centers[1]  # 隧道3中心
goals[4] = [width/2, length - wall_thickness - 0.3, 0.15]  # 终点
```

### 5. 使用方法

```python
# 在环境中访问地形生成器和导航点
from robot_lab.tasks.manager_based.locomotion.velocity.utils.terrains_asset.mesh_crawl import (
    CrawlTerrainImporter,
)

# 环境配置中
self.scene.terrain = TerrainImporterCfg(
    class_type=CrawlTerrainImporter,
    terrain_type="generator",
    terrain_generator=CRAWL_TERRAINS_CFG,
    ...
)

# 在环境运行时访问导航点
terrain_importer = env.scene.terrain  # CrawlTerrainImporter实例
terrain_generator = terrain_importer.terrain_generator_class  # CrawlTerrainGenerator实例
goals = terrain_generator.goals  # numpy数组 (num_rows, num_cols, num_goals, 3)

# 获取特定地形的导航点
row, col = 0, 0
terrain_goals = goals[row, col]  # (num_goals, 3)
start_point = terrain_goals[0]  # 起点
end_point = terrain_goals[4]  # 终点
```

### 6. 参考Parkour实现
本实现参考了`parkour_terrain_generator.py`和`parkour_terrain_importer.py`的设计模式：
- 自定义TerrainGenerator存储导航点信息
- 自定义TerrainImporter使用自定义Generator
- 通过`class_type`参数在配置中指定自定义Importer
- 导航点数据跟随地形实例一起存储和访问

### 7. 后续工作
- [ ] 在观察空间中添加导航方向信息
- [ ] 实现导航奖励函数（引导机器人朝向下一个导航点）
- [ ] 添加导航点可视化（在仿真中显示导航点位置）
- [ ] 实现目标点切换逻辑（到达当前点后切换到下一个点）

## 测试
创建了测试脚本`test_navigation_waypoints.py`用于验证导航点生成是否正确。

## 文件清单
### 新增:
- `crawl_terrain_generator.py`
- `crawl_terrain_importer.py`  
- `test_navigation_waypoints.py`

### 修改:
- `crawl_terrain.py` (three_tunnel_chamber_generator函数)
- `crawl_terrain_cfg.py` (添加CrawlTerrainGeneratorCfg)
- `crawl_env_cfg.py` (使用CrawlTerrainImporter)
- `__init__.py` (导出新类)
