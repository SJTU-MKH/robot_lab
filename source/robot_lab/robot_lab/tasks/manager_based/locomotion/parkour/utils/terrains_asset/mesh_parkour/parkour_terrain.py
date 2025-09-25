import numpy as np
import trimesh
from collections.abc import Callable

# 导入你的配置类（我们将在下一步创建它）
# from my_terrains_cfg import SteppingStonesTerrainCfg 
# 为了让代码独立运行，我们先用 SubTerrainBaseCfg 占位
from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg 


def stepping_stones_generator(difficulty: float, cfg: SubTerrainBaseCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """
    一个生成汀步石地形的函数。

    - 难度 (difficulty) 会影响石墩的最大高度。
    - 地形参数 (cfg) 控制石墩的半径、数量等。
    """
    # 1. 从配置中获取地形尺寸和自定义参数
    width, length = cfg.size
    # 这些是我们将在配置类中定义的自定义参数
    stone_radius = cfg.stone_radius
    grid_resolution = cfg.grid_resolution
    
    # 2. 利用难度参数 (difficulty) 来调整地形特征
    # 难度越高，石墩的最高高度就越大
    max_stone_height = cfg.stone_height_range[0] + difficulty * (cfg.stone_height_range[1] - cfg.stone_height_range[0])
    
    # 3. 创建几何体 (使用 trimesh 库)
    meshes = []

    # 创建一个薄的基础平面，防止机器人掉落
    base_plane = trimesh.creation.box(extents=[width, length, 0.05])
    # 将基础平面稍微下移
    base_plane.apply_translation([width / 2, length / 2, -0.025])
    meshes.append(base_plane)

    # 在网格上生成汀步石
    x_coords = np.arange(0, width, grid_resolution)
    y_coords = np.arange(0, length, grid_resolution)

    for x in x_coords:
        for y in y_coords:
            # 随机决定石墩的高度
            stone_height = np.random.uniform(0.05, max_stone_height)
            if stone_height < 0.06: # 有一定概率不生成石墩
                continue
            
            # 创建一个圆柱体作为石墩
            stone = trimesh.creation.cylinder(radius=stone_radius, height=stone_height)
            # 将石墩移动到正确的位置
            # 注意：trimesh 圆柱体的中心在 (0,0,0)，所以 Z 坐标是高度的一半
            stone.apply_translation([x, y, stone_height / 2])
            meshes.append(stone)
            
    # 4. 定义地形原点
    # 因为我们的坐标是从 (0,0) 开始的，所以原点就是 (0,0,0)
    origin = np.array([0.0, 0.0, 0.0])

    # 5. 返回 meshes 列表和 origin
    return meshes, origin