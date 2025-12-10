import numpy as np
import trimesh
from collections.abc import Callable

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg


def low_obstacles_generator(
    difficulty: float, cfg: SubTerrainBaseCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """
    生成需要匍匐前进的低矮障碍物地形。

    - 难度 (difficulty) 会影响障碍物的密度和高度。
    - 地形参数 (cfg) 控制障碍物的尺寸、分布等。
    """
    # 1. 从配置中获取地形尺寸和自定义参数
    width, length = cfg.size
    obstacle_height_range = cfg.obstacle_height_range
    obstacle_spacing = cfg.obstacle_spacing
    obstacle_width_range = cfg.obstacle_width_range

    # 2. 利用难度参数调整地形特征
    # 难度越高，障碍物越高越密集
    max_height = obstacle_height_range[0] + difficulty * (
        obstacle_height_range[1] - obstacle_height_range[0]
    )
    min_spacing = obstacle_spacing * (1.5 - difficulty * 0.5)  # 难度高时间距更小

    # 3. 创建几何体列表
    meshes = []

    # 创建基础地面
    base_plane = trimesh.creation.box(extents=[width, length, 0.05])
    base_plane.apply_translation([width / 2, length / 2, -0.025])
    meshes.append(base_plane)

    # 生成低矮障碍物 - 模拟需要爬行穿越的环境
    x_positions = np.arange(1.0, width - 1.0, min_spacing)
    y_positions = np.arange(1.0, length - 1.0, min_spacing)

    for x in x_positions:
        for y in y_positions:
            # 随机决定是否在此位置放置障碍物
            if np.random.random() < 0.6:  # 60%概率放置障碍物
                # 随机障碍物尺寸
                obstacle_width = np.random.uniform(*obstacle_width_range)
                obstacle_length = np.random.uniform(0.3, 0.8)
                obstacle_height = np.random.uniform(0.1, max_height)

                # 创建长方体障碍物
                obstacle = trimesh.creation.box(
                    extents=[obstacle_width, obstacle_length, obstacle_height]
                )

                # 添加随机偏移避免过于规则
                x_offset = np.random.uniform(-0.2, 0.2)
                y_offset = np.random.uniform(-0.2, 0.2)

                obstacle.apply_translation(
                    [x + x_offset, y + y_offset, obstacle_height / 2]
                )
                meshes.append(obstacle)

    # 4. 定义地形原点
    origin = np.array([0.0, 0.0, 0.0])

    return meshes, origin


def tunnel_generator(
    difficulty: float, cfg: SubTerrainBaseCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """
    生成隧道地形，机器人需要爬行通过。
    """
    width, length = cfg.size
    tunnel_height = cfg.tunnel_height
    tunnel_width = cfg.tunnel_width

    meshes = []

    # 基础地面
    base_plane = trimesh.creation.box(extents=[width, length, 0.05])
    base_plane.apply_translation([width / 2, length / 2, -0.025])
    meshes.append(base_plane)

    # 调整隧道高度基于难度
    actual_tunnel_height = tunnel_height * (1.2 - difficulty * 0.4)  # 难度高时隧道更低

    # 创建隧道顶部 - 在中间留出隧道空间
    tunnel_start_y = length * 0.3
    tunnel_end_y = length * 0.7
    tunnel_start_x = (width - tunnel_width) / 2
    tunnel_end_x = tunnel_start_x + tunnel_width

    # 左侧墙
    if tunnel_start_x > 0:
        left_wall = trimesh.creation.box(
            extents=[tunnel_start_x, length, actual_tunnel_height]
        )
        left_wall.apply_translation(
            [tunnel_start_x / 2, length / 2, actual_tunnel_height / 2]
        )
        meshes.append(left_wall)

    # 右侧墙
    if tunnel_end_x < width:
        right_wall_width = width - tunnel_end_x
        right_wall = trimesh.creation.box(
            extents=[right_wall_width, length, actual_tunnel_height]
        )
        right_wall.apply_translation(
            [tunnel_end_x + right_wall_width / 2, length / 2, actual_tunnel_height / 2]
        )
        meshes.append(right_wall)

    # 隧道顶部 - 在隧道区域上方
    tunnel_roof = trimesh.creation.box(
        extents=[tunnel_width, tunnel_end_y - tunnel_start_y, 0.1]
    )
    tunnel_roof.apply_translation(
        [width / 2, (tunnel_start_y + tunnel_end_y) / 2, actual_tunnel_height + 0.05]
    )
    meshes.append(tunnel_roof)

    origin = np.array([0.0, 0.0, 0.0])
    return meshes, origin
