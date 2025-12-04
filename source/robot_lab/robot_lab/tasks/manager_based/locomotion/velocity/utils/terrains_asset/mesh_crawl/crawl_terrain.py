import numpy as np
import trimesh
import random

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg


def three_tunnel_chamber_generator(
    difficulty: float, cfg: SubTerrainBaseCfg, num_goals: int = 3
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray, np.ndarray]:
    """
    生成带有三个隧道的房间地形，四周有1米围墙，三个隧道位置可以不居中对齐。
    每个隧道中心设置导航点。

    Args:
        difficulty: 难度系数 (0.0 to 1.0)
        cfg: 地形配置
        num_goals: 导航点数量

    Returns:
        tuple: (mesh列表, 原点坐标, 导航点坐标, 导航点高度)
    """
    width, length = cfg.size
    meshes = []

    # 创建基础平面
    base_plane = trimesh.creation.box(extents=[width, length, 0.05])
    base_plane.apply_translation([width / 2, length / 2, -0.025])
    meshes.append(base_plane)

    # 围墙参数
    wall_height = cfg.wall_height
    wall_thickness = cfg.wall_thickness

    # 创建四周围墙
    # 左墙
    left_wall = trimesh.creation.box(extents=[wall_thickness, length, wall_height])
    left_wall.apply_translation([wall_thickness / 2, length / 2, wall_height / 2])
    meshes.append(left_wall)

    # 右墙
    right_wall = trimesh.creation.box(extents=[wall_thickness, length, wall_height])
    right_wall.apply_translation(
        [width - wall_thickness / 2, length / 2, wall_height / 2]
    )
    meshes.append(right_wall)

    # 前墙（入口端）
    front_wall = trimesh.creation.box(extents=[width, wall_thickness, wall_height])
    front_wall.apply_translation([width / 2, wall_thickness / 2, wall_height / 2])
    meshes.append(front_wall)

    # 后墙（出口端）
    back_wall = trimesh.creation.box(extents=[width, wall_thickness, wall_height])
    back_wall.apply_translation(
        [width / 2, length - wall_thickness / 2, wall_height / 2]
    )
    meshes.append(back_wall)

    # 隧道参数（根据难度调整）
    min_wall_thickness, max_wall_thickness = cfg.tunnel_wall_thickness_range
    tunnel_wall_thickness = min_wall_thickness + difficulty * (
        max_wall_thickness - min_wall_thickness
    )

    min_tunnel_height, max_tunnel_height = cfg.tunnel_height_range
    tunnel_height = max_tunnel_height - difficulty * (
        max_tunnel_height - min_tunnel_height
    )

    min_tunnel_width, max_tunnel_width = cfg.tunnel_width_range
    tunnel_width = min_tunnel_width + difficulty * (max_tunnel_width - min_tunnel_width)

    # 可用内部空间（去除围墙）
    inner_width = width - 2 * wall_thickness
    inner_length = length - 2 * wall_thickness
    inner_start_x = wall_thickness
    inner_start_y = wall_thickness

    # 创建三个分隔墙和隧道
    tunnel_configs = []
    goals = np.zeros((num_goals, 2))
    goal_heights = np.zeros(num_goals)

    # 设置起点（索引0）：入口处中央位置
    start_x = width / 2
    start_y = wall_thickness + 0.3  # 入口墙后0.3米
    goals[0] = [start_x, start_y]
    goal_heights[0] = 0.15  # 爬行高度

    # 三个隧道的Y位置（分隔位置）
    divider_positions = [inner_length / 3, 2 * inner_length / 3]

    # 创建两个分隔墙（有隧道开口）
    for i, divider_y in enumerate(divider_positions):
        divider_abs_y = inner_start_y + divider_y

        # 随机隧道X位置（不居中对齐）
        tunnel_x_offset = random.uniform(-inner_width / 4, inner_width / 4)
        tunnel_center_x = inner_start_x + inner_width / 2 + tunnel_x_offset

        # 确保隧道不超出边界
        tunnel_center_x = max(
            inner_start_x + tunnel_width / 2,
            min(inner_start_x + inner_width - tunnel_width / 2, tunnel_center_x),
        )

        # 创建分隔墙（左侧部分）
        left_divider_width = tunnel_center_x - tunnel_width / 2 - inner_start_x
        if left_divider_width > 0.1:  # 只有足够宽度才创建
            left_divider = trimesh.creation.box(
                extents=[left_divider_width, tunnel_wall_thickness, wall_height]
            )
            left_divider.apply_translation(
                [inner_start_x + left_divider_width / 2, divider_abs_y, wall_height / 2]
            )
            meshes.append(left_divider)

        # 创建分隔墙（右侧部分）
        right_divider_start_x = tunnel_center_x + tunnel_width / 2
        right_divider_width = inner_start_x + inner_width - right_divider_start_x
        if right_divider_width > 0.1:  # 只有足够宽度才创建
            right_divider = trimesh.creation.box(
                extents=[right_divider_width, tunnel_wall_thickness, wall_height]
            )
            right_divider.apply_translation(
                [
                    right_divider_start_x + right_divider_width / 2,
                    divider_abs_y,
                    wall_height / 2,
                ]
            )
            meshes.append(right_divider)

        # 创建隧道顶部
        tunnel_roof = trimesh.creation.box(
            extents=[tunnel_width, tunnel_wall_thickness, wall_height - tunnel_height]
        )
        tunnel_roof.apply_translation(
            [
                tunnel_center_x,
                divider_abs_y,
                tunnel_height + (wall_height - tunnel_height) / 2,
            ]
        )
        meshes.append(tunnel_roof)

        # 记录隧道配置用于设置导航点
        tunnel_configs.append(
            {
                "center_x": tunnel_center_x,
                "center_y": divider_abs_y,
                "height": tunnel_height / 2,  # 隧道中心高度
            }
        )

    # 第三个隧道在最后一个房间中间（可选）
    if num_goals >= 3:
        # 在最后一个区域添加额外的导航点
        last_room_y = inner_start_y + 2 * inner_length / 3 + inner_length / 6
        last_room_x = (
            inner_start_x
            + inner_width / 2
            + random.uniform(-inner_width / 4, inner_width / 4)
        )
        last_room_x = max(
            inner_start_x + 0.5, min(inner_start_x + inner_width - 0.5, last_room_x)
        )

        tunnel_configs.append(
            {
                "center_x": last_room_x,
                "center_y": last_room_y,
                "height": 0.1,  # 地面上方一点
            }
        )

    # 设置导航点（索引1-3为三个隧道）
    for i in range(min(3, len(tunnel_configs))):
        goals[i + 1] = [tunnel_configs[i]["center_x"], tunnel_configs[i]["center_y"]]
        goal_heights[i + 1] = tunnel_configs[i]["height"]

    # 设置终点（索引4）：出口处中央位置
    if num_goals >= 5:
        end_x = width / 2
        end_y = length - wall_thickness - 0.3  # 出口墙前0.3米
        goals[4] = [end_x, end_y]
        goal_heights[4] = 0.15  # 爬行高度

    origin = np.array([0.0, 0.0, 0.0])
    return meshes, origin, goals, goal_heights
