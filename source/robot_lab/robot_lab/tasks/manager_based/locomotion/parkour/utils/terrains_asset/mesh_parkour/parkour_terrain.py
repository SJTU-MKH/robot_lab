import numpy as np
import trimesh
import math
import random

# 导入你的配置类（我们将在下一步创建它）
# from my_terrains_cfg import SteppingStonesTerrainCfg
# 为了让代码独立运行，我们先用 SubTerrainBaseCfg 占位
from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg


def stepping_stones_generator(difficulty: float, cfg: SubTerrainBaseCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    width, length = cfg.size
    stone_radius = cfg.stone_radius
    grid_resolution = cfg.grid_resolution

    max_stone_height = cfg.stone_height_range[0] + difficulty * (cfg.stone_height_range[1] - cfg.stone_height_range[0])

    meshes = []

    # 创建一个薄的基础平面，防止机器人掉落
    base_plane = trimesh.creation.box(extents=[width, length, 0.05])
    # 将基础平面稍微下移
    base_plane.apply_translation([width / 2, length / 2, -0.025])
    meshes.append(base_plane)

    x_coords = np.arange(0, width, grid_resolution)
    y_coords = np.arange(0, length, grid_resolution)

    for x in x_coords:
        for y in y_coords:
            # 随机决定石墩的高度
            stone_height = np.random.uniform(0.05, max_stone_height)
            # 有一定概率不生成石墩
            if stone_height < 0.06:
                continue

            stone = trimesh.creation.cylinder(radius=stone_radius, height=stone_height)
            # 将石墩移动到正确的位置
            # 注意：trimesh 圆柱体的中心在 (0,0,0)，所以 Z 坐标是高度的一半
            stone.apply_translation([x, y, stone_height / 2])
            meshes.append(stone)

    origin = np.array([0.0, 0.0, 0.0])

    return meshes, origin


def parkour_gap_generator(difficulty: float, cfg: SubTerrainBaseCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    X, Y = cfg.size
    gap_range = cfg.gap_range
    platform_range = cfg.platform_range
    gap_width = gap_range[0] + difficulty * (gap_range[1] - gap_range[0])
    platform_width = platform_range[0] + difficulty * (platform_range[1] - platform_range[0])
    platform_length = cfg.platform_length

    pad_height = cfg.pad_height
    pad_width = cfg.pad_width

    meshes = []

    # 创建底板，防止机器人掉落
    base_plane = trimesh.creation.box(extents=[X, Y, 0.1])
    base_plane.apply_translation([X / 2, Y / 2, -0.05 - pad_height])
    meshes.append(base_plane)

    # 起始平台
    start_platform = trimesh.creation.box(extents=[1, platform_length, pad_height])
    start_platform.apply_translation([0.5, Y / 2, -pad_height / 2])
    meshes.append(start_platform)

    x_offset = pad_width + 1.0 + gap_width
    # 生成平台和间隔序列
    for i in range(0, math.floor((X - 2.0 - pad_width * 2) / (gap_width + platform_width))):
        platform = trimesh.creation.box(extents=[platform_width, platform_length, pad_height])
        platform.apply_translation([x_offset + platform_width / 2, Y / 2, -pad_height / 2])
        meshes.append(platform)
        x_offset += platform_width

        x_offset += gap_width

    # 终点平台（至少为1m）
    terminal_platform = trimesh.creation.box(extents=[X - x_offset - pad_width, platform_length, pad_height])
    terminal_platform.apply_translation([X - pad_width - (X - x_offset - pad_width) / 2, Y / 2, -pad_height / 2])
    meshes.append(terminal_platform)

    # 边界pad（沿 y 方向）
    pad1 = trimesh.creation.box(extents=[pad_width, Y, pad_height])
    pad1.apply_translation([pad_width / 2, Y / 2, -pad_height / 2])
    meshes.append(pad1)

    pad2 = trimesh.creation.box(extents=[pad_width, Y, pad_height])
    pad2.apply_translation([X - pad_width / 2, Y / 2, -pad_height / 2])
    meshes.append(pad2)

    # 边界 pad（沿 x 方向）
    pad3 = trimesh.creation.box(extents=[X, pad_width, pad_height])
    pad3.apply_translation([X / 2, pad_width / 2, -pad_height / 2])
    meshes.append(pad3)

    pad4 = trimesh.creation.box(extents=[X, pad_width, pad_height])
    pad4.apply_translation([X / 2, Y - pad_width / 2, -pad_height / 2])
    meshes.append(pad4)

    origin = np.array([0.0, 0.0, 0.0])
    return meshes, origin


def parkour_slope_generator(difficulty: float, cfg: SubTerrainBaseCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    X, Y = cfg.size

    platform_range = cfg.platform_range
    platform_height = platform_range[0] + difficulty * (platform_range[1] - platform_range[0])
    platform_width = cfg.platform_width
    platform_length = cfg.platform_length

    pad_height = cfg.pad_height
    pad_width = cfg.pad_width

    meshes = []

    # 创建底板，防止机器人掉落
    base_plane = trimesh.creation.box(extents=[X, Y, 0.1])
    base_plane.apply_translation([X / 2, Y / 2, -0.05 - pad_height])
    meshes.append(base_plane)

    # 起始平台
    start_platform = trimesh.creation.box(extents=[1, platform_length, pad_height])
    start_platform.apply_translation([0.5, Y / 2, -pad_height / 2])
    meshes.append(start_platform)

    # 生成平台和间隔序列
    sign = 1.0
    x_offset = pad_width + 1.0
    for i in range(0, math.floor((X - 2.0 - pad_width * 2) / platform_width)):
        # 相邻平台朝向不同，错开一半
        slope = _make_triangular_prism(
            vertex=[x_offset, Y / 2 - sign * platform_length / 2 - sign * platform_length / 4, - pad_height],
            length=platform_width,
            base=sign * platform_length,
            height=platform_height,
        )
        meshes.append(slope)
        x_offset += platform_width
        sign *= -1.0

    terminal_platform = trimesh.creation.box(extents=[X - x_offset - pad_width, platform_length, pad_height])
    terminal_platform.apply_translation([X - pad_width - (X - x_offset - pad_width) / 2, Y / 2, -pad_height / 2])
    meshes.append(terminal_platform)

    # 边界pad（沿 y 方向）
    pad1 = trimesh.creation.box(extents=[pad_width, Y, pad_height])
    pad1.apply_translation([pad_width / 2, Y / 2, -pad_height / 2])
    meshes.append(pad1)

    pad2 = trimesh.creation.box(extents=[pad_width, Y, pad_height])
    pad2.apply_translation([X - pad_width / 2, Y / 2, -pad_height / 2])
    meshes.append(pad2)

    # 边界 pad（沿 x 方向）
    pad3 = trimesh.creation.box(extents=[X, pad_width, pad_height])
    pad3.apply_translation([X / 2, pad_width / 2, -pad_height / 2])
    meshes.append(pad3)

    pad4 = trimesh.creation.box(extents=[X, pad_width, pad_height])
    pad4.apply_translation([X / 2, Y - pad_width / 2, -pad_height / 2])
    meshes.append(pad4)

    origin = np.array([0.0, 0.0, 0.0])
    return meshes, origin


def parkour_stairs_generator(difficulty: float, cfg: SubTerrainBaseCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    X, Y = cfg.size

    step_range = cfg.step_range
    step_height = step_range[0] + difficulty * (step_range[1] - step_range[0])
    step_width = cfg.step_width
    step_length = cfg.step_length

    meshes = []
    # 创建底板，防止机器人掉落
    base_plane = trimesh.creation.box(extents=[X, Y, 0.1])
    base_plane.apply_translation([X / 2, Y / 2, -0.05])
    meshes.append(base_plane)

    x_offset = 1.0
    # step_num为偶数时，会拿出一个台阶高度不变
    step_num = math.floor((X - 2.0) / step_width)
    if step_num % 2 != 0:
        change_height_count = step_num + 1
    else:
        change_height_count = step_num
    height = 0
    changes = [-1] * (change_height_count // 2) + [1] * (change_height_count // 2)
    print(step_num)
    print(changes)
    for i in range(change_height_count):
        choice = random.choice(changes)
        if choice == -1:
            height += choice
            if height < 0 or (height == 0 and 1 in changes):
                height += 2
                choice = 1
        else:
            height += choice
        changes.remove(choice)
        if step_num % 2 == 0 and i == change_height_count // 2:
            step = trimesh.creation.box(extents=[step_width, step_length, step_height * height])
            step.apply_translation([x_offset, Y / 2, (step_height * height) / 2])
            meshes.append(step)
        if height > 0:
            step = trimesh.creation.box(extents=[step_width, step_length, step_height * height])
            step.apply_translation([x_offset, Y / 2, (step_height * height) / 2])
            meshes.append(step)
        x_offset += step_width

    origin = np.array([0.0, 0.0, 0.0])
    return meshes, origin


def _make_triangular_prism(vertex=[0.0, 0.0, 0.0], length=1.0, base=1.0, height=1.0, transform=None, **kwargs):
    # base:   直角三角形底边长度（沿 y 轴）
    # 通过base的正负决定三角形的朝向
    # height: 直角三角形高（沿 z 轴）

    # 原点、沿y顶点、沿z顶点
    vertex = np.asarray(vertex, dtype=np.float64)
    tri = np.array([
        vertex,
        [vertex[0], vertex[1] + base, vertex[2]],
        [vertex[0], vertex[1], vertex[2] + height],
    ])

    # 沿 x 方向拉伸出棱柱
    # 前底面和后顶面
    vertices = np.vstack([
        tri,
        tri + [length, 0, 0]
    ])

    # 面的索引定义（每个面都是三角形）
    faces = np.array([
        [0, 1, 2],          # 前三角形
        [3, 5, 4],          # 后三角形

        [0, 3, 1], [1, 3, 4],  # 底边矩形面
        [0, 2, 3], [2, 5, 3],  # 竖直矩形面
        [1, 2, 4], [2, 5, 4],  # 斜向矩形面（最大侧面）
    ])

    # 元信息
    if "metadata" not in kwargs:
        kwargs["metadata"] = {}
    kwargs["metadata"].update({
        "shape": "triangular_prism",
        "vertex": vertex,
        "length": length,
        "base": base,
        "height": height,
    })

    # 构建 Trimesh
    prism = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
        **kwargs
    )

    # 应用变换（如果提供）
    if transform is not None:
        prism.apply_transform(transform)

    return prism