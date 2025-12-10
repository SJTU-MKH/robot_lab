import numpy as np
import trimesh

# 导入你的配置类（我们将在下一步创建它）
# from my_terrains_cfg import SteppingStonesTerrainCfg
# 为了让代码独立运行，我们先用 SubTerrainBaseCfg 占位
from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg


def boxgarden_generator(difficulty: float, cfg: SubTerrainBaseCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    X, Y = cfg.size
    wall_height = cfg.wall_height

    meshes = []

    # 创建一个薄的基础平面，防止机器人掉落
    base_plane = trimesh.creation.box(extents=[X, Y, 0.05])
    # 将基础平面稍微下移
    base_plane.apply_translation([X / 2, Y / 2, -0.025])
    meshes.append(base_plane)

    ################################
    # 创建中间的墙壁
    ################################
    ref = X
    wall1 = trimesh.creation.box(extents=[0.18 * ref, 0.1 * ref, wall_height])
    wall1.apply_translation([0.21 * ref, 0.13 * ref, wall_height / 2])

    wall2 = trimesh.creation.box(extents=[0.24 * ref, 0.12 * ref, wall_height])
    wall2.apply_translation([0.56 * ref, 0.2 * ref, wall_height / 2])

    wall3 = trimesh.creation.box(extents=[0.12 * ref, 0.08 * ref, wall_height])
    wall3.apply_translation([0.94 * ref, 0.22 * ref, wall_height / 2])

    wall4 = trimesh.creation.box(extents=[0.26 * ref, 0.26 * ref, wall_height])
    wall4.apply_translation([0.19 * ref, 0.45 * ref, wall_height / 2])

    wall5 = trimesh.creation.box(extents=[0.12 * ref, 0.22 * ref, wall_height])
    wall5.apply_translation([0.76 * ref, 0.49 * ref, wall_height / 2])

    wall6 = trimesh.creation.box(extents=[0.24 * ref, 0.10 * ref, wall_height])
    wall6.apply_translation([0.54 * ref, 0.63 * ref, wall_height / 2])

    wall7 = trimesh.creation.box(extents=[0.42 * ref, 0.14 * ref, wall_height])
    wall7.apply_translation([0.23 * ref, 0.77 * ref, wall_height / 2])

    meshes.append(wall1)
    meshes.append(wall2)
    meshes.append(wall3)
    meshes.append(wall4)
    meshes.append(wall5)
    meshes.append(wall6)
    meshes.append(wall7)

    # 边界pad（沿 y 方向）
    pad1 = trimesh.creation.box(extents=[0.2, Y, wall_height])
    pad1.apply_translation([0.2 / 2, Y / 2, wall_height / 2])
    meshes.append(pad1)

    pad2 = trimesh.creation.box(extents=[0.2, Y, wall_height])
    pad2.apply_translation([X - 0.2 / 2, Y / 2, wall_height / 2])
    meshes.append(pad2)

    # 边界 pad（沿 x 方向）
    pad3 = trimesh.creation.box(extents=[X, 0.2, wall_height])
    pad3.apply_translation([X / 2, 0.2 / 2, wall_height / 2])
    meshes.append(pad3)

    pad4 = trimesh.creation.box(extents=[X, 0.2, wall_height])
    pad4.apply_translation([X / 2, Y - 0.2 / 2, wall_height / 2])
    meshes.append(pad4)

    origin = np.array([0.8 * ref, 0.8 * ref, 0.0])

    return meshes, origin
