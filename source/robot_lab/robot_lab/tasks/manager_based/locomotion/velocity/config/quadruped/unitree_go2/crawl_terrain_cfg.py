# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for crawl terrain with low obstacles suitable for crawling."""

import isaaclab.terrains as terrain_gen
from .crawl_terrain_mesh_cfg import LowObstaclesTerrainCfg, TunnelTerrainCfg

# 使用自定义mesh地形生成器创建匍匐前进场景
# 主要特点：需要机器人降低身体姿态爬行穿越的障碍物和隧道

CRAWL_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # 自定义低矮障碍物地形 - 需要匍匐穿越
        "low_obstacles_mesh": LowObstaclesTerrainCfg(
            proportion=0.5,
            obstacle_height_range=(0.15, 0.35),  # 低矮障碍物高度
            obstacle_spacing=1.2,  # 障碍物间距
            obstacle_width_range=(0.8, 2.0),  # 障碍物宽度
        ),
        # 自定义隧道地形 - 需要爬行通过
        "tunnel_mesh": TunnelTerrainCfg(
            proportion=0.2,
            tunnel_height=0.4,  # 隧道高度
            tunnel_width=1.5,  # 隧道宽度
        ),
        # 平坦地形：用于恢复和基础移动
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.0, 0.02), noise_step=0.01, border_width=0.25
        ),
    },
)
