# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for crawl-specific terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .mesh_crawl import (
    LowObstaclesTerrainCfg,
    TunnelTerrainCfg,
    CrawlBarriersTerrainCfg,
    CrawlMazeTerrainCfg,
)


CRAWL_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # 低矮障碍物地形 - 主要爬行地形
        "low_obstacles": LowObstaclesTerrainCfg(
            proportion=0.3,
            obstacle_height_range=(0.1, 0.3),
            obstacle_width=0.8,
            obstacle_length=0.6,
            spacing_x=1.2,
            spacing_y=1.0,
            obstacle_density=0.6,
        ),
        
        # 隧道地形 - 要求低姿态通过
        "tunnel": TunnelTerrainCfg(
            proportion=0.15,
            tunnel_height_range=(0.25, 0.5),
            tunnel_width=1.8,
            wall_thickness=0.3,
        ),
        
        # 爬行障碍地形 - 交错障碍物
        "crawl_barriers": CrawlBarriersTerrainCfg(
            proportion=0.15,
            barrier_height_range=(0.15, 0.35),
            barrier_width=0.4,
            gap_width=1.2,
        ),
        
        # 爬行迷宫 - 复杂导航
        "crawl_maze": CrawlMazeTerrainCfg(
            proportion=0.1,
            maze_height=0.35,
            wall_thickness=0.2,
            corridor_width=1.5,
        ),
        
        # 平地恢复区域 - 让机器人恢复正常姿态
        "flat_recovery": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3,
            noise_range=(0.0, 0.02),  # 几乎平地
            noise_step=0.01,
            border_width=0.25,
        ),
    },
)


# 简化版爬行地形配置 - 更适合初期训练
SIMPLE_CRAWL_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # 主要是低矮障碍物
        "low_obstacles": LowObstaclesTerrainCfg(
            proportion=0.7,
            obstacle_height_range=(0.1, 0.25),  # 更低的障碍物
            obstacle_width=1.0,
            obstacle_length=0.8,
            spacing_x=1.5,
            spacing_y=1.2,
            obstacle_density=0.5,  # 更稀疏的障碍物
        ),
        
        # 平地区域用于恢复
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3,
            noise_range=(0.0, 0.01),
            noise_step=0.005,
            border_width=0.25,
        ),
    },
)