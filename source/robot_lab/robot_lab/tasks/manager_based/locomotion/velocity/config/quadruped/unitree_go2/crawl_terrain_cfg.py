# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for crawl terrain with low obstacles suitable for crawling."""

from robot_lab.tasks.manager_based.locomotion.velocity.utils.terrains_asset.mesh_crawl import (
    ThreeTunnelChamberTerrainCfg,
    CrawlTerrainGeneratorCfg,
)

# 使用自定义mesh地形生成器创建匍匐前进场景
# 主要特点：需要机器人降低身体姿态爬行穿越的障碍物和隧道
# 包含导航点系统：起点 + 3个隧道中心 + 终点

CRAWL_TERRAINS_CFG = CrawlTerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    num_goals=5,  # 5个导航点
    sub_terrains={
        # 三隧道房间地形 - 四周围墙，三个不对齐的隧道，包含导航点
        "three_tunnel_chamber": ThreeTunnelChamberTerrainCfg(
            proportion=1.0,  # 只使用这一种地形
            wall_height=2.0,
            wall_thickness=1.0,
            tunnel_wall_thickness_range=(0.2, 0.5),
            tunnel_height_range=(0.25, 0.5),
            tunnel_width_range=(0.5, 0.8),
        ),
    },
)
