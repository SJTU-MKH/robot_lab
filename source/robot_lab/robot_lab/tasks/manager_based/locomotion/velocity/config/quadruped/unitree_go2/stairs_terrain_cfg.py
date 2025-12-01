# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for 2-step stairs terrain for climbing training."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

# 简单的2阶台阶地形配置
TWO_STEP_STAIRS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,  # 支持课程学习
    sub_terrains={
        # 只使用金字塔台阶地形，2阶台阶
        "two_step_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,  # 100% 台阶地形
            step_height_range=(0.08, 0.15),  # 台阶高度范围 8-15cm
            step_width=0.35,  # 每阶台阶宽度 35cm
            platform_width=3.0,  # 平台宽度
            border_width=1.0,
            holes=False,  # 不要空洞
        ),
    },
)
"""2阶台阶地形配置，用于训练机器人上下楼梯。"""
