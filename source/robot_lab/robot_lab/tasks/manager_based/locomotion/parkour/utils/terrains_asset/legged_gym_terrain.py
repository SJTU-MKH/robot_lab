# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from .mesh_parkour.parkour_terrain_cfg import SteppingStonesTerrainCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from .mesh_parkour.parkour_terrain_cfg import ParkourGapTerrainCfg
from .mesh_parkour.parkour_terrain_cfg import ParkourSlopeTerrainCfg
from .mesh_parkour.parkour_terrain_cfg import ParkourStairsTerrainCfg


PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stepping_stones": SteppingStonesTerrainCfg(
            proportion=0.1,
            stone_radius=0.2,
            stone_height_range=(0.05, 0.2),
            grid_resolution=1.0,
        ),
        "parkour_gap_terrain": ParkourGapTerrainCfg(
            proportion=0.1,
        ),
        "parkour_slope_terrain": ParkourSlopeTerrainCfg(
            proportion=0.1,
        ),
        "park_stairs_terrain": ParkourStairsTerrainCfg(
            proportion=0.1,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
