# Copyright (c) 2022-2025, The Isaac Lab Project Developers. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for terrains inspired by the original `legged_gym` project.

This file translates the procedural logic from `legged_gym`'s Terrain class
into the declarative configuration style of Isaac Lab. Each terrain type
is defined by a @configclass and a corresponding generator function.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.sub_terrain_cfg import SubTerrain, SubTerrainBaseCfg
from isaaclab.utils import configclass


# Note: `terrain_gen` refers to the built-in Isaac Lab terrain generators.
# We will define our own generators in this file for the custom legged_gym logic.


# -- Begin Helper functions from legged_gym (needed for some generators) --
# (Here we can add any complex helper functions like parkour_terrain if needed.
# For this example, we will focus on the more standard terrain types.)

# -- End Helper functions --


# Definine custom generator functions and their @configclass configurations

# 1. Sloped Terrain (with optional roughness)
# Corresponds to proportions[0] to proportions[2] in the original code.
@configclass
class LgSlopedTerrainCfg(terrain_gen.TerrainCfg):
    """Configuration for a sloped terrain, similar to legged_gym."""
    slope_range: tuple[float, float] = (0.1, 0.4)
    platform_width: float = 3.0
    add_roughness: bool = True
    roughness_range: tuple[float, float] = (0.05, 0.1)


def lg_sloped_terrain_generator(terrain: SubTerrain, cfg: LgSlopedTerrainCfg):
    """Generates a sloped terrain with optional uniform noise."""
    # Create the base slope
    terrain_gen.pyramid_sloped_terrain(terrain, slope=cfg.slope, platform_size=cfg.platform_width)
    # Add roughness if enabled
    if cfg.add_roughness:
        terrain_gen.random_uniform_terrain(terrain, min_height=-cfg.roughness, max_height=cfg.roughness, step=0.005)


# 2. Pyramid Stairs Terrain (with roughness)
# Corresponds to proportions[2] to proportions[4].
@configclass
class LgPyramidStairsTerrainCfg(terrain_gen.TerrainCfg):
    """Configuration for pyramid stairs terrain."""
    step_height_range: tuple[float, float] = (0.05, 0.15)
    step_width: float = 0.31
    platform_width: float = 3.0
    add_roughness: bool = True
    roughness_range: tuple[float, float] = (0.05, 0.1)


def lg_pyramid_stairs_terrain_generator(terrain: SubTerrain, cfg: LgPyramidStairsTerrainCfg):
    """Generates pyramid stairs with optional roughness."""
    terrain_gen.pyramid_stairs_terrain(
        terrain, step_width=cfg.step_width, step_height=cfg.step_height, platform_size=cfg.platform_width
    )
    if cfg.add_roughness:
        terrain_gen.random_uniform_terrain(terrain, min_height=-cfg.roughness, max_height=cfg.roughness, step=0.005)


# 3. Discrete Obstacles Terrain (Boxes)
# Corresponds to proportions[4] to proportions[5].
@configclass
class LgDiscreteObstaclesTerrainCfg(terrain_gen.TerrainCfg):
    """Configuration for discrete obstacles terrain."""
    height_range: tuple[float, float] = (0.05, 0.20)
    min_obstacle_size: float = 0.5
    max_obstacle_size: float = 2.0
    num_obstacles: int = 20
    platform_width: float = 3.0
    add_roughness: bool = True
    roughness_range: tuple[float, float] = (0.05, 0.1)


def lg_discrete_obstacles_terrain_generator(terrain: SubTerrain, cfg: LgDiscreteObstaclesTerrainCfg):
    """Generates discrete obstacles with optional roughness."""
    terrain_gen.discrete_obstacles_terrain(
        terrain,
        max_height=cfg.height,
        min_size=cfg.min_obstacle_size,
        max_size=cfg.max_obstacle_size,
        num_rects=cfg.num_obstacles,
        platform_size=cfg.platform_width,
    )
    if cfg.add_roughness:
        terrain_gen.random_uniform_terrain(terrain, min_height=-cfg.roughness, max_height=cfg.roughness, step=0.005)


# 4. Stepping Stones Terrain
# Corresponds to proportions[5] to proportions[6].
@configclass
class LgSteppingStonesTerrainCfg(terrain_gen.TerrainCfg):
    """Configuration for stepping stones terrain."""
    # Note: In original code, stone size decreased with difficulty.
    # We model this by having the range go from large to small.
    stone_size_range: tuple[float, float] = (1.5, 0.3)
    stone_distance_range: tuple[float, float] = (0.05, 0.5)
    max_height_range: tuple[float, float] = (0.0, 0.2)
    platform_width: float = 1.2
    add_roughness: bool = True
    roughness_range: tuple[float, float] = (0.05, 0.1)


def lg_stepping_stones_terrain_generator(terrain: SubTerrain, cfg: LgSteppingStonesTerrainCfg):
    """Generates stepping stones with optional roughness."""
    terrain_gen.stepping_stones_terrain(
        terrain,
        stone_size=cfg.stone_size,
        stone_distance=cfg.stone_distance,
        max_height=cfg.max_height,
        platform_size=cfg.platform_width,
        depth=-10.0,  # A deep pit
    )
    if cfg.add_roughness:
        terrain_gen.random_uniform_terrain(terrain, min_height=-cfg.roughness, max_height=cfg.roughness, step=0.005)


# 5. Gap/Pit Terrain
# Combines Gap (prop[7]) and Pit (prop[9]) logic.
@configclass
class LgGapPitTerrainCfg(terrain_gen.TerrainCfg):
    """Configuration for terrains with gaps or pits."""
    gap_size_range: tuple[float, float] = (0.1, 1.0)
    pit_depth_range: tuple[float, float] = (0.1, 1.0)
    platform_width: float = 4.0
    add_roughness: bool = True
    roughness_range: tuple[float, float] = (0.05, 0.1)
    # Custom flag to decide if it's a gap or a full pit
    is_pit: bool = False


def lg_gap_pit_terrain_generator(terrain: SubTerrain, cfg: LgGapPitTerrainCfg):
    """Generates a gap or a pit."""
    if cfg.is_pit:
        depth_in_meters = cfg.pit_depth
        depth_in_units = -int(depth_in_meters / terrain.vertical_scale)
        platform_size_in_units = int(cfg.platform_width / terrain.horizontal_scale / 2)
        
        x1 = terrain.length // 2 - platform_size_in_units
        x2 = terrain.length // 2 + platform_size_in_units
        y1 = terrain.width // 2 - platform_size_in_units
        y2 = terrain.width // 2 + platform_size_in_units
        terrain.height_field_raw[x1:x2, y1:y2] = depth_in_units
    else:
        # Using isaaclab's built-in gap terrain as a close equivalent
        terrain_gen.gap_terrain(terrain, gap_width=cfg.gap_size, platform_width=cfg.platform_width, depth=-10.0)
    
    if cfg.add_roughness:
        terrain_gen.random_uniform_terrain(terrain, min_height=-cfg.roughness, max_height=cfg.roughness, step=0.005)


# The main configuration that combines all the above terrains.
# You can import this variable directly into your environment config file.
LEGGED_GYM_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    # Top-level settings from legged_gym config
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False, # Must be False for curriculum to work
    # The dictionary of sub-terrains to generate
    sub_terrains={
        "lg_sloped_plain": LgSlopedTerrainCfg(
            proportion=0.1,
            add_roughness=False,
            func=f"{__name__}.lg_sloped_terrain_generator",
        ),
        "lg_sloped_rough": LgSlopedTerrainCfg(
            proportion=0.1,
            add_roughness=True,
            func=f"{__name__}.lg_sloped_terrain_generator",
        ),
        "lg_stairs": LgPyramidStairsTerrainCfg(
            proportion=0.2,
            func=f"{__name__}.lg_pyramid_stairs_terrain_generator",
        ),
        "lg_boxes": LgDiscreteObstaclesTerrainCfg(
            proportion=0.2,
            func=f"{__name__}.lg_discrete_obstacles_terrain_generator",
        ),
        "lg_stepping_stones": LgSteppingStonesTerrainCfg(
            proportion=0.2,
            func=f"{__name__}.lg_stepping_stones_terrain_generator",
        ),
        "lg_gap": LgGapPitTerrainCfg(
            proportion=0.1,
            is_pit=False,
            func=f"{__name__}.lg_gap_pit_terrain_generator",
        ),
        "lg_pit": LgGapPitTerrainCfg(
            proportion=0.1,
            is_pit=True,
            func=f"{__name__}.lg_gap_pit_terrain_generator",
        ),
    },
)