from dataclasses import MISSING
from collections.abc import Callable

from isaaclab.utils import configclass
from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from .crawl_terrain import low_obstacles_generator, tunnel_generator


@configclass
class LowObstaclesTerrainCfg(SubTerrainBaseCfg):
    """低矮障碍物地形配置 - 需要机器人降低身体高度爬行通过。"""

    # 关键: 将生成函数赋值给 'function' 属性
    function: Callable = low_obstacles_generator

    # 自定义参数
    # 障碍物高度范围 (米)
    obstacle_height_range: tuple[float, float] = (0.15, 0.35)

    # 障碍物间距 (米)
    obstacle_spacing: float = 1.5

    # 障碍物宽度范围 (米)
    obstacle_width_range: tuple[float, float] = (0.8, 2.0)


@configclass
class TunnelTerrainCfg(SubTerrainBaseCfg):
    """隧道地形配置 - 需要机器人爬行通过狭窄空间。"""

    function: Callable = tunnel_generator

    # 隧道高度 (米)
    tunnel_height: float = 0.4

    # 隧道宽度 (米)
    tunnel_width: float = 1.5
