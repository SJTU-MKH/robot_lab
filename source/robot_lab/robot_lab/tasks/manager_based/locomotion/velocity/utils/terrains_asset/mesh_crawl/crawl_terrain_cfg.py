from collections.abc import Callable

from isaaclab.utils import configclass
from isaaclab.terrains.terrain_generator_cfg import (
    SubTerrainBaseCfg,
    TerrainGeneratorCfg,
)

from robot_lab.tasks.manager_based.locomotion.velocity.utils.terrains_asset.mesh_crawl.crawl_terrain import (
    three_tunnel_chamber_generator,
)


@configclass
class ThreeTunnelChamberTerrainCfg(SubTerrainBaseCfg):
    """三隧道房间地形配置，四周围墙内有三个位置不对齐的隧道。"""

    function: Callable = three_tunnel_chamber_generator

    # 围墙参数
    wall_height: float = 2.0
    wall_thickness: float = 1.0

    # 隧道参数范围（根据难度调整）
    tunnel_wall_thickness_range: tuple[float, float] = (0.2, 0.5)
    tunnel_height_range: tuple[float, float] = (0.25, 0.5)
    tunnel_width_range: tuple[float, float] = (0.5, 0.8)


@configclass
class CrawlTerrainGeneratorCfg(TerrainGeneratorCfg):
    """爬行地形生成器配置，支持导航点系统。"""

    # 导航点数量（起点 + 3个隧道 + 终点）
    num_goals: int = 5
