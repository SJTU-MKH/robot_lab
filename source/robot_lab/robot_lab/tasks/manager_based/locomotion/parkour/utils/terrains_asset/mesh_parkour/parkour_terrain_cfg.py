from dataclasses import MISSING
from collections.abc import Callable

from isaaclab.utils import configclass

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from robot_lab.tasks.manager_based.locomotion.parkour.utils.terrains_asset.mesh_parkour.parkour_terrain import stepping_stones_generator


@configclass
class SteppingStonesTerrainCfg(SubTerrainBaseCfg):
    """汀步石地形的配置。"""

    # 关键一步: 将我们的生成函数赋值给 'function' 属性
    function: Callable = stepping_stones_generator
    
    # -- 自定义参数，这些参数可以在生成函数中通过 cfg.xxx 访问
    
    # 石墩的半径范围 (米)
    stone_radius: float = 0.2
    
    # 石墩的高度范围 [min, max] (米)
    stone_height_range: tuple[float, float] = (0.05, 0.4)
    
    # 生成石墩的网格间距 (米)
    grid_resolution: float = 1.0