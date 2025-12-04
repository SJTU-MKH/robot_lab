from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from collections.abc import Callable
from isaaclab.utils import configclass

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from .boxgarden_terrain import boxgarden_generator


@configclass
class BoxGardenTerrainCfg(SubTerrainBaseCfg):

    function: Callable = boxgarden_generator
    wall_height: float = 1.0
    # stone_height_range: tuple[float, float] = (0.05, 0.4)


BOXGARDEN_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),

    border_width=0.5,

    num_rows=1,
    num_cols=1,

    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,

    sub_terrains={
        "box_garden": BoxGardenTerrainCfg(
            proportion=1.0,
            wall_height=1.0,
        ),
    },
)
