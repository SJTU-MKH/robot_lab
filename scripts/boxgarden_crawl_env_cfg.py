# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""BoxGarden navigation environment compatible with crawl height checkpoint."""

from isaaclab.utils import configclass
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go2.crawl_height_env_cfg import (
    UnitreeGo2CrawlHeightEnvCfg,
)
from robot_lab.tasks.manager_based.navigation.waypoint.boxgarden_terrain_cfg import (
    BOXGARDEN_TERRAIN_CFG,
)


@configclass
class BoxGardenCrawlCompatEnvCfg(UnitreeGo2CrawlHeightEnvCfg):
    """BoxGarden environment configuration compatible with crawl height checkpoint"""

    def __post_init__(self):
        # 调用父类初始化（这会设置爬行高度的观察空间）
        super().__post_init__()

        # 使用BoxGarden地形替换平地
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BOXGARDEN_TERRAIN_CFG

        # 确保保持与爬行高度checkpoint兼容的观察空间
        # 父类已经禁用了height_scanner，所以观察空间应该是47维

        print(f"[INFO] BoxGarden地形已配置，保持与爬行高度checkpoint兼容的观察空间")


@configclass
class BoxGardenCrawlCompatEnvCfg_PLAY(BoxGardenCrawlCompatEnvCfg):
    """Play模式配置 - 固定高度命令，使用BoxGarden地形"""

    def __post_init__(self):
        super().__post_init__()

        # 继承爬行高度PLAY模式的设置
        # 禁用随机化
        self.events.randomize_reset_base = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_joint_parameters = None
        self.events.randomize_apply_external_force_torque = None

        # 固定目标高度为0.25米
        if hasattr(self.commands, "base_height"):
            self.commands.base_height.ranges.heading = (0.25, 0.25)

        # 较慢的速度命令
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
