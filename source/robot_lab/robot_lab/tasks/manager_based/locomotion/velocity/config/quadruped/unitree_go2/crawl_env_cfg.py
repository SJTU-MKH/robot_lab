# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm

from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

##
# Pre-defined configs
##
# # use cloud assets
# from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
# use local assets
from robot_lab.assets.unitree import UNITREE_GO2_CFG  # isort: skip

# 导入爬行地形配置和自定义导入器
from .crawl_terrain_cfg import CRAWL_TERRAINS_CFG  # isort: skip
from robot_lab.tasks.manager_based.locomotion.velocity.utils.terrains_asset.mesh_crawl import (
    CrawlTerrainImporter,
)

# 导入导航MDP函数和命令生成器
from robot_lab.tasks.manager_based.locomotion.velocity import mdp


@configclass
class UnitreeGo2CrawlEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"
    # fmt: off
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Scene------------------------------
        # 使用自定义的地形导入器和爬行地形
        self.scene.terrain = TerrainImporterCfg(
            class_type=CrawlTerrainImporter,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=CRAWL_TERRAINS_CFG,
            max_init_terrain_level=5,
            collision_group=-1,
            physics_material=self.scene.terrain.physics_material,
            visual_material=self.scene.terrain.visual_material,
            debug_vis=False,
        )

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = (
            "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        )
        self.scene.height_scanner_base.prim_path = (
            "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        )

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 1.5  # 降低速度scale，适应爬行
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        # 保留高度扫描，对爬行很重要
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = (
            self.joint_names
        )
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = (
            self.joint_names
        )

        # ------------------------------导航观察------------------------------
        # 添加导航相关的观察
        self.observations.policy.target_direction = ObsTerm(
            func=mdp.target_direction,
            scale=1.0,
        )
        self.observations.policy.target_distance = ObsTerm(
            func=mdp.target_distance,
            scale=0.1,  # 归一化距离
        )
        self.observations.policy.target_yaw_relative = ObsTerm(
            func=mdp.target_yaw_relative,
            scale=0.3,  # 归一化角度
        )

        # ------------------------------Actions------------------------------
        # 适应爬行的动作scale
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.15,
            ".*_thigh_joint": 0.3,
            ".*_calf_joint": 0.25,
        }
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        # 导航系统初始化（startup时调用一次）
        self.events.initialize_navigation = EventTerm(
            func=mdp.initialize_navigation_system,
            mode="startup",
        )

        # 更新导航目标（每个step调用）
        self.events.update_navigation = EventTerm(
            func=mdp.update_navigation_targets,
            mode="interval",
            interval_range_s=(0.0, 0.0),  # 每个step都调用
        )

        # 重置时重置导航状态
        self.events.reset_navigation = EventTerm(
            func=mdp.reset_navigation_on_termination,
            mode="reset",
        )

        # 将机器人重置到第一个导航点（起点）位置
        self.events.reset_robot_position = EventTerm(
            func=mdp.reset_robot_to_start_position,
            mode="reset",
        )

        # 根据导航表现调整地形难度（课程学习）
        self.events.adjust_difficulty = EventTerm(
            func=mdp.adjust_terrain_difficulty,
            mode="reset",
        )

        # 移除原来的随机位置重置（使用Parkour风格的固定起点重置）
        self.events.randomize_reset_base = None
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        self.events.randomize_apply_external_force_torque.params[
            "asset_cfg"
        ].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties - 适应爬行的奖励
        self.rewards.lin_vel_z_l2.weight = -1.0  # 减少垂直速度惩罚
        self.rewards.ang_vel_xy_l2.weight = -0.02  # 减少角速度惩罚
        self.rewards.flat_orientation_l2.weight = -0.5  # 允许一定程度的倾斜

        # 降低目标高度，适应爬行姿态
        self.rewards.base_height_l2.weight = -1.0
        self.rewards.base_height_l2.params["target_height"] = 0.20  # 降低目标高度
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [
            self.base_link_name
        ]

        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [
            self.base_link_name
        ]

        # Joint penalties - 适应爬行的关节惩罚
        self.rewards.joint_torques_l2.weight = -1.5e-5  # 减少力矩惩罚
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -1.5e-7
        self.rewards.joint_pos_limits.weight = -3.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -1e-5
        self.rewards.stand_still.weight = -1.0  # 减少站立惩罚
        self.rewards.joint_pos_penalty.weight = -0.5
        self.rewards.joint_mirror.weight = -0.02
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.005  # 减少动作变化惩罚

        # Contact sensor - 爬行时身体接触地面是正常的
        self.rewards.undesired_contacts.weight = -0.5  # 减少不期望接触的惩罚
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            f"^(?!.*{self.foot_link_name}).*"
        ]
        self.rewards.contact_forces.weight = -0.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]

        # Velocity-tracking rewards - 适应爬行的速度奖励
        self.rewards.track_lin_vel_xy_exp.weight = 2.0  # 减少速度跟踪权重
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        # Feet rewards - 适应爬行步态
        self.rewards.feet_air_time.weight = 0.05  # 减少腾空时间奖励
        self.rewards.feet_air_time.params["threshold"] = 0.3  # 降低腾空时间阈值
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_air_time_variance.weight = -0.5
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_contact.weight = 0.1  # 增加足部接触奖励
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_contact_without_cmd.weight = 0.05
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_stumble.weight = -0.1  # 增加绊倒惩罚
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_slide.weight = -0.05
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = -0.1  # 不要求太高的足部高度
        self.rewards.feet_height.params["target_height"] = 0.02
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -2.0  # 减少足部相对body高度惩罚
        self.rewards.feet_height_body.params["target_height"] = (
            -0.15
        )  # 允许更低的相对高度
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_gait.weight = 0.2  # 减少步态奖励，允许更灵活的步态
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (
            ("FL_foot", "RR_foot"),
            ("FR_foot", "RL_foot"),
        )
        self.rewards.upward.weight = 0.5  # 减少向上奖励

        # ------------------------------导航奖励------------------------------
        # 添加导航相关的奖励
        self.rewards.heading_to_target = RewTerm(
            func=mdp.heading_to_target,
            weight=1.0,  # 鼓励朝向目标
        )
        self.rewards.progress_to_target = RewTerm(
            func=mdp.progress_to_target,
            weight=0.5,  # 鼓励接近目标
        )
        self.rewards.reached_target = RewTerm(
            func=mdp.reached_target,
            weight=10.0,  # 到达目标给予大奖励
            params={"threshold": 0.5},
        )
        self.rewards.velocity_to_target = RewTerm(
            func=mdp.velocity_to_target,
            weight=0.8,  # 鼓励朝目标方向移动
        )
        self.rewards.next_goal_awareness = RewTerm(
            func=mdp.next_goal_awareness,
            weight=0.3,  # 鼓励考虑下一个目标
        )

        # 添加爬行特定的奖励：鼓励低姿态前进
        # 这个可以通过现有的base_height和velocity tracking实现

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2CrawlEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # 爬行时允许更多的身体接触
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        # 启用地形难度课程学习
        # self.curriculum.terrain_levels 保持启用（从父类继承）
        self.curriculum.command_levels = None

        # ------------------------------Commands------------------------------
        # 使用导航命令生成器，根据导航点自动生成速度命令
        self.commands.base_velocity = mdp.NavigationCommandCfg(
            asset_name="robot",
            lin_vel_x_range=(0.3, 0.8),  # 爬行的前进速度范围
            ang_vel_z_range=(-0.8, 0.8),  # 允许的最大转向速度
            heading_control_stiffness=0.5,  # 朝向控制刚度
            resampling_time_range=(10.0, 10.0),  # 命令重采样时间
            debug_vis=True,  # 启用可视化：绿色=目标速度，蓝色=实际速度
        )
