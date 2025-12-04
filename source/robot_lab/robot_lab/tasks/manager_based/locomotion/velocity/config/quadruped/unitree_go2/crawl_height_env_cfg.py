# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Crawl height control environment configuration for Unitree Go2."""

import torch
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm

from .rough_env_cfg import UnitreeGo2RoughEnvCfg

# 使用本地资源
from robot_lab.assets.unitree import UNITREE_GO2_CFG  # isort: skip

# 导入MDP函数
from robot_lab.tasks.manager_based.locomotion.velocity import mdp


@configclass
class UnitreeGo2CrawlHeightEnvCfg(UnitreeGo2RoughEnvCfg):
    """四足机器人指定高度爬行环境配置（基于flat地形）"""

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
        # 调用父类初始化
        super().__post_init__()

        # ------------------------------Scene------------------------------
        # 修改地形为flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # 禁用高度扫描
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        if hasattr(self.observations, "critic") and hasattr(
            self.observations.critic, "height_scan"
        ):
            self.observations.critic.height_scan = None
        # 禁用地形课程
        self.curriculum.terrain_levels = None

        # 使用Go2配置
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 修改默认关节角度以适应爬行姿态
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.20)  # 降低初始高度，适合爬行
        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 1.2,  # 前腿大腿关节角度增大（更弯曲）
            "R.*_thigh_joint": 1.2,  # 后腿大腿关节角度增大（更弯曲）
            ".*_calf_joint": -2.0,  # 小腿关节角度增大（更弯曲）
        }

        # ------------------------------Observations------------------------------
        # 添加目标高度观察（从base_height命令中获取）
        self.observations.policy.target_height = ObsTerm(
            func=mdp.base_height_command,
            params={"command_name": "base_height"},
            scale=1.0,
        )

        # 添加当前高度观察（相对于地面）
        self.observations.policy.current_height = ObsTerm(
            func=mdp.base_height_above_terrain,
            params={"asset_cfg": mdp.SceneEntityCfg("robot")},
            scale=1.0,
        )

        # 保留其他重要观察（如果存在）
        if self.observations.policy.base_lin_vel is not None:
            self.observations.policy.base_lin_vel.scale = 1.5
        if self.observations.policy.base_ang_vel is not None:
            self.observations.policy.base_ang_vel.scale = 0.25
        if self.observations.policy.joint_pos is not None:
            self.observations.policy.joint_pos.scale = 1.0
            self.observations.policy.joint_pos.params["asset_cfg"].joint_names = (
                self.joint_names
            )
        if self.observations.policy.joint_vel is not None:
            self.observations.policy.joint_vel.scale = 0.05
            self.observations.policy.joint_vel.params["asset_cfg"].joint_names = (
                self.joint_names
            )

        # ------------------------------Actions------------------------------
        # 爬行动作配置
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.15,
            ".*_thigh_joint": 0.3,
            ".*_calf_joint": 0.25,
        }
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        # 爬行时的初始化位置调整
        if self.events.randomize_reset_base is not None:
            self.events.randomize_reset_base.params = {
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.15),  # 低高度初始化
                    "roll": (-0.3, 0.3),
                    "pitch": (-0.3, 0.3),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {
                    "x": (-0.3, 0.3),
                    "y": (-0.3, 0.3),
                    "z": (-0.1, 0.1),
                    "roll": (-0.3, 0.3),
                    "pitch": (-0.3, 0.3),
                    "yaw": (-0.3, 0.3),
                },
            }

        # 配置其他事件的body_names
        if self.events.randomize_rigid_body_mass_base is not None:
            self.events.randomize_rigid_body_mass_base.params[
                "asset_cfg"
            ].body_names = [self.base_link_name]
        if self.events.randomize_rigid_body_mass_others is not None:
            self.events.randomize_rigid_body_mass_others.params[
                "asset_cfg"
            ].body_names = [f"^(?!.*{self.base_link_name}).*"]
        if self.events.randomize_com_positions is not None:
            self.events.randomize_com_positions.params["asset_cfg"].body_names = [
                self.base_link_name
            ]
        if self.events.randomize_apply_external_force_torque is not None:
            self.events.randomize_apply_external_force_torque.params[
                "asset_cfg"
            ].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # 高度控制奖励 - 使用指数核跟踪命令高度
        self.rewards.track_base_height = RewTerm(
            func=mdp.track_base_height_exp,
            weight=5.0,
            params={
                "std": 0.05,
                "command_name": "base_height",
            },
        )

        # 高度惩罚（与命令高度的L2距离）
        if self.rewards.base_height_l2 is not None:
            self.rewards.base_height_l2.weight = -2.0
            self.rewards.base_height_l2.params["target_height"] = 0.0
            self.rewards.base_height_l2.params["sensor_cfg"] = (
                None  # flat地形没有height_scanner_base
            )
            self.rewards.base_height_l2.params["asset_cfg"].body_names = [
                self.base_link_name
            ]

        # 姿态稳定性
        if self.rewards.flat_orientation_l2 is not None:
            self.rewards.flat_orientation_l2.weight = -1.0
        if self.rewards.ang_vel_xy_l2 is not None:
            self.rewards.ang_vel_xy_l2.weight = -0.05

        # 速度跟踪（降低权重，高度控制优先）
        if self.rewards.track_lin_vel_xy_exp is not None:
            self.rewards.track_lin_vel_xy_exp.weight = 1.0
        if self.rewards.track_ang_vel_z_exp is not None:
            self.rewards.track_ang_vel_z_exp.weight = 0.5

        # 能量消耗
        if self.rewards.joint_torques_l2 is not None:
            self.rewards.joint_torques_l2.weight = -1.5e-5
        if self.rewards.joint_acc_l2 is not None:
            self.rewards.joint_acc_l2.weight = -2.5e-7
        if self.rewards.action_rate_l2 is not None:
            self.rewards.action_rate_l2.weight = -0.01
        if self.rewards.joint_power is not None:
            self.rewards.joint_power.weight = -2e-5

        # 接触惩罚（允许低姿态时body接触）
        if self.rewards.undesired_contacts is not None:
            self.rewards.undesired_contacts.weight = -0.5
            self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
                f"^(?!.*{self.foot_link_name}).*"
            ]
        if self.rewards.contact_forces is not None:
            self.rewards.contact_forces.weight = -1e-4
            self.rewards.contact_forces.params["sensor_cfg"].body_names = [
                self.foot_link_name
            ]

        # 足部奖励（适应低姿态）
        if self.rewards.feet_air_time is not None:
            self.rewards.feet_air_time.weight = 0.1
            self.rewards.feet_air_time.params["threshold"] = 0.3
            self.rewards.feet_air_time.params["sensor_cfg"].body_names = [
                self.foot_link_name
            ]

        if self.rewards.feet_contact is not None:
            self.rewards.feet_contact.weight = 0.2
            self.rewards.feet_contact.params["sensor_cfg"].body_names = [
                self.foot_link_name
            ]

        if self.rewards.feet_stumble is not None:
            self.rewards.feet_stumble.weight = -0.2
            self.rewards.feet_stumble.params["sensor_cfg"].body_names = [
                self.foot_link_name
            ]

        if self.rewards.feet_slide is not None:
            self.rewards.feet_slide.weight = -0.1
            self.rewards.feet_slide.params["sensor_cfg"].body_names = [
                self.foot_link_name
            ]
            self.rewards.feet_slide.params["asset_cfg"].body_names = [
                self.foot_link_name
            ]

        # 足部高度控制
        if self.rewards.feet_height_body is not None:
            self.rewards.feet_height_body.weight = -3.0
            self.rewards.feet_height_body.params["target_height"] = -0.12
            self.rewards.feet_height_body.params["asset_cfg"].body_names = [
                self.foot_link_name
            ]

        # 步态
        if self.rewards.feet_gait is not None:
            self.rewards.feet_gait.weight = 0.3
            self.rewards.feet_gait.params["synced_feet_pair_names"] = (
                ("FL_foot", "RR_foot"),
                ("FR_foot", "RL_foot"),
            )

        # 关节限制
        if self.rewards.joint_pos_limits is not None:
            self.rewards.joint_pos_limits.weight = -5.0
        if self.rewards.joint_vel_limits is not None:
            self.rewards.joint_vel_limits.weight = 0

        # 移除不需要的奖励
        if self.rewards.is_terminated is not None:
            self.rewards.is_terminated.weight = 0
        if self.rewards.lin_vel_z_l2 is not None:
            self.rewards.lin_vel_z_l2.weight = -0.5
        if self.rewards.body_lin_acc_l2 is not None:
            self.rewards.body_lin_acc_l2.weight = 0
        if self.rewards.joint_vel_l2 is not None:
            self.rewards.joint_vel_l2.weight = 0
        if self.rewards.stand_still is not None:
            self.rewards.stand_still.weight = 0
        if self.rewards.joint_pos_penalty is not None:
            self.rewards.joint_pos_penalty.weight = -0.3
        if self.rewards.joint_mirror is not None:
            self.rewards.joint_mirror.weight = -0.01
            self.rewards.joint_mirror.params["mirror_joints"] = [
                ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
                ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
            ]
        if self.rewards.feet_air_time_variance is not None:
            self.rewards.feet_air_time_variance.weight = -0.3
            self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [
                self.foot_link_name
            ]
        if self.rewards.feet_contact_without_cmd is not None:
            self.rewards.feet_contact_without_cmd.weight = 0
        if self.rewards.feet_height is not None:
            self.rewards.feet_height.weight = 0
        if self.rewards.upward is not None:
            self.rewards.upward.weight = 0

        # 禁用零权重奖励
        if self.__class__.__name__ == "UnitreeGo2CrawlHeightEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # 允许body接触
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        self.curriculum.terrain_levels = None  # 暂时禁用地形难度课程
        self.curriculum.command_levels = None

        # ------------------------------Commands------------------------------
        # 添加base_height命令
        self.commands.base_height = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.0),  # 不使用
                lin_vel_y=(0.0, 0.0),  # 不使用
                ang_vel_z=(0.0, 0.0),  # 不使用
                heading=(0.15, 0.35),  # 使用heading范围作为目标高度范围（0.15-0.35米）
            ),
        )

        # 保留velocity命令用于移动控制
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class UnitreeGo2CrawlHeightEnvCfg_PLAY(UnitreeGo2CrawlHeightEnvCfg):
    """Play模式配置 - 固定高度命令"""

    def __post_init__(self):
        super().__post_init__()

        # 禁用随机化
        self.events.randomize_reset_base = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_joint_parameters = None
        self.events.randomize_apply_external_force_torque = None

        # 固定目标高度为0.25米
        self.commands.base_height.ranges.heading = (0.25, 0.25)

        # 较慢的速度命令
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
