# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# 观测，奖励，终止
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
# 场景
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils import configclass

from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.parkour.mdp as mdp


@configclass
class UnitreeGo2WObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        # TODO 1: observation
        """Observations for policy group."""

        # 132维 TODO 后处理
        height_scan = ObsTerm(
            func=mdp.relative_height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "base_height_offset": 0.3, # 这就是那个 0.3 的偏移量
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        # 3维 0.25
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )

        # 确认quaternions_roll_pitch是否正确(机器人在世界坐标系下的横滚和俯仰角)
        # 2维
        quaternions_roll_pitch = ObsTerm(
            func=mdp.base_euler_xy,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        # 4维 现在还不是 TODO：[0, 0, v_x_cmd, v_y_cmd]
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=0.05,
        )

        # 12维 关节偏移量
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 12维 关节速度 0.05
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )

        # 12维 历史action
        history_action = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 4维 足部触地flag
        foot_contact = ObsTerm(
            func=mdp.foot_contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-0.5, 0.5),
            scale=1.0,
        )

        # priv explicit,可以留一些占位符号
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=2.0,
        )

        ## priv latent
        # observation terms (order preserved)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # 132维 TODO 后处理
        height_scan = ObsTerm(
            func=mdp.relative_height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "base_height_offset": 0.3, # 这就是那个 0.3 的偏移量
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        # 3维 0.25
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )

        # 2维
        quaternions_roll_pitch = ObsTerm(
            func=mdp.base_euler_xy,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=0.05,
        )

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )

        history_action = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        foot_contact = ObsTerm(
            func=mdp.foot_contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-0.5, 0.5),
            scale=1.0,
        )

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=2.0,
        )

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
