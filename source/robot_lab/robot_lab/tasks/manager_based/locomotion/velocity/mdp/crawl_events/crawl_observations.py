# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""MDP观察函数 - 用于爬行导航"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_direction(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    到当前目标点的归一化方向向量(xy平面)

    Returns:
        target_direction: (num_envs, 2) - 归一化的xy方向
    """
    if not hasattr(env, "navigation_event"):
        return torch.zeros(env.num_envs, 2, device=env.device)

    return env.navigation_event.target_dir_norm


def target_distance(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    到当前目标点的距离

    Returns:
        distance: (num_envs, 1)
    """
    if not hasattr(env, "navigation_event"):
        return torch.zeros(env.num_envs, 1, device=env.device)

    return env.navigation_event.target_distance.unsqueeze(-1)


def target_yaw_relative(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    相对于机器人当前朝向的目标yaw角

    Returns:
        yaw: (num_envs, 1) - 范围 [-pi, pi]
    """
    if not hasattr(env, "navigation_event"):
        return torch.zeros(env.num_envs, 1, device=env.device)

    robot: Articulation = env.scene[asset_cfg.name]
    
    # 从四元数获取机器人yaw角
    _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    robot_yaw = wrap_to_pi(robot_yaw)

    target_yaw_rel = env.navigation_event.target_yaw - robot_yaw
    # 归一化到 [-pi, pi]
    target_yaw_rel = wrap_to_pi(target_yaw_rel)

    return target_yaw_rel.unsqueeze(-1)


def current_goal_index(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    当前目标点的索引(归一化到[0, 1])

    Returns:
        goal_idx: (num_envs, 1)
    """
    if not hasattr(env, "navigation_event"):
        return torch.zeros(env.num_envs, 1, device=env.device)

    # 归一化索引
    max_idx = env.navigation_event.num_goals - 1
    normalized_idx = env.navigation_event.cur_goal_idx.float() / max(max_idx, 1)

    return normalized_idx.unsqueeze(-1)


def next_target_yaw_relative(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    相对于机器人当前朝向的下一个目标yaw角

    Returns:
        yaw: (num_envs, 1) - 范围 [-pi, pi]
    """
    if not hasattr(env, "navigation_event"):
        return torch.zeros(env.num_envs, 1, device=env.device)

    robot: Articulation = env.scene[asset_cfg.name]
    
    # 从四元数获取机器人yaw角
    _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    robot_yaw = wrap_to_pi(robot_yaw)

    next_target_yaw_rel = env.navigation_event.next_target_yaw - robot_yaw
    # 归一化到 [-pi, pi]
    next_target_yaw_rel = wrap_to_pi(next_target_yaw_rel)

    return next_target_yaw_rel.unsqueeze(-1)
