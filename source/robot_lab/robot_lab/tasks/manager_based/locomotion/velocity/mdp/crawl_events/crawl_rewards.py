# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""MDP奖励函数 - 用于爬行导航"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def heading_to_target(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    朝向目标的奖励 - 鼓励机器人朝向当前目标点

    Returns:
        reward: (num_envs,) - 范围 [-1, 1]，朝向目标时为正
    """
    if not hasattr(env, "navigation_event"):
        return torch.zeros(env.num_envs, device=env.device)

    robot: Articulation = env.scene[asset_cfg.name]
    
    # 从四元数获取yaw角，计算朝向向量
    _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    robot_heading = torch.stack([torch.cos(robot_yaw), torch.sin(robot_yaw)], dim=-1)

    # 归一化朝向向量
    robot_heading_norm = robot_heading / (
        torch.norm(robot_heading, dim=-1, keepdim=True) + 1e-5
    )

    # 计算朝向与目标方向的对齐度（点积）
    alignment = torch.sum(
        robot_heading_norm * env.navigation_event.target_dir_norm, dim=-1
    )

    return alignment


def progress_to_target(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    接近目标的奖励 - 鼓励机器人减小与目标的距离

    Returns:
        reward: (num_envs,)
    """
    if not hasattr(env, "navigation_event"):
        return torch.zeros(env.num_envs, device=env.device)

    # 使用距离的负值作为奖励（距离越小奖励越大）
    return -env.navigation_event.target_distance * 0.1


def reached_target(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    到达目标的奖励 - 到达目标点时给予大奖励

    Args:
        threshold: 到达的距离阈值

    Returns:
        reward: (num_envs,)
    """
    if not hasattr(env, "navigation_event"):
        return torch.zeros(env.num_envs, device=env.device)

    # 检查是否到达目标
    reached = env.navigation_event.target_distance < threshold

    return reached.float() * 10.0


def velocity_to_target(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    朝向目标移动的速度奖励 - 鼓励机器人朝目标方向移动

    Returns:
        reward: (num_envs,)
    """
    if not hasattr(env, "navigation_event"):
        return torch.zeros(env.num_envs, device=env.device)

    robot: Articulation = env.scene[asset_cfg.name]

    # 获取机器人在xy平面的速度（世界坐标系）
    robot_vel_w = robot.data.root_lin_vel_w[:, :2]

    # 计算速度与目标方向的对齐度
    vel_norm = torch.norm(robot_vel_w, dim=-1, keepdim=True) + 1e-5
    vel_direction = robot_vel_w / vel_norm

    # 点积计算对齐度
    alignment = torch.sum(vel_direction * env.navigation_event.target_dir_norm, dim=-1)

    # 速度大小加权
    speed = vel_norm.squeeze(-1)

    return alignment * torch.clamp(speed, 0, 1.0)


def next_goal_awareness(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    下一个目标感知奖励 - 鼓励机器人考虑下一个目标点的方向

    当接近当前目标时，开始朝向下一个目标

    Returns:
        reward: (num_envs,)
    """
    if not hasattr(env, "navigation_event"):
        return torch.zeros(env.num_envs, device=env.device)

    robot: Articulation = env.scene[asset_cfg.name]
    
    # 从四元数获取yaw角，计算朝向向量
    _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    robot_heading = torch.stack([torch.cos(robot_yaw), torch.sin(robot_yaw)], dim=-1)
    
    robot_heading_norm = robot_heading / (
        torch.norm(robot_heading, dim=-1, keepdim=True) + 1e-5
    )

    # 计算朝向与下一个目标方向的对齐度
    alignment = torch.sum(
        robot_heading_norm * env.navigation_event.next_target_dir_norm, dim=-1
    )

    # 只在接近当前目标时才给予奖励
    near_current = env.navigation_event.target_distance < 1.0

    return alignment * near_current.float() * 0.5


def goal_progress_rate(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    目标进度奖励 - 根据完成的导航点数量给予奖励

    Returns:
        reward: (num_envs,)
    """
    if not hasattr(env, "navigation_event"):
        return torch.zeros(env.num_envs, device=env.device)

    # 完成的目标数量
    completed_goals = env.navigation_event.cur_goal_idx.float()
    total_goals = float(env.navigation_event.num_goals)

    # 归一化进度
    progress = completed_goals / max(total_goals, 1.0)

    return progress * 2.0
