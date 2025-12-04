# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Reset event functions for crawl navigation."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_robot_to_start_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """
    重置机器人到爬行地形的起点导航点位置

    直接将机器人放在第一个导航点（起点）位置，不需要额外偏移
    因为地形生成器已经定义了5个导航点：起点、3个隧道、终点

    Args:
        env: 环境实例
        env_ids: 需要重置的环境ID
        asset_cfg: 机器人资产配置
    """
    if not hasattr(env, "navigation_event"):
        return

    asset: Articulation = env.scene[asset_cfg.name]
    nav_event = env.navigation_event

    # 获取默认状态
    root_states = asset.data.default_root_state[env_ids].clone()

    # 获取环境原点
    origin = env.scene.env_origins[env_ids].clone()

    # 获取起点导航点位置（第0个导航点）
    # nav_event.env_goals shape: (num_envs, num_goals, 3)
    start_goals = nav_event.env_goals[env_ids, 0, :]  # (len(env_ids), 3)

    # 将导航点从地形坐标系转换到世界坐标系
    start_positions_world = start_goals + origin

    # 设置z高度为爬行高度（保持导航点的z值）
    positions = torch.cat(
        [
            start_positions_world[:, :2],  # xy使用导航点位置
            start_positions_world[:, 2:3],  # z使用导航点高度
        ],
        dim=-1,
    )

    # 写入位置和姿态（保持默认姿态）
    asset.write_root_pose_to_sim(
        torch.cat([positions, root_states[:, 3:7]], dim=-1), env_ids=env_ids
    )

    # 写入速度（初始速度设为0）
    asset.write_root_velocity_to_sim(root_states[:, 7:13], env_ids=env_ids)


def adjust_terrain_difficulty(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """
    根据导航表现调整地形难度

    参考Parkour的_resample_command逻辑：
    - 如果机器人到达了终点（距离起点很远），升级地形难度
    - 如果机器人没走多远就失败了，降低地形难度

    Args:
        env: 环境实例
        env_ids: 需要重置的环境ID
        asset_cfg: 机器人资产配置
    """
    if not hasattr(env, "navigation_event"):
        return

    asset: Articulation = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    nav_event = env.navigation_event
    terrain_gen_cfg = terrain.cfg.terrain_generator

    # 计算起点位置
    start_pos = env.scene.env_origins[env_ids, :2] - torch.tensor(
        (
            terrain_gen_cfg.size[1] + 0.5,
            0,
        )  # 与reset_robot_to_start_position的offset一致
    ).to(env.device)

    # 计算机器人离起点的距离
    robot_pos = asset.data.root_pos_w[env_ids, :2]
    dis_to_start = torch.norm(start_pos - robot_pos, dim=1)

    # 计算期望的前进距离（基于episode长度和期望速度）
    # 假设期望速度为0.5 m/s，episode长度20秒，期望前进10米
    expected_distance = 10.0

    # 判断是否应该升级/降级
    # 如果前进超过80%期望距离，升级难度
    move_up = dis_to_start > 0.8 * expected_distance
    # 如果前进不到40%期望距离，降低难度
    move_down = dis_to_start < 0.4 * expected_distance

    # 调整地形级别
    terrain.terrain_levels[env_ids] += 1 * move_up - 1 * move_down

    # 限制在有效范围内
    terrain.terrain_levels[env_ids] = torch.clamp(
        terrain.terrain_levels[env_ids], min=0, max=terrain.max_terrain_level - 1
    )

    # 完成最高难度的机器人随机分配到一个难度
    max_level_envs = terrain.terrain_levels[env_ids] >= terrain.max_terrain_level
    if max_level_envs.any():
        max_level_env_ids = env_ids[max_level_envs]
        terrain.terrain_levels[max_level_env_ids] = torch.randint(
            0, terrain.max_terrain_level, (max_level_envs.sum(),), device=env.device
        )

    # 更新环境原点（根据新的地形级别）
    env.scene.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]
