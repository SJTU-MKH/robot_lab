# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Crawl navigation event manager - handles waypoint navigation logic."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

from robot_lab.tasks.manager_based.locomotion.velocity.utils.terrains_asset.mesh_crawl import (
    CrawlTerrainImporter,
    CrawlTerrainGenerator,
    CrawlTerrainGeneratorCfg,
)

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


class CrawlNavigationEvent:
    """
    管理爬行环境中的导航点系统

    功能:
    - 追踪每个环境的当前目标导航点
    - 计算到目标点的方向和距离
    - 处理目标点切换逻辑
    - 提供导航相关的观察数据
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        next_goal_threshold: float = 0.5,
        reach_goal_delay: float = 0.5,
    ):
        """
        初始化导航事件管理器

        Args:
            env: 环境实例
            asset_cfg: 机器人资产配置
            next_goal_threshold: 到达目标点的距离阈值(米)
            reach_goal_delay: 到达目标后等待多久再切换下一个(秒)
        """
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.dt = env.step_dt

        # 获取机器人
        self.robot: Articulation = env.scene[asset_cfg.name]

        # 导航参数
        self.next_goal_threshold = next_goal_threshold
        self.reach_goal_delay = reach_goal_delay

        # 获取地形生成器和导航点信息
        self.terrain: CrawlTerrainImporter = env.scene.terrain
        terrain_generator: CrawlTerrainGenerator = self.terrain.terrain_generator_class
        terrain_cfg: CrawlTerrainGeneratorCfg = self.terrain.cfg.terrain_generator

        self.num_goals = terrain_cfg.num_goals
        self.env_origins = self.terrain.env_origins

        # 将导航点转换为torch tensor
        terrain_goals = terrain_generator.goals  # (num_rows, num_cols, num_goals, 3)
        self.terrain_goals = torch.from_numpy(terrain_goals).to(self.device).float()

        # 为每个环境存储导航点
        self.env_goals = torch.zeros(
            self.num_envs, self.num_goals, 3, device=self.device, requires_grad=False
        )

        # 根据每个环境的地形位置获取对应的导航点
        terrain_levels = self.terrain.terrain_levels
        terrain_types = self.terrain.terrain_types
        self.env_goals[:] = self.terrain_goals[terrain_levels, terrain_types]

        # 当前目标索引 (每个环境从0开始)
        self.cur_goal_idx = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        # 到达目标的计时器
        self.reach_goal_timer = torch.zeros(self.num_envs, device=self.device)

        # 当前目标和下一个目标
        self.cur_goals = self._get_goals_at_idx(self.cur_goal_idx)
        self.next_goals = self._get_goals_at_idx(self.cur_goal_idx + 1)

        # 计算初始方向
        self._update_navigation_data()

    def _get_goals_at_idx(self, indices: torch.Tensor) -> torch.Tensor:
        """
        根据索引获取目标点

        Args:
            indices: 目标索引 (num_envs,)

        Returns:
            goals: 目标点位置 (num_envs, 3)
        """
        # 限制索引范围
        indices = torch.clamp(indices, 0, self.num_goals - 1)
        # 使用gather获取对应索引的目标点
        return self.env_goals.gather(
            1, indices[:, None, None].expand(-1, -1, 3)
        ).squeeze(1)

    def _update_navigation_data(self):
        """更新导航相关数据：方向、距离、yaw角等"""
        # 获取机器人在地形坐标系中的位置
        robot_pos_w = self.robot.data.root_pos_w[:, :2]  # (num_envs, 2)
        robot_pos_terrain = robot_pos_w - self.env_origins[:, :2]

        # 计算到当前目标和下一个目标的相对位置
        self.target_pos_rel = self.cur_goals[:, :2] - robot_pos_terrain
        self.next_target_pos_rel = self.next_goals[:, :2] - robot_pos_terrain

        # 计算距离
        self.target_distance = torch.norm(self.target_pos_rel, dim=-1)
        self.next_target_distance = torch.norm(self.next_target_pos_rel, dim=-1)

        # 计算归一化方向
        norm = self.target_distance[:, None] + 1e-5
        self.target_dir_norm = self.target_pos_rel / norm

        norm = self.next_target_distance[:, None] + 1e-5
        self.next_target_dir_norm = self.next_target_pos_rel / norm

        # 计算目标yaw角
        self.target_yaw = torch.atan2(
            self.target_dir_norm[:, 1], self.target_dir_norm[:, 0]
        )
        self.next_target_yaw = torch.atan2(
            self.next_target_dir_norm[:, 1], self.next_target_dir_norm[:, 0]
        )

    def update(self):
        """每个step调用，更新导航状态"""
        # 检查是否到达当前目标
        reached_goal = self.target_distance < self.next_goal_threshold

        # 更新到达计时器
        self.reach_goal_timer[reached_goal] += 1

        # 检查是否应该切换到下一个目标
        should_switch = self.reach_goal_timer > (self.reach_goal_delay / self.dt)

        # 切换目标
        if should_switch.any():
            self.cur_goal_idx[should_switch] = torch.clamp(
                self.cur_goal_idx[should_switch] + 1, max=self.num_goals - 1
            )
            self.reach_goal_timer[should_switch] = 0

            # 更新当前和下一个目标
            self.cur_goals = self._get_goals_at_idx(self.cur_goal_idx)
            self.next_goals = self._get_goals_at_idx(self.cur_goal_idx + 1)

        # 更新导航数据
        self._update_navigation_data()

    def reset(self, env_ids: torch.Tensor):
        """
        重置指定环境的导航状态

        Args:
            env_ids: 需要重置的环境ID
        """
        # 重置目标索引到起点
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0

        # 更新导航点（考虑地形级别可能改变）
        terrain_levels = self.terrain.terrain_levels[env_ids]
        terrain_types = self.terrain.terrain_types[env_ids]
        self.env_goals[env_ids] = self.terrain_goals[terrain_levels, terrain_types]

        # 更新目标
        self.cur_goals = self._get_goals_at_idx(self.cur_goal_idx)
        self.next_goals = self._get_goals_at_idx(self.cur_goal_idx + 1)

        # 更新导航数据
        self._update_navigation_data()

    def get_observations(self) -> dict[str, torch.Tensor]:
        """
        获取导航相关的观察数据

        Returns:
            包含导航观察的字典
        """
        # 获取机器人的yaw角
        robot_heading = self.robot.data.heading_w  # (num_envs, 3)
        robot_yaw = torch.atan2(robot_heading[:, 1], robot_heading[:, 0])

        # 计算相对于机器人朝向的目标角度
        target_yaw_rel = self.target_yaw - robot_yaw
        next_target_yaw_rel = self.next_target_yaw - robot_yaw

        # 归一化角度到 [-pi, pi]
        target_yaw_rel = torch.atan2(
            torch.sin(target_yaw_rel), torch.cos(target_yaw_rel)
        )
        next_target_yaw_rel = torch.atan2(
            torch.sin(next_target_yaw_rel), torch.cos(next_target_yaw_rel)
        )

        return {
            "target_direction": self.target_dir_norm,  # (num_envs, 2)
            "target_distance": self.target_distance[:, None],  # (num_envs, 1)
            "target_yaw": self.target_yaw[:, None],  # (num_envs, 1)
            "target_yaw_rel": target_yaw_rel[:, None],  # (num_envs, 1)
            "next_target_direction": self.next_target_dir_norm,  # (num_envs, 2)
            "next_target_yaw_rel": next_target_yaw_rel[:, None],  # (num_envs, 1)
            "goal_idx": self.cur_goal_idx.float()[:, None],  # (num_envs, 1)
        }
