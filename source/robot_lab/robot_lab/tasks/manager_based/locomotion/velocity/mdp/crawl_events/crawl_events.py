# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Events for crawl navigation - initialize navigation system."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .crawl_navigation_event import CrawlNavigationEvent


def initialize_navigation_system(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    初始化导航系统（在startup时调用一次）

    Args:
        env: 环境实例
        env_ids: 环境ID（startup时是所有环境）
        asset_cfg: 机器人资产配置
    """
    # 只初始化一次
    if not hasattr(env, "navigation_event"):
        env.navigation_event = CrawlNavigationEvent(
            env=env,
            asset_cfg=asset_cfg,
            next_goal_threshold=0.5,
            reach_goal_delay=0.5,
        )
        print("[INFO] Navigation system initialized for Crawl environment")


def update_navigation_targets(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    更新导航目标（每个step调用）

    Args:
        env: 环境实例
        env_ids: 环境ID
        asset_cfg: 机器人资产配置
    """
    if hasattr(env, "navigation_event"):
        env.navigation_event.update()


def reset_navigation_on_termination(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    重置导航状态（在环境reset时调用）

    Args:
        env: 环境实例
        env_ids: 需要重置的环境ID
        asset_cfg: 机器人资产配置
    """
    if hasattr(env, "navigation_event"):
        env.navigation_event.reset(env_ids)
