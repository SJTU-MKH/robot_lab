# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Crawl navigation MDP components."""

from .crawl_navigation_event import CrawlNavigationEvent
from .crawl_command import NavigationCommand, NavigationCommandCfg
from .crawl_observations import *
from .crawl_rewards import *
from .crawl_events import *
from .crawl_reset_events import *

__all__ = [
    "CrawlNavigationEvent",
    "NavigationCommand",
    "NavigationCommandCfg",
    # Observations
    "target_direction",
    "target_distance",
    "target_yaw_relative",
    "current_goal_index",
    "next_target_yaw_relative",
    # Rewards
    "heading_to_target",
    "progress_to_target",
    "reached_target",
    "velocity_to_target",
    "next_goal_awareness",
    "goal_progress_rate",
    # Events
    "initialize_navigation_system",
    "update_navigation_targets",
    "reset_navigation_on_termination",
    # Reset events
    "reset_robot_to_start_position",
    "adjust_terrain_difficulty",
]
