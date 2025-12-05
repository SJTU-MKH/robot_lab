# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Navigation-based command generator for crawl task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
from isaaclab.utils.configclass import configclass
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class NavigationCommandCfg(CommandTermCfg):
    """Configuration for navigation-based command generation."""

    class_type: type = None  # Will be set to NavigationCommand in __post_init__

    # Asset name
    asset_name: str = "robot"

    # Command velocity parameters
    lin_vel_x_range: tuple[float, float] = (0.3, 0.8)  # Forward velocity range
    ang_vel_z_range: tuple[float, float] = (-0.8, 0.8)  # Angular velocity range

    # Heading control
    heading_control_stiffness: float = 0.5  # How aggressively to turn towards target

    # Resampling (not used in navigation mode, but kept for compatibility)
    resampling_time_range: tuple[float, float] = (10.0, 10.0)

    # Debug visualization
    debug_vis: bool = True

    # Visualization markers
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/navigation_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = (
        BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/navigation_current")
    )

    def __post_init__(self):
        """Post initialization to set class_type."""
        if self.class_type is None:
            self.class_type = NavigationCommand
        # Set marker scales
        self.goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
        self.current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
        super().__post_init__()


class NavigationCommand(CommandTerm):
    """
    Command generator that uses navigation waypoints to generate velocity commands.

    Instead of randomly sampling velocity commands, this generator:
    1. Reads the target waypoint from the navigation event manager
    2. Computes the desired heading towards the waypoint
    3. Generates forward velocity and angular velocity commands
    """

    cfg: NavigationCommandCfg

    def __init__(self, cfg: NavigationCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # Velocity command buffer (num_envs, 3) - [lin_vel_x, lin_vel_y, ang_vel_z]
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)

        # Metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "NavigationCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tForward velocity range: {self.cfg.lin_vel_x_range}\n"
        msg += f"\tAngular velocity range: {self.cfg.ang_vel_z_range}\n"
        msg += f"\tHeading control stiffness: {self.cfg.heading_control_stiffness}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    def _update_metrics(self):
        # Time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        # Log tracking errors
        self.metrics["error_vel_xy"] += (
            torch.norm(
                self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2],
                dim=-1,
            )
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
            / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """
        Resample commands for specified environments.
        For navigation mode, we just reset to zero and let _update_command handle it.
        """
        # Sample forward velocity
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.lin_vel_x_range)

        # Zero out lateral and angular velocity (will be computed in _update_command)
        self.vel_command_b[env_ids, 1] = 0.0
        self.vel_command_b[env_ids, 2] = 0.0

    def _update_command(self):
        """
        Update commands based on navigation waypoints.
        This is called every step.
        """
        # Check if navigation event exists
        if not hasattr(self._env, "navigation_event"):
            return

        # Get navigation target yaw from navigation event manager
        from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat

        nav_event = self._env.navigation_event
        target_yaw = nav_event.target_yaw

        # Get robot's current yaw
        _, _, robot_yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)

        # Calculate heading error
        heading_error = wrap_to_pi(target_yaw - robot_yaw)

        # Generate angular velocity command proportional to heading error
        ang_vel_command = heading_error * self.cfg.heading_control_stiffness

        # Clip angular velocity to limits
        ang_vel_command = torch.clamp(
            ang_vel_command,
            min=self.cfg.ang_vel_z_range[0],
            max=self.cfg.ang_vel_z_range[1],
        )

        # Set angular velocity command
        self.vel_command_b[:, 2] = ang_vel_command

        # Keep lateral velocity at zero for crawling
        self.vel_command_b[:, 1] = 0.0

        # Forward velocity is already set in _resample_command
        # Optionally, we could modulate it based on distance to target:
        # distance_to_target = nav_event.target_distance
        # self.vel_command_b[:, 0] *= torch.clamp(distance_to_target / 2.0, 0.0, 1.0)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization."""
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                # Create visualizers
                self.goal_vel_visualizer = VisualizationMarkers(
                    self.cfg.goal_vel_visualizer_cfg
                )
                self.current_vel_visualizer = VisualizationMarkers(
                    self.cfg.current_vel_visualizer_cfg
                )
            # Set visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualization."""
        if not self.robot.is_initialized:
            return

        # Get marker location (above robot base)
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        # Resolve scales and quaternions for arrows
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.command[:, :2]
        )
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b[:, :2]
        )

        # Display markers
        self.goal_vel_visualizer.visualize(
            base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale
        )
        self.current_vel_visualizer.visualize(
            base_pos_w, vel_arrow_quat, vel_arrow_scale
        )

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert xy velocity to arrow scale and quaternion for visualization."""
        # Get default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale

        # Arrow scale (length proportional to velocity magnitude)
        # Ensure arrow_scale is on the same device as xy_velocity
        arrow_scale = torch.tensor(default_scale, device=xy_velocity.device).repeat(
            xy_velocity.shape[0], 1
        )
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        # Arrow direction (yaw angle from velocity)
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        # Convert from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
