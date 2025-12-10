# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.sensors import ContactSensor
from isaaclab.sensors.ray_caster import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def base_euler_xy(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    获取机器人基座在世界坐标系下的欧拉角（roll 和 pitch）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    base_quat = asset.data.root_quat_w
    # 返回 roll 和 pitch 分量（shape: [num_envs, 2]）
    # 1. 调用函数，此时 euler_angles 是一个元组: (roll_tensor, pitch_tensor, yaw_tensor)
    euler_angles_tuple = euler_xyz_from_quat(base_quat)
    
    # 2. 从元组中取出我们需要的 roll 和 pitch
    roll = euler_angles_tuple[0]
    pitch = euler_angles_tuple[1]
    
    # 3. 使用 torch.stack 将这两个 tensor 沿着新的维度（dim=1）合并
    #    这会创建一个形状为 (num_envs, 2) 的新 tensor，正是我们想要的
    roll_pitch_tensor = torch.stack([roll, pitch], dim=1)
    
    # --- 修复结束 ---
    
    return roll_pitch_tensor


def base_euler_xyz(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The base roll, pitch and yaw of the asset in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    base_quat = asset.data.root_quat_w
    
    return euler_xyz_from_quat(base_quat)[:, :3]  # return roll, pitch and yaw


def foot_contact_state(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """NOTE: 来获取四只脚着地状态的  author:zzy
    The contact state of the specified bodies based on a force threshold.

    This function checks the contact force magnitude from a contact sensor against a threshold
    to determine if a body is in contact. It returns 1.0 if the force exceeds the
    threshold (in contact) and 0.0 otherwise.

    The sensor data is filtered based on the ``body_names`` provided in the ``sensor_cfg``.
    If ``body_names`` is None, it returns the contact state for all bodies tracked by the sensor.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # get contact forces from the current step
    # shape: (num_envs, num_bodies, 3)
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    # compute the norm of the force vector
    # shape: (num_envs, num_bodies)
    force_magnitude = torch.norm(forces, dim=-1)
    # return contact state based on threshold
    return (force_magnitude > threshold).float() - 0.5  # return -0.5 or 0.5 for better learning


def relative_height_scan(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, base_height_offset: float
) -> torch.Tensor:
    """
    计算机器人基座高度减去地形高度，以获得相对高度扫描。
    这精确地复现了 `root_states[:, 2] - offset - measured_heights` 的逻辑。
    """
    # 1. 从场景中获取机器人和传感器对象
    robot: Articulation = env.scene["robot"]
    sensor: RayCaster = env.scene[sensor_cfg.name]

    # 2. 获取机器人基座的世界 Z 坐标
    # [:, 2] 是 Z 坐标
    # Shape is (num_instances, num_bodies, 3).
    base_z = robot.data.body_com_pos_w[:, 0, 2].unsqueeze(1)  # 变形为 (num_envs, 1) 以便广播

    # 3. 获取传感器测量的世界坐标系下的地形高度
    # sensor.data.hit_points_w 的形状是 (num_envs, num_rays, 3)
    # [..., 2] 获取所有击中点的 Z 坐标
    measured_heights = sensor.data.ray_hits_w[..., 2]

    # 4. 执行您想要的计算
    relative_heights = base_z - base_height_offset - measured_heights

    return relative_heights