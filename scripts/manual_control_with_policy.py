#!/usr/bin/env python3
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""手动控制脚本 - 加载训练好的策略网络并通过键盘控制机器人"""

import argparse
import os
import torch

# 启动Isaac Sim
from isaaclab.app import AppLauncher

# 添加参数
parser = argparse.ArgumentParser(description="手动控制机器人")
parser.add_argument(
    "--task",
    type=str,
    default="custom_boxgarden",
    help="任务名称",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/workspace/robot_lab/logs/rsl_rl/unitree_go2_rough/2025-12-04_03-10-41/model_14999.pt",
    help="模型checkpoint路径",
)
parser.add_argument("--num_envs", type=int, default=1, help="环境数量")
parser.add_argument("--enable_lidar", action="store_true", default=True, help="启用激光雷达传感器")
parser.add_argument("--lidar_vis", action="store_true", default=True, help="可视化激光雷达数据")
parser.add_argument("--lidar_rays", type=int, default=32, choices=[16, 32, 64], help="激光雷达光束数量")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动模拟器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 现在可以导入其他模块
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.managers import SceneEntityCfg

# 导入任务注册
import robot_lab.tasks  # noqa: F401
from boxgarden_crawl_env_cfg import BoxGardenCrawlCompatEnvCfg_PLAY


class ManualController:
    """手动控制器 - 通过键盘控制机器人"""

    def __init__(self, env: ManagerBasedRLEnv, policy_path: str, agent_cfg, enable_lidar_vis: bool = False):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.device = self.env.device
        self.agent_cfg = agent_cfg
        self.enable_lidar_vis = enable_lidar_vis

        # 手动命令
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel_z = 0.0

        # 记录上一帧按键状态，用于检测按键事件
        self.last_key_states = {}

        # 加载策略网络 (this also sets self.env_wrapper)
        self.policy = self._load_policy(policy_path)
        
        # 检查是否有激光雷达
        self.has_lidar = hasattr(self.env.scene, "lidar")
        if self.has_lidar:
            print(f"[INFO] 检测到激光雷达传感器: {self.env.scene.lidar}")
        
        print("\n" + "=" * 60)
        print("手动控制模式 - 增量式控制")
        print("=" * 60)
        print("键盘控制:")
        print("  W/S: 前进速度 +0.1/-0.1")
        print("  A/D: 左移速度 +0.1/-0.1")
        print("  Q/E: 旋转速度 +0.2/-0.2")
        print("  Space: 停止")
        if self.has_lidar:
            print("  L: 切换激光雷达可视化")
        print("  ESC: 退出")
        print("=" * 60 + "\n")

    def _load_policy(self, policy_path: str):
        """加载策略网络"""
        print(f"[INFO] 加载策略网络: {policy_path}")

        # Import required modules
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        # Use the agent config passed to constructor
        agent_cfg = self.agent_cfg
        agent_cfg.device = str(self.device)

        # Wrap environment for RSL-RL
        self.env_wrapper = RslRlVecEnvWrapper(
            self.env, clip_actions=agent_cfg.clip_actions
        )

        # Create runner and load checkpoint
        runner = OnPolicyRunner(
            self.env_wrapper, agent_cfg.to_dict(), log_dir=None, device=self.device
        )
        runner.load(policy_path)

        # Get inference policy
        policy = runner.get_inference_policy(device=self.device)

        print(f"[INFO] 策略网络加载成功!")
        return policy

    def process_keyboard_input(self):
        """处理键盘输入 - 增量式控制"""
        import carb
        import omni.appwindow

        # Get keyboard from app window
        appwindow = omni.appwindow.get_default_app_window()
        input_interface = carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()

        # 速度增量
        lin_vel_step = 0.1
        ang_vel_step = 0.2

        # 检查按键状态变化（从未按下到按下）
        def is_key_pressed(key):
            """检测按键是否刚被按下（边缘触发）"""
            current_state = input_interface.get_keyboard_value(keyboard, key)
            last_state = self.last_key_states.get(key, 0)
            self.last_key_states[key] = current_state
            # 只在从0变1时触发（按下的瞬间）
            return current_state > 0 and last_state == 0

        # W - 前进（按一下增加0.1）
        if is_key_pressed(carb.input.KeyboardInput.W):
            self.lin_vel_x = min(self.lin_vel_x + lin_vel_step, 1.0)
            print(f"[CMD] vx = {self.lin_vel_x:.2f}")

        # S - 后退（按一下减少0.1）
        if is_key_pressed(carb.input.KeyboardInput.S):
            self.lin_vel_x = max(self.lin_vel_x - lin_vel_step, -1.0)
            print(f"[CMD] vx = {self.lin_vel_x:.2f}")

        # A - 左移（按一下增加0.1）
        if is_key_pressed(carb.input.KeyboardInput.A):
            self.lin_vel_y = min(self.lin_vel_y + lin_vel_step, 0.5)
            print(f"[CMD] vy = {self.lin_vel_y:.2f}")

        # D - 右移（按一下减少0.1）
        if is_key_pressed(carb.input.KeyboardInput.D):
            self.lin_vel_y = max(self.lin_vel_y - lin_vel_step, -0.5)
            print(f"[CMD] vy = {self.lin_vel_y:.2f}")

        # Q - 左转（按一下增加0.2）
        if is_key_pressed(carb.input.KeyboardInput.Q):
            self.ang_vel_z = min(self.ang_vel_z + ang_vel_step, 1.0)
            print(f"[CMD] wz = {self.ang_vel_z:.2f}")

        # E - 右转（按一下减少0.2）
        if is_key_pressed(carb.input.KeyboardInput.E):
            self.ang_vel_z = max(self.ang_vel_z - ang_vel_step, -1.0)
            print(f"[CMD] wz = {self.ang_vel_z:.2f}")

        # Space - 停止（清零所有速度）
        if is_key_pressed(carb.input.KeyboardInput.SPACE):
            self.lin_vel_x = 0.0
            self.lin_vel_y = 0.0
            self.ang_vel_z = 0.0
            print(f"[CMD] STOP - All velocities reset to 0")

        # ESC - 退出
        if is_key_pressed(carb.input.KeyboardInput.ESCAPE):
            return False

        # L - 切换激光雷达可视化
        if self.has_lidar and is_key_pressed(carb.input.KeyboardInput.L):
            self.enable_lidar_vis = not self.enable_lidar_vis
            # 切换debug_vis
            if hasattr(self.env.scene.lidar, "cfg"):
                self.env.scene.lidar.cfg.debug_vis = self.enable_lidar_vis
            print(f"[INFO] 激光雷达可视化: {'开启' if self.enable_lidar_vis else '关闭'}")

        return True

    def set_command(self):
        """设置机器人命令"""
        # 构建命令向量 [lin_vel_x, lin_vel_y, ang_vel_z]
        command = torch.tensor(
            [[self.lin_vel_x, self.lin_vel_y, self.ang_vel_z]], device=self.device
        )

        # 直接设置到命令管理器
        if hasattr(self.env.command_manager, "_terms"):
            for term_name, term in self.env.command_manager._terms.items():
                if hasattr(term, "command"):
                    term.command[:] = command

    def run(self):
        """运行控制循环"""
        # Reset environment and get initial observations
        self.env_wrapper.reset()
        obs = self.env_wrapper.get_observations()

        while True:
            # 处理键盘输入
            if not self.process_keyboard_input():
                break

            # 设置手动命令
            self.set_command()

            # 使用策略网络生成动作
            with torch.inference_mode():
                actions = self.policy(obs)

            # 执行动作 (wrapper returns obs, rewards, dones, extras)
            obs, rewards, dones, extras = self.env_wrapper.step(actions)

            # 打印激光雷达信息
            if self.has_lidar and self.env.episode_length_buf[0] % 50 == 0:
                lidar_data = self.env.scene.lidar.data
                if hasattr(lidar_data, "distances"):
                    distances = lidar_data.distances[0].cpu()  # 第一个环境
                    min_dist = distances.min().item()
                    mean_dist = distances.mean().item()
                    print(f"[LIDAR] 最近障碍物: {min_dist:.2f}m, 平均距离: {mean_dist:.2f}m")

            # 打印当前状态
            if self.env.episode_length_buf[0] % 50 == 0:
                print(
                    f"命令: vx={self.lin_vel_x:.2f}, vy={self.lin_vel_y:.2f}, wz={self.ang_vel_z:.2f}, "
                    f"episode_len={self.env.episode_length_buf[0].item()}"
                )

            # 检查是否触发重置
            if dones.any():
                print(
                    f"[WARNING] 环境被重置! episode_length={self.env.episode_length_buf[0].item()}"
                )
                # 打印终止原因
                if hasattr(self.env, "termination_manager"):
                    print(f"终止原因: {extras}")
                # 注释掉自动重置，让我们看看是否还会重置
                # self.env_wrapper.reset()
                # obs = self.env_wrapper.get_observations()

        print("\n[INFO] 退出手动控制模式")


def main():
    """主函数"""
    # 检查checkpoint是否存在
    if not os.path.exists(args_cli.checkpoint):
        print(f"[ERROR] Checkpoint不存在: {args_cli.checkpoint}")
        simulation_app.close()
        return

    # 使用自定义BoxGarden环境配置
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    print(f"[INFO] 使用自定义BoxGarden环境（兼容爬行高度checkpoint）")
    env_cfg = BoxGardenCrawlCompatEnvCfg_PLAY()
    
    # 添加激光雷达传感器
    if args_cli.enable_lidar:
        print(f"[INFO] 添加激光雷达传感器 ({args_cli.lidar_rays} 光束)")
        
        # 创建激光雷达配置
        lidar_cfg = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),  # 在base上方20cm
            ray_alignment="yaw",  # 只跟随yaw旋转
            pattern_cfg=patterns.LidarPatternCfg(
                channels=1,  # 单线激光雷达
                vertical_fov_range=(0.0, 0.0),  # 水平扫描
                horizontal_fov_range=(-180.0, 180.0),  # 360度
                horizontal_res=360.0 / args_cli.lidar_rays,  # 根据光束数计算分辨率
            ),
            max_distance=10.0,  # 最大检测距离10米
            drift_range=(0.0, 0.0),  # 无漂移
            debug_vis=args_cli.lidar_vis,  # 可视化
            mesh_prim_paths=["/World/ground"],  # 检测地面
        )
        
        # 添加到场景配置
        env_cfg.scene.lidar = lidar_cfg
    
    # 使用爬行高度的agent配置
    agent_cfg = load_cfg_from_registry(
        "RobotLab-Isaac-Velocity-CrawlHeight-Unitree-Go2-Play-v0",
        "rsl_rl_cfg_entry_point",
    )

    # 手动控制模式：只使用1个机器人
    env_cfg.scene.num_envs = 1

    # 禁用所有终止条件，让机器人可以一直运行
    if hasattr(env_cfg, "terminations"):
        # 禁用超时
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        # 禁用地形边界检查
        if hasattr(env_cfg.terminations, "terrain_out_of_bounds"):
            env_cfg.terminations.terrain_out_of_bounds = None
        # 禁用其他可能的终止条件
        for attr_name in dir(env_cfg.terminations):
            if not attr_name.startswith("_"):
                try:
                    setattr(env_cfg.terminations, attr_name, None)
                except:
                    pass

    # 设置很长的episode时间
    env_cfg.episode_length_s = 100000.0  # 非常长的时间

    print(f"[INFO] 设置手动控制模式：1个机器人，禁用超时重置")

    # 创建环境
    from isaaclab.envs import ManagerBasedRLEnv

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)

    # 创建控制器并运行
    controller = ManualController(env, args_cli.checkpoint, agent_cfg, enable_lidar_vis=args_cli.lidar_vis)
    controller.run()

    # 关闭环境
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
