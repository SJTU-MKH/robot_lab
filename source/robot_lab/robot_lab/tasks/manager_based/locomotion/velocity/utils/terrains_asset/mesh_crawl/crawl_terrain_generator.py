import numpy as np
import trimesh

from isaaclab.terrains.terrain_generator import TerrainGenerator
from .crawl_terrain_cfg import CrawlTerrainGeneratorCfg, ThreeTunnelChamberTerrainCfg


class CrawlTerrainGenerator(TerrainGenerator):
    """自定义地形生成器，支持导航点系统"""

    def __init__(self, cfg: CrawlTerrainGeneratorCfg, device: str = "cpu"):
        self.num_goals = cfg.num_goals
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, self.num_goals, 3))
        self.terrain_names = np.zeros((cfg.num_rows, cfg.num_cols, 1)).astype(str)
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale) + 1
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale) + 1
        self.total_width_pixels = width_pixels * cfg.num_rows
        self.total_length_pixels = length_pixels * cfg.num_cols

        super().__init__(cfg=cfg, device=device)
        self.cfg: CrawlTerrainGeneratorCfg

    def _generate_random_terrains(self):
        """根据随机采样的难度参数添加地形"""
        # 归一化子地形的比例
        proportions = np.array(
            [sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()]
        )
        proportions /= np.sum(proportions)
        # 创建所有地形配置的列表
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
        sub_terrains_names = list(self.cfg.sub_terrains.keys())
        # 随机采样子地形
        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            # 子地形的坐标索引
            (sub_row, sub_col) = np.unravel_index(
                index, (self.cfg.num_rows, self.cfg.num_cols)
            )
            # 随机采样地形索引
            sub_index = self.np_rng.choice(len(proportions), p=proportions)
            # 随机采样难度参数
            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
            # 生成地形
            sub_terrains_name = sub_terrains_names[sub_index]
            self.terrain_type[sub_row, sub_col] = sub_col
            sub_terrains_cfg = sub_terrains_cfgs[sub_index]
            mesh, origin, sub_terrain_goal = self._get_terrain_mesh(
                difficulty, sub_terrains_cfg
            )
            # 添加到子地形
            self.terrain_names[sub_row, sub_col] = sub_terrains_name
            self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrain_goal)

    def _generate_curriculum_terrains(self):
        """基于课程学习的难度参数添加地形"""
        # 归一化子地形的比例
        proportions = np.array(
            [sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()]
        )
        proportions /= np.sum(proportions)

        sub_indices = []
        for index in range(self.cfg.num_cols):
            sub_index = np.min(
                np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0]
            )
            sub_indices.append(sub_index)
        sub_indices = np.array(sub_indices, dtype=np.int32)
        # 创建所有地形配置的列表
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
        sub_terrains_names = list(self.cfg.sub_terrains.keys())
        # 基于课程的子地形
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                lower, upper = self.cfg.difficulty_range
                if self.cfg.random_difficulty:
                    difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                else:
                    difficulty = sub_row / (self.cfg.num_rows - 1)

                difficulty = lower + (upper - lower) * difficulty
                # 生成地形
                sub_terrains_cfg = sub_terrains_cfgs[sub_indices[sub_col]]
                sub_terrains_name = sub_terrains_names[sub_indices[sub_col]]
                mesh, origin, sub_terrain_goal = self._get_terrain_mesh(
                    difficulty, sub_terrains_cfg
                )
                # 添加到子地形
                self.terrain_type[sub_row, sub_col] = sub_indices[sub_col]
                self.terrain_names[sub_row, sub_col] = sub_terrains_name
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrain_goal)

    def _get_terrain_mesh(
        self,
        difficulty: float,
        cfg: ThreeTunnelChamberTerrainCfg,
    ) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
        # 复制配置
        cfg: ThreeTunnelChamberTerrainCfg = cfg.copy()
        # 添加其他参数到子地形配置
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        # 生成地形（返回4个值：meshes, origin, goals_xy, goal_heights）
        meshes, origin, goals_xy, goal_heights = cfg.function(
            difficulty, cfg, self.num_goals
        )
        mesh = trimesh.util.concatenate(meshes)
        # 偏移mesh使其位于中心
        transform = np.eye(4)
        transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        # 改变原点位于子地形中心
        origin += transform[0:3, -1]
        # 对goals也应用相同的偏移，并合并xy和z坐标
        goals_xy[:, :2] += transform[0:2, -1]
        # 将xy坐标和高度合并为xyz坐标
        goals = np.zeros((self.num_goals, 3))
        goals[:, :2] = goals_xy  # xy坐标
        goals[:, 2] = goal_heights  # z坐标（高度）

        return mesh, origin, goals

    def _add_sub_terrain(
        self,
        mesh: trimesh.Trimesh,
        origin: np.ndarray,
        row: int,
        col: int,
        sub_terrain_goal: np.ndarray,
    ):
        # 变换mesh到正确位置
        transform = np.eye(4)
        transform[0:2, -1] = (row + 0.5) * self.cfg.size[0], (
            col + 0.5
        ) * self.cfg.size[1]
        mesh.apply_transform(transform)
        # 添加mesh到列表
        self.terrain_meshes.append(mesh)
        # 添加原点到列表
        self.terrain_origins[row, col] = origin + transform[:3, -1]
        # 添加导航点（完整的xyz坐标）
        self.goals[row, col] = sub_terrain_goal
