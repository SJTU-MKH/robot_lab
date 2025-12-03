# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.terrains.terrain_importer import TerrainImporter
from .crawl_terrain_generator import CrawlTerrainGenerator

if TYPE_CHECKING:
    from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg


class CrawlTerrainImporter(TerrainImporter):
    """自定义地形导入器，支持爬行地形和导航点系统"""

    terrain_prim_paths: list[str]
    terrain_origins: torch.Tensor | None
    env_origins: torch.Tensor

    def __init__(self, cfg: TerrainImporterCfg):
        # 检查配置是否有效
        cfg.validate()
        # 存储输入
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        # 为地形创建缓冲区
        self.terrain_prim_paths = list()
        self.terrain_origins = None
        self.env_origins = None  # 稍后在调用 `configure_env_origins` 时分配
        # 私有变量
        self._terrain_flat_patches = dict()

        # 根据配置自动导入地形
        if self.cfg.terrain_type == "generator":
            # 检查是否提供了配置
            if self.cfg.terrain_generator is None:
                raise ValueError(
                    "Input terrain type is 'generator' but no value provided for 'terrain_generator'."
                )
            # 生成地形
            self._terrain_generator_class = CrawlTerrainGenerator(
                cfg=self.cfg.terrain_generator,
                device=self.device,
            )
            self.import_mesh("terrain", self._terrain_generator_class.terrain_mesh)
            # 根据地形生成器配置地形原点
            self.configure_env_origins(self._terrain_generator_class.terrain_origins)
            # 引用平坦区域
            self._terrain_flat_patches = self._terrain_generator_class.flat_patches
        else:
            raise TypeError(
                f"Crawl Terrain type only supports generator, not {self.cfg.terrain_type}"
            )
        # 设置调试可视化的初始状态
        self.set_debug_vis(self.cfg.debug_vis)

    @property
    def terrain_generator_class(self):
        """返回地形生成器类实例"""
        return self._terrain_generator_class

    def _compute_env_origins_grid(self, num_envs: int, env_spacing: float) -> torch.Tensor:
        """基于配置的间距在网格中计算环境的原点"""
        # 根据环境数量创建张量
        env_origins = torch.zeros(num_envs, 3, device=self.device)
        # 创建原点网格
        num_rows = np.ceil(num_envs / int(np.sqrt(num_envs)))
        num_cols = np.ceil(num_envs / num_rows)
        ii, jj = torch.meshgrid(
            torch.arange(num_rows, device=self.device),
            torch.arange(num_cols, device=self.device),
            indexing="ij",
        )
        env_origins[:, 0] = -(ii.flatten()[:num_envs] - (num_rows - 1) / 2) * env_spacing
        env_origins[:, 1] = (jj.flatten()[:num_envs] - (num_cols - 1) / 2) * env_spacing
        env_origins[:, 2] = 0.0
        return env_origins
