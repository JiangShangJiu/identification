"""
机器人动力学参数辨识模块

包含：
- simulation: MuJoCo 仿真数据采集
- trajectory: 激励轨迹生成
- dynamics: 动力学模型与回归矩阵
- solver: 最小二乘辨识
"""

from .simulation import SimulationCollector, SimulationConfig, collect_data, save_data, load_data
from .trajectory import generate_trajectory, HarmonicExcitationTrajectory
from .solver import identify
from .compare import compare

__all__ = [
    "SimulationCollector",
    "SimulationConfig",
    "collect_data",
    "save_data",
    "load_data",
    "generate_trajectory",
    "HarmonicExcitationTrajectory",
    "identify",
    "compare",
]
