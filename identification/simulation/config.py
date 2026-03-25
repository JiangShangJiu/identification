"""仿真配置：模型路径、DOF 等"""

from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    """MuJoCo 仿真与数据采集配置"""

    model_path: str | Path = "scene.xml"
    model_root: str | Path | None = None
    dof: int = 7
    torque_noise_sigma: float = 0.5

    def resolve_model_root(self) -> Path:
        """解析模型根目录"""
        if self.model_root is not None:
            return Path(self.model_root)
        # 默认查找：当前包附近、或环境变量
        candidates = [
            Path(__file__).resolve().parents[2] / "franka_emika_panda",
            Path(__file__).resolve().parents[2] / "mujoco" / "franka_emika_panda",
        ]
        for root in candidates:
            if root.is_dir() and (root / str(self.model_path)).exists():
                return root
        raise FileNotFoundError(
            f"未找到模型目录。请设置 model_root 或确保 franka_emika_panda 存在。"
        )
