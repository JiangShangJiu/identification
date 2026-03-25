"""MuJoCo 模型加载（与具体项目解耦）"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # mujoco 类型提示


def load_mujoco_model(
    model_file: str = "scene.xml",
    model_root: Path | str | None = None,
):
    """
    加载 MuJoCo 模型

    Args:
        model_file: 模型文件名
        model_root: 模型根目录，None 则需在 sys.path 中有可用的 load_panda 或直接指定

    Returns:
        model, data
    """
    try:
        import mujoco
    except ImportError:
        raise ImportError("请安装 mujoco: pip install mujoco")

    root = Path(model_root) if model_root else _find_model_root(model_file)
    xml_path = root / model_file
    if not xml_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def _find_model_root(model_file: str) -> Path:
    """查找模型根目录"""
    pkg_dir = Path(__file__).resolve().parents[1]
    candidates = [
        pkg_dir.parent / "franka_emika_panda",
        pkg_dir.parent / "mujoco" / "franka_emika_panda",
        Path("/home/xiaomeng/code/learn_robot/mujoco/franka_emika_panda"),
    ]
    for root in candidates:
        if root.is_dir() and (root / model_file).exists():
            return root
    raise FileNotFoundError(
        "未找到 Franka Panda 模型。请设置 model_root 或复制 franka_emika_panda 到项目目录。"
    )
