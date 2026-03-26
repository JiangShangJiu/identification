"""MuJoCo 仿真数据采集：沿轨迹运行，采集 (q, qd, qdd, tau)"""

import numpy as np
from pathlib import Path

from .loader import load_mujoco_model
from .config import SimulationConfig

try:
    import mujoco
except ImportError:
    mujoco = None


def collect_data(
    config: SimulationConfig | None = None,
    model_file: str = "scene.xml",
    model_root: str | Path | None = None,
    dof: int = 7,
    duration: float | None = None,
    dt: float = 0.001,
    use_harmonic: bool = True,
    n_periods: int = 3,
    trajectory_type: str = "sine",
    add_noise: bool = False,
    noise_sigma: float | None = None,
    noise_mix_laplace: bool = True,
    noise_laplace_ratio: float = 0.55,
    noise_seed: int | None = None,
    trajectory_factory=None,
    verbose: bool = True,
) -> dict:
    """
    采集辨识数据：在 MuJoCo 中沿轨迹设置 (q,qd,qdd)，用 mj_inverse 得到 τ

    Args:
        config: 仿真配置，若提供则覆盖部分参数
        model_file, model_root, dof: 模型相关
        duration, dt: 时长与采样间隔
        use_harmonic: 是否用多谐波轨迹
        n_periods: 多谐波周期数
        trajectory_type: sine|polynomial|random（非 harmonic 时）
        add_noise: 是否在力矩上加随机噪声
        noise_sigma: 高斯分量标准差 (Nm)，默认取 config.torque_noise_sigma
        noise_mix_laplace: 是否再叠加拉普拉斯噪声（更重尾、抖动更“乱”）
        noise_laplace_ratio: 拉普拉斯尺度 = sigma * 该系数
        noise_seed: 随机种子；None 表示每次运行噪声不同
        trajectory_factory: 自定义轨迹生成函数 (duration, dt) -> (t, q, qd, qdd)
        verbose: 是否打印

    Returns:
        dict: time, q, qd, qdd, tau
    """
    if mujoco is None:
        raise ImportError("请安装 mujoco: pip install mujoco")

    cfg = config or SimulationConfig(
        model_path=model_file, model_root=model_root, dof=dof
    )
    dof = cfg.dof

    # 生成轨迹
    if trajectory_factory is not None:
        t_arr, q_arr, qd_arr, qdd_arr = trajectory_factory(duration or 10.0, dt)
    else:
        from ..trajectory import make_trajectory_for_collect
        t_arr, q_arr, qd_arr, qdd_arr = make_trajectory_for_collect(
            use_harmonic=use_harmonic,
            duration=duration,
            dt=dt,
            n_periods=n_periods,
            trajectory_type=trajectory_type,
            dof=dof,
        )

    n_samples = len(t_arr)
    _model_root = getattr(cfg, "model_root", None) or model_root

    model, data = load_mujoco_model(
        model_file=str(cfg.model_path),
        model_root=_model_root,
    )

    tau_arr = np.zeros((n_samples, dof))
    if verbose:
        traj_name = "harmonic" if use_harmonic else trajectory_type
        print(f"采集数据: {n_samples} 点, dt={dt}s, trajectory={traj_name}")

    for i in range(n_samples):
        data.qpos[:dof] = q_arr[i]
        data.qvel[:dof] = qd_arr[i]
        data.qacc[:dof] = qdd_arr[i]
        mujoco.mj_inverse(model, data)
        tau_arr[i] = data.qfrc_inverse[:dof]

    if add_noise:
        sigma = noise_sigma if noise_sigma is not None else cfg.torque_noise_sigma
        rng = np.random.default_rng(noise_seed)
        noise = rng.normal(0.0, sigma, tau_arr.shape)
        if noise_mix_laplace:
            b = sigma * noise_laplace_ratio
            noise = noise + rng.laplace(0.0, b, tau_arr.shape)
        tau_arr += noise
        if verbose:
            extra = f"+ Laplace(0,{sigma * noise_laplace_ratio:.3f})" if noise_mix_laplace else ""
            seed_info = f", seed={noise_seed}" if noise_seed is not None else ", seed=None(每次不同)"
            print(f"  力矩加噪: N(0,{sigma}^2) {extra}{seed_info}")

    return {
        "time": t_arr,
        "q": q_arr,
        "qd": qd_arr,
        "qdd": qdd_arr,
        "tau": tau_arr,
    }


class SimulationCollector:
    """封装的数据采集器"""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def collect(self, **kwargs) -> dict:
        return collect_data(config=self.config, **kwargs)


def save_data(data: dict, path: str | Path):
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in data.items()})


def load_data(path: str | Path) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}
