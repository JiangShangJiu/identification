"""MuJoCo 仿真数据采集：沿轨迹运行，采集 (q, qd, qdd, tau)"""

import numpy as np
from pathlib import Path

from .loader import load_mujoco_model
from .config import SimulationConfig

try:
    import mujoco
except ImportError:
    mujoco = None


def _expand_to_dof_array(value, dof: int, default: float) -> np.ndarray:
    """将标量/数组参数扩展为 (dof,) 形状。"""
    if value is None:
        return np.full(dof, default, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(dof, float(arr), dtype=float)
    if arr.shape != (dof,):
        raise ValueError(f"参数形状应为标量或 ({dof},)，实际为 {arr.shape}")
    return arr


def _ema_filter(x: np.ndarray, alpha: float) -> np.ndarray:
    """沿时间轴做一阶 EMA 平滑。alpha 越小越平滑。"""
    a = float(np.clip(alpha, 1e-6, 1.0))
    y = np.array(x, copy=True, dtype=float)
    for i in range(1, len(y)):
        y[i] = a * y[i] + (1.0 - a) * y[i - 1]
    return y


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
    add_friction: bool = True,
    coulomb_friction=None,
    viscous_friction=None,
    friction_smoothing: float = 0.02,
    add_stribeck: bool = True,
    stribeck_extra_ratio: float = 0.25,
    stribeck_velocity: float = 0.10,
    add_nonlinear_friction: bool = True,
    nonlinear_friction_scale: float = 0.03,
    nonlinear_friction_seed: int | None = None,
    add_state_noise: bool = True,
    state_noise_sigma_q: float = 8e-4,
    state_noise_seed: int | None = None,
    state_vel_ema_alpha: float = 0.25,
    state_acc_ema_alpha: float = 0.2,
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
        add_friction: 是否叠加摩擦力矩（库仑+粘性，可附加随机非线性项）
        coulomb_friction: 库仑摩擦系数（标量或 shape=(dof,)；None 则默认 0.20 Nm）
        viscous_friction: 粘性摩擦系数（标量或 shape=(dof,)；None 则默认 0.05 Nms/rad）
        friction_smoothing: 库仑摩擦平滑参数，tanh(qd / friction_smoothing)
        add_stribeck: 是否叠加 Stribeck 低速效应
        stribeck_extra_ratio: 静摩擦相对库仑摩擦的增量比例（Fs = Fc * (1 + ratio)）
        stribeck_velocity: Stribeck 特征速度（越小则低速突起区越窄）
        add_nonlinear_friction: 是否叠加随机非线性摩擦项
        nonlinear_friction_scale: 非线性摩擦强度系数（越大越“非线性”）
        nonlinear_friction_seed: 非线性摩擦随机种子；None 表示每次运行不同
        add_state_noise: 是否给状态量添加测量噪声（默认开启）
        state_noise_sigma_q: 关节位置噪声标准差 (rad)
        state_noise_seed: 状态噪声随机种子；None 表示每次运行不同
        state_vel_ema_alpha: 对差分得到的 qd 做 EMA 平滑系数
        state_acc_ema_alpha: 对差分得到的 qdd 做 EMA 平滑系数
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

    if add_friction:
        fc = _expand_to_dof_array(coulomb_friction, dof=dof, default=0.20)
        fv = _expand_to_dof_array(viscous_friction, dof=dof, default=0.05)
        v_eps = max(float(friction_smoothing), 1e-6)
        v_abs = np.abs(qd_arr)

        if add_stribeck:
            vs = max(float(stribeck_velocity), 1e-6)
            fs = fc * (1.0 + float(stribeck_extra_ratio))
            stribeck_gain = np.exp(-((v_abs / vs) ** 2))
            fc_eff = fc + (fs - fc) * stribeck_gain
        else:
            fc_eff = fc

        friction_tau = fc_eff * np.tanh(qd_arr / v_eps) + fv * qd_arr

        if add_nonlinear_friction:
            rng_nl = np.random.default_rng(nonlinear_friction_seed)
            quad = rng_nl.uniform(-1.0, 1.0, size=dof) * nonlinear_friction_scale
            cubic = rng_nl.uniform(-1.0, 1.0, size=dof) * (0.2 * nonlinear_friction_scale)
            sin_amp = rng_nl.uniform(0.0, 1.0, size=dof) * nonlinear_friction_scale
            sin_freq = rng_nl.uniform(2.0, 6.0, size=dof)
            sin_phase = rng_nl.uniform(0.0, 2 * np.pi, size=dof)
            friction_tau += quad * (qd_arr * np.abs(qd_arr))
            friction_tau += cubic * (qd_arr**3)
            friction_tau += sin_amp * np.sin(sin_freq * qd_arr + sin_phase)

        tau_arr += friction_tau

        if verbose:
            fric_msg = "  已叠加摩擦: Coulomb + viscous"
            if add_stribeck:
                fric_msg += " + Stribeck"
            if add_nonlinear_friction:
                fric_msg += " + nonlinear(random)"
            print(fric_msg)
            print(f"    Coulomb系数范围: [{fc.min():.4f}, {fc.max():.4f}] Nm")
            print(f"    粘性系数范围: [{fv.min():.4f}, {fv.max():.4f}] Nms/rad")
            if add_stribeck:
                print(f"    Stribeck: extra_ratio={stribeck_extra_ratio:.3f}, v_s={stribeck_velocity:.4f} rad/s")

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

    q_true = q_arr
    qd_true = qd_arr
    qdd_true = qdd_arr
    q_meas = q_true
    qd_meas = qd_true
    qdd_meas = qdd_true

    if add_state_noise:
        rng_state = np.random.default_rng(state_noise_seed)
        q_meas = q_true + rng_state.normal(0.0, float(state_noise_sigma_q), size=q_true.shape)
        qd_meas = np.gradient(q_meas, dt, axis=0)
        qdd_meas = np.gradient(qd_meas, dt, axis=0)
        qd_meas = _ema_filter(qd_meas, state_vel_ema_alpha)
        qdd_meas = _ema_filter(qdd_meas, state_acc_ema_alpha)
        if verbose:
            seed_info = f", seed={state_noise_seed}" if state_noise_seed is not None else ", seed=None(每次不同)"
            print(f"  状态加噪: q ~ N(0,{state_noise_sigma_q}^2){seed_info}")
            print(f"  状态平滑: EMA(alpha_qd={state_vel_ema_alpha:.3f}, alpha_qdd={state_acc_ema_alpha:.3f})")

    return {
        "time": t_arr,
        "q": q_meas,
        "qd": qd_meas,
        "qdd": qdd_meas,
        "tau": tau_arr,
        "q_true": q_true,
        "qd_true": qd_true,
        "qdd_true": qdd_true,
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
