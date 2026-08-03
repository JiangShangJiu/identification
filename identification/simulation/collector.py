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


def _zero_phase_lowpass(x: np.ndarray, dt: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """零相位 Butterworth 低通（离线辨识标准做法）。优先 Gustaffson 延拓以减轻端点效应。"""
    from scipy.signal import butter, filtfilt

    x = np.asarray(x, dtype=float)
    fs = 1.0 / float(dt)
    nyq = 0.5 * fs
    wn = min(max(float(cutoff_hz) / nyq, 1e-6), 0.99)
    b, a = butter(int(order), wn, btype="low")
    padlen = min(3 * max(len(a), len(b)), max(0, len(x) - 1))

    def _filt(sig: np.ndarray) -> np.ndarray:
        try:
            return filtfilt(b, a, sig, method="gust")
        except TypeError:
            return filtfilt(b, a, sig, padlen=padlen)

    if x.ndim == 1:
        return _filt(x)
    out = np.empty_like(x, dtype=float)
    for j in range(x.shape[1]):
        out[:, j] = _filt(x[:, j])
    return out


def _trim_edge_samples(data: dict, n_trim: int) -> dict:
    """去掉首尾各 n_trim 个采样，抑制滤波边界暂态。"""
    if n_trim <= 0:
        return data
    n = len(data["time"])
    if 2 * n_trim >= n:
        raise ValueError(f"edge trim 过大: n_trim={n_trim}, n_samples={n}")
    sl = slice(n_trim, n - n_trim)
    out = {}
    for k, v in data.items():
        arr = np.asarray(v)
        if arr.ndim >= 1 and len(arr) == n:
            out[k] = arr[sl]
        else:
            out[k] = v
    return out


def _estimate_states_butterworth(
    q_noisy: np.ndarray,
    dt: float,
    cutoff_hz: float = 3.0,
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    由含噪位置估计 (q, qd, qdd)：
    先对 q 零相位低通，再差分；每次差分后再低通，抑制噪声放大。
    """
    q_f = _zero_phase_lowpass(q_noisy, dt, cutoff_hz, order=order)
    qd = np.gradient(q_f, dt, axis=0)
    qd_f = _zero_phase_lowpass(qd, dt, cutoff_hz, order=order)
    qdd = np.gradient(qd_f, dt, axis=0)
    qdd_f = _zero_phase_lowpass(qdd, dt, cutoff_hz, order=order)
    return q_f, qd_f, qdd_f


def collect_data(
    config: SimulationConfig | None = None,
    model_file: str = "panda.xml",
    model_root: str | Path | None = None,
    dof: int = 7,
    duration: float | None = None,
    dt: float = 0.001,
    use_harmonic: bool = True,
    n_periods: int = 3,
    trajectory_type: str = "sine",
    traj_time_offset: float = 0.0,
    add_noise: bool = False,
    noise_sigma: float | None = None,
    noise_mix_laplace: bool = True,
    noise_laplace_ratio: float = 0.55,
    noise_seed: int | None = 0,
    add_friction: bool = True,
    coulomb_friction=None,
    viscous_friction=None,
    friction_smoothing: float = 0.02,
    add_stribeck: bool = True,
    stribeck_extra_ratio: float = 0.25,
    stribeck_velocity: float = 0.10,
    add_nonlinear_friction: bool = True,
    nonlinear_friction_scale: float = 0.03,
    nonlinear_friction_seed: int | None = 0,
    add_state_noise: bool = True,
    state_noise_sigma_q: float = 8e-4,
    state_noise_seed: int | None = 0,
    state_derivative_mode: str = "butterworth",
    state_filter_cutoff_hz: float = 3.0,
    state_filter_order: int = 4,
    state_vel_ema_alpha: float = 0.25,
    state_acc_ema_alpha: float = 0.2,
    filter_torque: bool = True,
    edge_trim_sec: float | None = None,
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
        traj_time_offset: 轨迹时间相位偏移（秒），用于生成与辨识不同的验证轨迹
        add_noise: 是否在力矩上加随机噪声
        noise_sigma: 高斯分量标准差 (Nm)，默认取 config.torque_noise_sigma
        noise_mix_laplace: 是否再叠加拉普拉斯噪声（更重尾、抖动更“乱”）
        noise_laplace_ratio: 拉普拉斯尺度 = sigma * 该系数
        noise_seed: 随机种子；默认 0（可复现）；显式传 None 则每次不同
        add_friction: 是否叠加摩擦力矩（库仑+粘性，可附加随机非线性项）
        coulomb_friction: 库仑摩擦系数（标量或 shape=(dof,)；None 则默认 0.20 Nm）
        viscous_friction: 粘性摩擦系数（标量或 shape=(dof,)；None 则默认 0.05 Nms/rad）
        friction_smoothing: 库仑摩擦平滑参数，tanh(qd / friction_smoothing)
        add_stribeck: 是否叠加 Stribeck 低速效应
        stribeck_extra_ratio: 静摩擦相对库仑摩擦的增量比例（Fs = Fc * (1 + ratio)）
        stribeck_velocity: Stribeck 特征速度（越小则低速突起区越窄）
        add_nonlinear_friction: 是否叠加随机非线性摩擦项
        nonlinear_friction_scale: 非线性摩擦强度系数（越大越“非线性”）
        nonlinear_friction_seed: 非线性摩擦随机种子；默认 0
        add_state_noise: 是否给状态量添加测量噪声（默认开启）
        state_noise_sigma_q: 关节位置噪声标准差 (rad)
        state_noise_seed: 状态噪声随机种子；默认 0
        state_derivative_mode: 含噪位置下 qd/qdd 估计方式
            - butterworth: 零相位低通 + 差分（离线辨识推荐，默认）
            - ema: 裸差分 + EMA（旧实现，噪声会被严重放大）
            - reference: 位置用测量值（可滤波），速度/加速度用参考轨迹
        state_filter_cutoff_hz: Butterworth 截止频率 (Hz)；应高于轨迹带宽、远低于 Nyquist
        state_filter_order: Butterworth 阶数
        state_vel_ema_alpha: mode=ema 时 qd 的 EMA 平滑系数
        state_acc_ema_alpha: mode=ema 时 qdd 的 EMA 平滑系数
        filter_torque: butterworth/reference 时是否对 τ 做同截止频率零相位滤波（保持线性关系）
        edge_trim_sec: 滤波后丢弃首尾时长（秒）。None 时对 butterworth/reference 自动取 max(1.0, 3/fc)
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
            time_offset=traj_time_offset,
        )

    n_samples = len(t_arr)
    _model_root = getattr(cfg, "model_root", None) or model_root

    model, data = load_mujoco_model(
        model_file=str(cfg.model_path),
        model_root=_model_root,
    )
    disable_contact = bool(getattr(cfg, "disable_contact", True))
    if disable_contact:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    tau_arr = np.zeros((n_samples, dof))
    if verbose:
        traj_name = "harmonic" if use_harmonic else trajectory_type
        off_msg = f", traj_offset={traj_time_offset:.3f}s" if abs(traj_time_offset) > 1e-12 else ""
        print(f"采集数据: {n_samples} 点, dt={dt}s, trajectory={traj_name}{off_msg}")
        print(f"  模型: {cfg.model_path}" + ("（已禁用接触）" if disable_contact else ""))

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
    mode = str(state_derivative_mode).lower().strip()

    if add_state_noise:
        rng_state = np.random.default_rng(state_noise_seed)
        q_noisy = q_true + rng_state.normal(0.0, float(state_noise_sigma_q), size=q_true.shape)
        if mode == "butterworth":
            q_meas, qd_meas, qdd_meas = _estimate_states_butterworth(
                q_noisy,
                dt=dt,
                cutoff_hz=state_filter_cutoff_hz,
                order=state_filter_order,
            )
            if filter_torque:
                tau_arr = _zero_phase_lowpass(
                    tau_arr, dt, state_filter_cutoff_hz, order=state_filter_order
                )
        elif mode == "ema":
            q_meas = q_noisy
            qd_meas = np.gradient(q_meas, dt, axis=0)
            qdd_meas = np.gradient(qd_meas, dt, axis=0)
            qd_meas = _ema_filter(qd_meas, state_vel_ema_alpha)
            qdd_meas = _ema_filter(qdd_meas, state_acc_ema_alpha)
        elif mode == "reference":
            # 实际中闭环跟踪较好时，常用参考 qd/qdd + 测量 q
            q_meas = _zero_phase_lowpass(
                q_noisy, dt, state_filter_cutoff_hz, order=state_filter_order
            )
            qd_meas = qd_true
            qdd_meas = qdd_true
            if filter_torque:
                tau_arr = _zero_phase_lowpass(
                    tau_arr, dt, state_filter_cutoff_hz, order=state_filter_order
                )
        else:
            raise ValueError(
                f"未知 state_derivative_mode={state_derivative_mode!r}，"
                "可选: butterworth | ema | reference"
            )
        if verbose:
            seed_info = f", seed={state_noise_seed}" if state_noise_seed is not None else ", seed=None(每次不同)"
            print(f"  状态加噪: q ~ N(0,{state_noise_sigma_q}^2){seed_info}")
            if mode == "butterworth":
                print(
                    f"  状态估计: Butterworth 零相位低通+差分 "
                    f"(fc={state_filter_cutoff_hz:.2f} Hz, order={state_filter_order}"
                    f"{', 同步滤波 τ' if filter_torque else ''})"
                )
            elif mode == "ema":
                print(
                    f"  状态平滑: EMA(alpha_qd={state_vel_ema_alpha:.3f}, "
                    f"alpha_qdd={state_acc_ema_alpha:.3f})"
                )
            else:
                print(
                    f"  状态估计: reference qd/qdd + 滤波 q "
                    f"(fc={state_filter_cutoff_hz:.2f} Hz"
                    f"{', 同步滤波 τ' if filter_torque else ''})"
                )

    out = {
        "time": t_arr,
        "q": q_meas,
        "qd": qd_meas,
        "qdd": qdd_meas,
        "tau": tau_arr,
        "q_true": q_true,
        "qd_true": qd_true,
        "qdd_true": qdd_true,
    }

    # 滤波边界暂态：丢弃首尾一段，避免图上开头/结尾明显不重合
    need_trim = add_state_noise and mode in ("butterworth", "reference")
    if need_trim:
        if edge_trim_sec is None:
            trim_sec = max(1.0, 3.0 / max(float(state_filter_cutoff_hz), 1e-6))
        else:
            trim_sec = float(edge_trim_sec)
        n_trim = int(round(trim_sec / float(dt)))
        if n_trim > 0:
            out = _trim_edge_samples(out, n_trim)
            if verbose:
                print(f"  已裁剪滤波边界: 首尾各 {trim_sec:.2f}s ({n_trim} 点)")

    return out


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
