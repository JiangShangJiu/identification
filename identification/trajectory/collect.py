"""为 collect_data 提供轨迹的便捷接口"""

import numpy as np

HARMONIC_PERIOD = 2 * np.pi / (0.15 * np.pi)


def make_trajectory_for_collect(
    use_harmonic: bool = True,
    duration: float | None = None,
    dt: float = 0.001,
    n_periods: int = 3,
    trajectory_type: str = "sine",
    dof: int = 7,
    time_offset: float = 0.0,
) -> tuple:
    """
    生成采集用轨迹，返回 (time, q, qd, qdd)

    trajectory_type: sine|polynomial|random（仅 use_harmonic=False 时有效）
    time_offset: 谐波/正弦的时间相位偏移（秒）
    """
    if duration is None:
        duration = n_periods * HARMONIC_PERIOD if use_harmonic else 10.0

    if use_harmonic:
        from .generators import HarmonicExcitationTrajectory

        gen = HarmonicExcitationTrajectory(
            dof=dof, duration=duration, dt=dt, time_offset=time_offset
        )
        traj = gen.generate()
        return traj["time"], traj["q"], traj["qd"], traj["qdd"]

    from .generators import SinusoidalTrajectory, generate_trajectory

    if trajectory_type == "sine":
        # 各关节不同频率的正弦；与论文多谐波激励结构明显不同
        gen = SinusoidalTrajectory(dof=dof, duration=duration, dt=dt)
        if abs(time_offset) > 1e-12:
            gen.phase_offsets = np.asarray(gen.phase_offsets, dtype=float) + (
                2.0 * np.pi * np.asarray(gen.frequencies, dtype=float) * float(time_offset)
            )
        traj = gen.generate()
        return traj["time"], traj["q"], traj["qd"], traj["qdd"]

    if trajectory_type in ("polynomial", "random"):
        traj = generate_trajectory(trajectory_type, dof=dof, duration=duration, dt=dt)
        if abs(time_offset) > 1e-12:
            n = len(traj["time"])
            shift = int(round(float(time_offset) / dt)) % n
            if shift:
                for key in ("q", "qd", "qdd"):
                    traj[key] = np.roll(traj[key], -shift, axis=0)
        return traj["time"], traj["q"], traj["qd"], traj["qdd"]

    raise ValueError(f"未知 trajectory_type={trajectory_type!r}")
