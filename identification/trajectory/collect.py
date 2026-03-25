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
) -> tuple:
    """
    生成采集用轨迹，返回 (time, q, qd, qdd)

    trajectory_type: sine|polynomial|random（仅 use_harmonic=False 时有效）
    """
    if duration is None:
        duration = n_periods * HARMONIC_PERIOD if use_harmonic else 10.0

    if use_harmonic:
        from .generators import HarmonicExcitationTrajectory
        gen = HarmonicExcitationTrajectory(dof=dof, duration=duration, dt=dt)
        traj = gen.generate()
        return traj["time"], traj["q"], traj["qd"], traj["qdd"]

    from .generators import generate_trajectory
    if trajectory_type in ("polynomial", "random"):
        traj = generate_trajectory(trajectory_type, dof=dof, duration=duration, dt=dt)
        return traj["time"], traj["q"], traj["qd"], traj["qdd"]
    else:
        # 简单正弦
        t = np.arange(0, duration + dt * 0.5, dt)
        omega = 0.3 * 2 * np.pi
        q0 = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0])[:dof]
        amp = np.array([0.3] * dof)
        phase = np.linspace(0, 2 * np.pi, dof, endpoint=False)
        q = q0 + amp * np.sin(omega * t[:, None] + phase)
        qd = amp * omega * np.cos(omega * t[:, None] + phase)
        qdd = -amp * omega**2 * np.sin(omega * t[:, None] + phase)
        return t, q, qd, qdd

    return traj["time"], traj["q"], traj["qd"], traj["qdd"]
