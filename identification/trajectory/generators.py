"""激励轨迹生成器"""

import numpy as np
from scipy.interpolate import interp1d


def get_paper_fourier_params():
    """论文多谐波参数 (N=5)"""
    a_coeffs = np.array([
        [-0.2031, -0.0699, 0.3076, 0.1269, 0.5864, -0.0253, -0.2773],
        [0.1295, -0.5380, 0.2100, -0.2273, -0.0857, -0.1194, -0.2015],
        [-0.0090, 0.4810, -0.0950, 0.2319, -0.5535, -0.2099, -0.1228],
        [0.2178, 0.5311, -0.0964, -0.2405, 0.4810, -0.0950, 0.2319],
        [-0.7598, 0.2352, 0.5725, -0.5273, 0.6984, -0.1639, -0.0399],
    ])
    b_coeffs = np.array([
        [-0.1136, -0.2437, 0.6600, 0.1820, 0.3898, 0.0401, 0.5491],
        [0.0692, 0.2023, -0.0268, -0.1503, 0.1046, -0.6946, -0.2570],
        [-0.5010, -0.1858, -0.2390, 0.0179, 0.2359, 0.0407, -0.4442],
        [0.1867, -0.2647, 0.4767, -0.2022, -0.0432, -0.5749, 0.0774],
        [0.2357, 0.4252, 0.2726, 0.3693, 0.2448, -0.0698, -0.4833],
    ])
    q0 = np.array([-0.5850, -0.1744, -0.3373, -1.8767, -1.0631, 1.7917, -0.7284])
    return {"a_coeffs": a_coeffs, "b_coeffs": b_coeffs, "q0": q0}


class TrajectoryGenerator:
    """轨迹生成器基类"""

    def __init__(self, dof=7, duration=10.0, dt=0.01, q0=None):
        self.dof = dof
        self.duration = duration
        self.dt = dt
        self.n_samples = int(duration / dt) + 1
        self.time = np.linspace(0, duration, self.n_samples)
        self.q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])[:dof]
        self.q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])[:dof]
        self.qd_max = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])[:dof]
        default_q0 = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0])[:dof]
        self.q0 = np.array(q0) if q0 is not None else default_q0

    def generate(self):
        raise NotImplementedError

    def _clip_to_limits(self, q, qd=None):
        q_clipped = np.clip(q, self.q_min, self.q_max)
        if qd is not None:
            return q_clipped, np.clip(qd, -self.qd_max, self.qd_max)
        return q_clipped


class SinusoidalTrajectory(TrajectoryGenerator):
    def __init__(self, dof=7, duration=10.0, dt=0.01, amplitudes=None, frequencies=None, phase_offsets=None):
        super().__init__(dof, duration, dt)
        self.amplitudes = np.array(amplitudes) if amplitudes is not None else 0.3 * (self.q_max - self.q_min)
        self.frequencies = np.array(frequencies) if frequencies is not None else np.array([0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25])[:dof]
        self.phase_offsets = np.array(phase_offsets) if phase_offsets is not None else np.linspace(0, 2 * np.pi, dof, endpoint=False)

    def generate(self):
        q = self.q0 + self.amplitudes * np.sin(2 * np.pi * self.frequencies * self.time[:, None] + self.phase_offsets)
        qd = self.amplitudes * self.frequencies * 2 * np.pi * np.cos(2 * np.pi * self.frequencies * self.time[:, None] + self.phase_offsets)
        qdd = -self.amplitudes * (self.frequencies * 2 * np.pi) ** 2 * np.sin(2 * np.pi * self.frequencies * self.time[:, None] + self.phase_offsets)
        for i in range(self.n_samples):
            q[i], qd[i] = self._clip_to_limits(q[i], qd[i])
        return {"time": self.time, "q": q, "qd": qd, "qdd": qdd}


class PolynomialTrajectory(TrajectoryGenerator):
    def __init__(self, dof=7, duration=10.0, dt=0.01, waypoints=None, waypoint_times=None):
        super().__init__(dof, duration, dt)
        if waypoints is None:
            n_wp = 5
            waypoints = [self.q0 + 0.5 * (self.q_max - self.q_min) * (np.random.random(self.dof) - 0.5) for _ in range(n_wp)]
            waypoints.insert(0, self.q0)
            waypoints.append(self.q0)
        self.waypoints = np.array(waypoints)
        self.waypoint_times = np.linspace(0, duration, len(self.waypoints)) if waypoint_times is None else np.array(waypoint_times)

    def generate(self):
        q = np.zeros((self.n_samples, self.dof))
        qd = np.zeros((self.n_samples, self.dof))
        qdd = np.zeros((self.n_samples, self.dof))
        for j in range(self.dof):
            f = interp1d(self.waypoint_times, self.waypoints[:, j], kind="cubic", fill_value="extrapolate")
            q[:, j] = f(self.time)
            qd[:, j] = np.gradient(q[:, j], self.dt)
            qdd[:, j] = np.gradient(qd[:, j], self.dt)
        for i in range(self.n_samples):
            q[i], qd[i] = self._clip_to_limits(q[i], qd[i])
        return {"time": self.time, "q": q, "qd": qd, "qdd": qdd}


class RandomTrajectory(TrajectoryGenerator):
    def __init__(self, dof=7, duration=10.0, dt=0.01, frequency_range=(0.05, 0.5), amplitude_ratio=0.3):
        super().__init__(dof, duration, dt)
        self.freq_range = frequency_range
        self.amp_ratio = amplitude_ratio

    def generate(self):
        freqs = np.random.uniform(*self.freq_range, self.dof)
        phases = np.random.uniform(0, 2 * np.pi, self.dof)
        amps = self.amp_ratio * (self.q_max - self.q_min) * np.random.uniform(0.5, 1.0, self.dof)
        phase = 2 * np.pi * freqs * self.time[:, None] + phases
        q = self.q0 + amps * np.sin(phase)
        qd = amps * freqs * 2 * np.pi * np.cos(phase)
        qdd = -amps * (freqs * 2 * np.pi) ** 2 * np.sin(phase)
        for i in range(self.n_samples):
            q[i], qd[i] = self._clip_to_limits(q[i], qd[i])
        return {"time": self.time, "q": q, "qd": qd, "qdd": qdd}


class HarmonicExcitationTrajectory(TrajectoryGenerator):
    def __init__(self, dof=7, duration=10.0, dt=0.01, n_harmonics=5, omega_f=0.15 * np.pi,
                 a_coeffs=None, b_coeffs=None, q0=None, scale_to_limits=True, random_seed=None):
        paper = get_paper_fourier_params()
        if a_coeffs is None and b_coeffs is None:
            a_coeffs = paper["a_coeffs"]
            b_coeffs = paper["b_coeffs"]
            q0 = q0 or paper["q0"]
            n_harmonics = a_coeffs.shape[0]
        super().__init__(dof, duration, dt, q0=q0)
        self.n_harmonics = n_harmonics
        self.omega_f = omega_f
        self.scale_to_limits = scale_to_limits
        rng = np.random.default_rng(random_seed)
        if a_coeffs is None or b_coeffs is None:
            base_amp = 0.2 * self.qd_max
            a_coeffs = rng.uniform(-1, 1, (n_harmonics, dof)) * base_amp
            b_coeffs = rng.uniform(-1, 1, (n_harmonics, dof)) * base_amp
        self.a_coeffs = np.array(a_coeffs)[:n_harmonics, :dof]
        self.b_coeffs = np.array(b_coeffs)[:n_harmonics, :dof]

    def _compute(self, a, b):
        q = np.zeros((self.n_samples, self.dof))
        qd = np.zeros((self.n_samples, self.dof))
        qdd = np.zeros((self.n_samples, self.dof))
        k_arr = np.arange(1, self.n_harmonics + 1, dtype=float)
        for i in range(self.n_samples):
            t = self.time[i]
            kwt = np.outer(k_arr, np.ones(self.dof)) * (self.omega_f * t)
            sk, ck = np.sin(kwt), np.cos(kwt)
            q[i] = self.q0 + np.sum((a / (k_arr[:, None] * self.omega_f)) * sk - (b / (k_arr[:, None] * self.omega_f)) * ck, axis=0)
            qd[i] = np.sum(a * ck + b * sk, axis=0)
            qdd[i] = np.sum((k_arr[:, None] * self.omega_f) * (-a * sk + b * ck), axis=0)
        return q, qd, qdd

    def _scale(self, a, b):
        q, _, _ = self._compute(a, b)
        scale = np.ones(self.dof)
        for j in range(self.dof):
            pos = max(q[:, j].max() - self.q0[j], 1e-9)
            neg = max(self.q0[j] - q[:, j].min(), 1e-9)
            scale[j] = min(1.0, (self.q_max[j] - self.q0[j]) / pos if pos > 1e-9 else 1,
                          (self.q0[j] - self.q_min[j]) / neg if neg > 1e-9 else 1)
        return a * scale[None, :], b * scale[None, :]

    def generate(self):
        a, b = self.a_coeffs.copy(), self.b_coeffs.copy()
        if self.scale_to_limits:
            a, b = self._scale(a, b)
        q, qd, qdd = self._compute(a, b)
        for i in range(self.n_samples):
            q[i], qd[i] = self._clip_to_limits(q[i], qd[i])
        return {"time": self.time, "q": q, "qd": qd, "qdd": qdd}


def generate_trajectory(trajectory_type="sinusoidal", **kwargs):
    types = {
        "sinusoidal": SinusoidalTrajectory,
        "polynomial": PolynomialTrajectory,
        "random": RandomTrajectory,
        "harmonic": HarmonicExcitationTrajectory,
    }
    if trajectory_type == "harmonic_paper":
        p = get_paper_fourier_params()
        kwargs.setdefault("a_coeffs", p["a_coeffs"])
        kwargs.setdefault("b_coeffs", p["b_coeffs"])
        kwargs.setdefault("q0", p["q0"])
        trajectory_type = "harmonic"
    gen = types[trajectory_type](**kwargs)
    return gen.generate()
