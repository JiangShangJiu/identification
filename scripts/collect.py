#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""采集辨识数据"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from identification.simulation import collect_data, save_data
from identification.simulation.config import SimulationConfig


def main():
    parser = argparse.ArgumentParser(description="采集 MuJoCo 辨识数据")
    parser.add_argument("--harmonic", action="store_true", help="多谐波激励轨迹")
    parser.add_argument("--noise", action="store_true", help="力矩加随机噪声（高斯+拉普拉斯混合）")
    parser.add_argument("--noise-sigma", type=float, default=None, help="高斯分量标准差 (Nm)")
    parser.add_argument("--noise-seed", type=int, default=None, help="噪声随机种子")
    parser.add_argument("--noise-gaussian-only", action="store_true", help="仅高斯白噪声")
    parser.add_argument("--state-noise", dest="state_noise", action="store_true", default=True, help="状态测量噪声（默认开启：q 加噪后差分得到 qd/qdd）")
    parser.add_argument("--no-state-noise", dest="state_noise", action="store_false", help="关闭状态测量噪声，使用理想 q/qd/qdd")
    parser.add_argument("--state-noise-sigma-q", type=float, default=8e-4, help="位置测量噪声标准差 (rad)")
    parser.add_argument("--state-noise-seed", type=int, default=None, help="状态噪声随机种子")
    parser.add_argument("--state-vel-ema-alpha", type=float, default=0.25, help="qd 的 EMA 平滑系数")
    parser.add_argument("--state-acc-ema-alpha", type=float, default=0.2, help="qdd 的 EMA 平滑系数")
    parser.add_argument("--no-friction", action="store_true", help="关闭摩擦力矩叠加")
    parser.add_argument("--coulomb-friction", type=float, default=None, help="库仑摩擦系数 (Nm)，默认 0.20")
    parser.add_argument("--viscous-friction", type=float, default=None, help="粘性摩擦系数 (Nms/rad)，默认 0.05")
    parser.add_argument("--no-stribeck", action="store_true", help="关闭 Stribeck 低速效应")
    parser.add_argument("--stribeck-extra-ratio", type=float, default=0.25, help="Stribeck 增量比例 Fs/Fc - 1")
    parser.add_argument("--stribeck-velocity", type=float, default=0.10, help="Stribeck 特征速度 (rad/s)")
    parser.add_argument("--no-nonlinear-friction", action="store_true", help="仅使用库仑+粘性摩擦，不加随机非线性项")
    parser.add_argument("--nonlinear-friction-scale", type=float, default=0.03, help="随机非线性摩擦强度")
    parser.add_argument("--nonlinear-friction-seed", type=int, default=None, help="随机非线性摩擦随机种子")
    parser.add_argument("--model-root", type=str, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--n-periods", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--save", "-o", type=str, required=True, help="输出 .npz 路径")
    args = parser.parse_args()

    cfg = SimulationConfig(model_root=args.model_root)
    data = collect_data(
        config=cfg,
        duration=args.duration,
        dt=args.dt,
        use_harmonic=args.harmonic,
        n_periods=args.n_periods,
        add_noise=args.noise,
        noise_sigma=args.noise_sigma,
        noise_mix_laplace=not args.noise_gaussian_only,
        noise_seed=args.noise_seed,
        add_state_noise=args.state_noise,
        state_noise_sigma_q=args.state_noise_sigma_q,
        state_noise_seed=args.state_noise_seed,
        state_vel_ema_alpha=args.state_vel_ema_alpha,
        state_acc_ema_alpha=args.state_acc_ema_alpha,
        add_friction=not args.no_friction,
        coulomb_friction=args.coulomb_friction,
        viscous_friction=args.viscous_friction,
        add_stribeck=not args.no_stribeck,
        stribeck_extra_ratio=args.stribeck_extra_ratio,
        stribeck_velocity=args.stribeck_velocity,
        add_nonlinear_friction=not args.no_nonlinear_friction,
        nonlinear_friction_scale=args.nonlinear_friction_scale,
        nonlinear_friction_seed=args.nonlinear_friction_seed,
        verbose=True,
    )
    save_data(data, args.save)
    print(f"已保存: {args.save}")


if __name__ == "__main__":
    main()
