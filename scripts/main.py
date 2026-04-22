#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键主流程：采集 -> 辨识 -> 对比"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from identification.simulation import collect_data, load_data, save_data
from identification.dynamics import get_regressor, get_base_params_info, build_H_stack
from identification.dynamics.base_params import convert_full_to_base_params
from identification.ground_truth import extract_ground_truth
from identification.solver import identify
from identification.compare import compare


def main():
    parser = argparse.ArgumentParser(description="Franka Panda 动力学参数辨识")
    parser.add_argument("--harmonic", dest="harmonic", action="store_true", default=True, help="多谐波激励轨迹（默认开启）")
    parser.add_argument("--no-harmonic", dest="harmonic", action="store_false", help="关闭多谐波激励轨迹")
    parser.add_argument("--noise", dest="noise", action="store_true", default=True, help="力矩加随机噪声（默认开启：高斯+拉普拉斯混合，更重尾）")
    parser.add_argument("--no-noise", dest="noise", action="store_false", help="关闭力矩随机噪声")
    parser.add_argument("--noise-sigma", type=float, default=None, help="高斯分量标准差 (Nm)，默认见 SimulationConfig")
    parser.add_argument("--noise-seed", type=int, default=None, help="噪声随机种子；不设则每次采集噪声不同")
    parser.add_argument("--noise-gaussian-only", action="store_true", help="仅加高斯白噪声，不混合拉普拉斯")
    parser.add_argument("--state-noise", dest="state_noise", action="store_true", default=True, help="状态测量噪声（默认开启：q 加噪后差分得到 qd/qdd）")
    parser.add_argument("--no-state-noise", dest="state_noise", action="store_false", help="关闭状态测量噪声，使用理想 q/qd/qdd")
    parser.add_argument("--state-noise-sigma-q", type=float, default=8e-4, help="位置测量噪声标准差 (rad)")
    parser.add_argument("--state-noise-seed", type=int, default=None, help="状态噪声随机种子")
    parser.add_argument("--state-vel-ema-alpha", type=float, default=0.25, help="qd 的 EMA 平滑系数")
    parser.add_argument("--state-acc-ema-alpha", type=float, default=0.2, help="qdd 的 EMA 平滑系数")
    parser.add_argument("--no-friction", action="store_true", help="关闭摩擦力矩叠加（默认开启）")
    parser.add_argument("--coulomb-friction", type=float, default=None, help="库仑摩擦系数 (Nm)，默认 0.20")
    parser.add_argument("--viscous-friction", type=float, default=None, help="粘性摩擦系数 (Nms/rad)，默认 0.05")
    parser.add_argument("--no-stribeck", action="store_true", help="关闭 Stribeck 低速效应")
    parser.add_argument("--stribeck-extra-ratio", type=float, default=0.25, help="Stribeck 增量比例 Fs/Fc - 1")
    parser.add_argument("--stribeck-velocity", type=float, default=0.10, help="Stribeck 特征速度 (rad/s)")
    parser.add_argument("--no-nonlinear-friction", action="store_true", help="仅使用库仑+粘性摩擦，不加随机非线性项")
    parser.add_argument("--nonlinear-friction-scale", type=float, default=0.03, help="随机非线性摩擦强度")
    parser.add_argument("--nonlinear-friction-seed", type=int, default=None, help="随机非线性摩擦随机种子")
    parser.add_argument("--save-data", type=str, default=None, help="保存采集数据")
    parser.add_argument("--load-data", type=str, default=None, help="加载已有数据")
    parser.add_argument("--model-root", type=str, default=None, help="MuJoCo 模型根目录 (如 learn_robot/mujoco/franka_emika_panda)")
    parser.add_argument("--sympybotics-path", type=str, default=None, help="SymPyBotics 路径 (如 learn_robot/model/SymPyBotics)")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--n-periods", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--plot", action="store_true", default=True, help="绘图（默认开启）：轨迹 q/qd/qdd + 力矩对比 H·π̂，共两张图")
    parser.add_argument("--plot-traj-out", type=str, default="trajectory_states.png", help="轨迹状态图保存路径（q/qd/qdd）")
    parser.add_argument("--plot-out", type=str, default="torque_compare.png", help="力矩对比图保存路径")
    parser.add_argument("--plot-no-show", action="store_true", help="不弹窗，仅保存图片（对 --plot-traj-out 和 --plot-out 都生效）")
    args = parser.parse_args()

    print("=" * 60)
    print("Franka Panda 动力学参数辨识")
    print("=" * 60)

    if args.load_data:
        print("\n[1] 加载数据:", args.load_data)
        data = load_data(args.load_data)
    else:
        print("\n[1] 采集数据")
        from identification.simulation.config import SimulationConfig
        cfg = SimulationConfig(model_root=args.model_root)
        # 若在 learn_robot 下运行，可自动检测
        if args.model_root is None:
            _lr = Path(__file__).resolve().parents[2] / "learn_robot" / "mujoco" / "franka_emika_panda"
            if _lr.exists():
                cfg.model_root = str(_lr)
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
        if args.save_data:
            save_data(data, args.save_data)
            print(f"  已保存: {args.save_data}")

    print("\n[2] 模型建立 + 辨识")
    reg_func, n_params = get_regressor(use_base=True, sympybotics_path=args.sympybotics_path)
    H_stack, tau_stack = build_H_stack(reg_func, data, n_params, dof=7)
    result = identify(H_stack, tau_stack, verbose=True)

    print("\n[3] 真实参数 (MuJoCo XML)")
    xml_path = (Path(args.model_root) / "panda.xml") if args.model_root else None
    if xml_path and Path(xml_path).exists():
        gt = extract_ground_truth(xml_path=xml_path, verbose=True)
    else:
        gt = extract_ground_truth(verbose=True)
    pi_true_full = gt["dynparms_array"]

    base_info = get_base_params_info()
    if base_info is not None and n_params < len(pi_true_full):
        pi_true = convert_full_to_base_params(pi_true_full, base_info)
        pi_id = result["pi_identified"][: len(pi_true)]
    else:
        pi_true = pi_true_full[:n_params] if len(pi_true_full) >= n_params else pi_true_full
        pi_id = result["pi_identified"][: len(pi_true)]

    print("\n[4] 对比")
    comp = compare(pi_id, pi_true, verbose=True)
    print("\n" + "=" * 60)
    print(f"RMSE: {comp['rmse']:.6e}")
    print(f"max relative error: {comp['max_rel_error']:.2%}")
    print(f"mean relative error: {comp['mean_rel_error']:.2%}")
    print("=" * 60)

    if args.plot:
        from identification.torque_plot import (
            plot_trajectory_states,
            plot_measured_vs_identified_torque,
        )

        print("\n[5] 轨迹状态图 (q / qd / qdd，同一样本)")
        plot_trajectory_states(
            data,
            dof=7,
            out_path=args.plot_traj_out,
            show=not args.plot_no_show,
            verbose=True,
        )

        print("\n[6] 力矩对比图 (τ_measured vs H·π̂，同一样本)")
        plot_measured_vs_identified_torque(
            data,
            H_stack,
            result["pi_identified"],
            dof=7,
            out_path=args.plot_out,
            show=not args.plot_no_show,
            verbose=True,
        )

        if not args.plot_no_show:
            # 两张图都已非阻塞弹出，这里统一阻塞等待，避免窗口一闪而过。
            import matplotlib.pyplot as plt

            plt.show(block=True)


if __name__ == "__main__":
    main()
