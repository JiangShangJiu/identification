#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键主流程：采集(辨识轨) -> 辨识 -> 采集(验证轨) -> 对比"""

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
from identification.torque_plot import torque_prediction_rmse


def _collect_kwargs(
    args,
    *,
    traj_time_offset: float = 0.0,
    use_harmonic: bool | None = None,
    trajectory_type: str | None = None,
    duration: float | None = None,
    n_periods: int | None = None,
) -> dict:
    return dict(
        duration=args.duration if duration is None else duration,
        dt=args.dt,
        use_harmonic=args.harmonic if use_harmonic is None else use_harmonic,
        n_periods=args.n_periods if n_periods is None else n_periods,
        trajectory_type=("sine" if trajectory_type is None else trajectory_type),
        traj_time_offset=traj_time_offset,
        add_noise=args.noise,
        noise_sigma=args.noise_sigma,
        noise_mix_laplace=not args.noise_gaussian_only,
        noise_seed=args.noise_seed,
        add_state_noise=args.state_noise,
        state_noise_sigma_q=args.state_noise_sigma_q,
        state_noise_seed=args.state_noise_seed,
        state_derivative_mode=args.state_derivative_mode,
        state_filter_cutoff_hz=args.state_filter_cutoff,
        state_filter_order=args.state_filter_order,
        state_vel_ema_alpha=args.state_vel_ema_alpha,
        state_acc_ema_alpha=args.state_acc_ema_alpha,
        filter_torque=not args.no_filter_torque,
        edge_trim_sec=args.edge_trim_sec,
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


def main():
    parser = argparse.ArgumentParser(description="Franka Panda 动力学参数辨识")
    parser.add_argument("--harmonic", dest="harmonic", action="store_true", default=True, help="多谐波激励轨迹（默认开启）")
    parser.add_argument("--no-harmonic", dest="harmonic", action="store_false", help="关闭多谐波激励轨迹")
    parser.add_argument("--noise", dest="noise", action="store_true", default=True, help="力矩加随机噪声（默认开启：高斯+拉普拉斯混合，更重尾）")
    parser.add_argument("--no-noise", dest="noise", action="store_false", help="关闭力矩随机噪声")
    parser.add_argument("--noise-sigma", type=float, default=None, help="高斯分量标准差 (Nm)，默认见 SimulationConfig")
    parser.add_argument("--noise-seed", type=int, default=0, help="力矩噪声随机种子（默认 0，可复现）")
    parser.add_argument("--noise-gaussian-only", action="store_true", help="仅加高斯白噪声，不混合拉普拉斯")
    parser.add_argument("--state-noise", dest="state_noise", action="store_true", default=True, help="状态测量噪声（默认开启：q 加噪后估计 qd/qdd）")
    parser.add_argument("--no-state-noise", dest="state_noise", action="store_false", help="关闭状态测量噪声，使用理想 q/qd/qdd")
    parser.add_argument("--state-noise-sigma-q", type=float, default=8e-4, help="位置测量噪声标准差 (rad)")
    parser.add_argument("--state-noise-seed", type=int, default=0, help="状态噪声随机种子（默认 0）")
    parser.add_argument(
        "--state-derivative-mode",
        type=str,
        default="butterworth",
        choices=["butterworth", "ema", "reference"],
        help="含噪位置下 qd/qdd 估计：butterworth(默认)/ema/reference",
    )
    parser.add_argument("--state-filter-cutoff", type=float, default=3.0, help="Butterworth 截止频率 (Hz)")
    parser.add_argument("--state-filter-order", type=int, default=4, help="Butterworth 阶数")
    parser.add_argument("--no-filter-torque", action="store_true", help="不对 τ 做与状态同截止的零相位滤波")
    parser.add_argument(
        "--edge-trim-sec",
        type=float,
        default=None,
        help="滤波后裁剪首尾时长(秒)；默认 max(1, 3/fc)，用于消除开头/结尾不匹配",
    )
    parser.add_argument("--state-vel-ema-alpha", type=float, default=0.25, help="mode=ema 时 qd 的 EMA 平滑系数")
    parser.add_argument("--state-acc-ema-alpha", type=float, default=0.2, help="mode=ema 时 qdd 的 EMA 平滑系数")
    parser.add_argument("--no-friction", action="store_true", help="关闭摩擦力矩叠加（默认开启）")
    parser.add_argument("--coulomb-friction", type=float, default=None, help="库仑摩擦系数 (Nm)，默认 0.20")
    parser.add_argument("--viscous-friction", type=float, default=None, help="粘性摩擦系数 (Nms/rad)，默认 0.05")
    parser.add_argument("--no-stribeck", action="store_true", help="关闭 Stribeck 低速效应")
    parser.add_argument("--stribeck-extra-ratio", type=float, default=0.25, help="Stribeck 增量比例 Fs/Fc - 1")
    parser.add_argument("--stribeck-velocity", type=float, default=0.10, help="Stribeck 特征速度 (rad/s)")
    parser.add_argument("--no-nonlinear-friction", action="store_true", help="仅使用库仑+粘性摩擦，不加随机非线性项")
    parser.add_argument("--nonlinear-friction-scale", type=float, default=0.03, help="随机非线性摩擦强度")
    parser.add_argument("--nonlinear-friction-seed", type=int, default=0, help="非线性摩擦随机种子（默认 0）")
    parser.add_argument("--save-data", type=str, default=None, help="保存辨识轨数据")
    parser.add_argument("--save-val-data", type=str, default=None, help="保存验证轨数据")
    parser.add_argument("--load-data", type=str, default=None, help="加载已有辨识轨数据")
    parser.add_argument("--load-val-data", type=str, default=None, help="加载已有验证轨数据；不设则现场采集验证轨")
    parser.add_argument("--model-root", type=str, default=None, help="MuJoCo 模型根目录 (如 learn_robot/mujoco/franka_emika_panda)")
    parser.add_argument("--sympybotics-path", type=str, default=None, help="SymPyBotics 路径 (如 learn_robot/model/SymPyBotics)")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--n-periods", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument(
        "--val-traj",
        type=str,
        default="sine",
        choices=["sine", "harmonic", "polynomial", "random"],
        help="验证轨迹类型（默认 sine，与辨识多谐波明显不同）",
    )
    parser.add_argument(
        "--val-duration",
        type=float,
        default=None,
        help="验证轨时长（秒）；默认与辨识轨同长（谐波 3 周期约 40s）",
    )
    parser.add_argument(
        "--val-traj-offset",
        type=float,
        default=0.0,
        help="验证轨时间相位偏移（秒）",
    )
    parser.add_argument("--no-val", action="store_true", help="不采验证轨，力矩图仍用辨识轨（不推荐）")
    parser.add_argument("--plot", action="store_true", default=True, help="绘图（默认开启）：辨识/验证轨状态 + 力矩对比")
    parser.add_argument(
        "--plot-traj-id-out",
        type=str,
        default="trajectory_states_id.png",
        help="辨识轨（harmonic）状态图保存路径 q/qd/qdd",
    )
    parser.add_argument(
        "--plot-traj-out",
        type=str,
        default="trajectory_states_val.png",
        help="验证轨（默认 sine）状态图保存路径 q/qd/qdd",
    )
    parser.add_argument("--plot-out", type=str, default="torque_compare.png", help="验证轨力矩对比图保存路径")
    parser.add_argument(
        "--plot-id-out",
        type=str,
        default="torque_compare_id.png",
        help="辨识轨力矩对比图保存路径（训练集上再测一次）",
    )
    parser.add_argument("--plot-no-show", action="store_true", help="不弹窗，仅保存图片")
    args = parser.parse_args()

    print("=" * 60)
    print("Franka Panda 动力学参数辨识")
    print("=" * 60)

    from identification.simulation.config import SimulationConfig
    cfg = SimulationConfig(model_root=args.model_root)
    if args.model_root is None:
        _lr = Path(__file__).resolve().parents[2] / "learn_robot" / "mujoco" / "franka_emika_panda"
        if _lr.exists():
            cfg.model_root = str(_lr)

    if args.load_data:
        print("\n[1] 加载辨识轨:", args.load_data)
        data_train = load_data(args.load_data)
    else:
        print("\n[1] 采集辨识轨")
        data_train = collect_data(config=cfg, **_collect_kwargs(args, traj_time_offset=0.0))
        if args.save_data:
            save_data(data_train, args.save_data)
            print(f"  已保存辨识轨: {args.save_data}")

    print("\n[2] 模型建立 + 辨识（仅用辨识轨）")
    reg_func, n_params = get_regressor(use_base=True, sympybotics_path=args.sympybotics_path)
    H_train, tau_train = build_H_stack(reg_func, data_train, n_params, dof=7)
    result = identify(H_train, tau_train, verbose=True)
    train_fit = torque_prediction_rmse(data_train["tau"], H_train, result["pi_identified"], dof=7)
    print(f"  辨识轨拟合 RMSE: {train_fit['rmse_all']:.6e} Nm")

    if args.no_val:
        data_val = data_train
        H_val = H_train
        print("\n[2b] 跳过验证轨（--no-val），力矩评估仍用辨识轨")
    elif args.load_val_data:
        print("\n[2b] 加载验证轨:", args.load_val_data)
        data_val = load_data(args.load_val_data)
        H_val, _ = build_H_stack(reg_func, data_val, n_params, dof=7)
    else:
        val_is_harmonic = args.val_traj == "harmonic"
        val_duration = args.val_duration
        if val_duration is None:
            if args.duration is not None:
                val_duration = args.duration
            elif args.harmonic:
                from identification.trajectory.collect import HARMONIC_PERIOD

                val_duration = args.n_periods * HARMONIC_PERIOD  # 与辨识轨同长，默认约 40s
            else:
                val_duration = 40.0
        print(
            f"\n[2b] 采集验证轨: type={args.val_traj}"
            f"{'' if val_is_harmonic else ' (各关节单频正弦，非辨识多谐波累加)'}"
            f", duration={val_duration}, offset={args.val_traj_offset:.3f}s"
        )
        val_kw = _collect_kwargs(
            args,
            traj_time_offset=args.val_traj_offset,
            use_harmonic=val_is_harmonic,
            trajectory_type=args.val_traj if not val_is_harmonic else "sine",
            duration=val_duration,
        )
        if args.noise_seed is not None:
            val_kw["noise_seed"] = int(args.noise_seed) + 1
        if args.state_noise_seed is not None:
            val_kw["state_noise_seed"] = int(args.state_noise_seed) + 1
        if args.nonlinear_friction_seed is not None:
            val_kw["nonlinear_friction_seed"] = int(args.nonlinear_friction_seed) + 1
        data_val = collect_data(config=cfg, **val_kw)
        if args.save_val_data:
            save_data(data_val, args.save_val_data)
            print(f"  已保存验证轨: {args.save_val_data}")
        H_val, _ = build_H_stack(reg_func, data_val, n_params, dof=7)

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

    print("\n[4] 参数对比")
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

        print("\n[5] 辨识轨状态图 (q / qd / qdd，harmonic)")
        plot_trajectory_states(
            data_train,
            dof=7,
            out_path=args.plot_traj_id_out,
            show=not args.plot_no_show,
            verbose=True,
            title="Identification trajectory states (harmonic)",
        )

        print("\n[6] 辨识轨力矩对比（训练集再测）")
        plot_measured_vs_identified_torque(
            data_train,
            H_train,
            result["pi_identified"],
            dof=7,
            out_path=args.plot_id_out,
            show=not args.plot_no_show,
            verbose=True,
            title="Torque: measured vs identified (identification: harmonic)",
        )

        if args.no_val:
            print("\n[7] 已跳过验证轨（--no-val）")
        else:
            print(f"\n[7] 验证轨状态图 (q / qd / qdd，{args.val_traj})")
            plot_trajectory_states(
                data_val,
                dof=7,
                out_path=args.plot_traj_out,
                show=not args.plot_no_show,
                verbose=True,
                title=f"Validation trajectory states ({args.val_traj})",
            )

            print(f"\n[8] 验证轨力矩对比 (τ_measured vs H·π̂，{args.val_traj})")
            plot_measured_vs_identified_torque(
                data_val,
                H_val,
                result["pi_identified"],
                dof=7,
                out_path=args.plot_out,
                show=not args.plot_no_show,
                verbose=True,
                title=f"Torque: measured vs identified (validation: {args.val_traj})",
            )
            val_fit = torque_prediction_rmse(data_val["tau"], H_val, result["pi_identified"], dof=7)
            print(
                f"\n  辨识轨力矩 RMSE: {train_fit['rmse_all']:.6e} Nm\n"
                f"  验证轨力矩 RMSE: {val_fit['rmse_all']:.6e} Nm"
            )

        if not args.plot_no_show:
            import matplotlib.pyplot as plt

            plt.show(block=True)


if __name__ == "__main__":
    main()
