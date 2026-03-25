"""同一轨迹上采集力矩 τ 与辨识预测力矩 H·π̂ 对比（与辨识所用样本一致）"""

from __future__ import annotations

import numpy as np


def torque_prediction_rmse(
    tau_measured: np.ndarray,
    H_stack: np.ndarray,
    pi_identified: np.ndarray,
    dof: int = 7,
) -> dict:
    """τ_meas 为 (n, dof)，与 build_H_stack 堆叠顺序一致。"""
    tau_meas = np.asarray(tau_measured)
    n_samples = len(tau_meas)
    pi = np.asarray(pi_identified).ravel()
    tau_pred = (H_stack @ pi).reshape(n_samples, dof)
    err = tau_meas[:, :dof] - tau_pred
    rmse_j = np.sqrt(np.mean(err**2, axis=0))
    rmse_all = float(np.sqrt(np.mean(err**2)))
    return {
        "tau_predicted": tau_pred,
        "rmse_per_joint": rmse_j,
        "rmse_all": rmse_all,
    }


def plot_measured_vs_identified_torque(
    data: dict,
    H_stack: np.ndarray,
    pi_identified: np.ndarray,
    dof: int = 7,
    out_path: str | None = "torque_compare.png",
    show: bool = True,
    verbose: bool = True,
) -> dict:
    """
    在同一组 (q, qd, qdd) 下对比：
    - 采集力矩：仿真 mj_inverse 得到的 τ（及可选噪声）
    - 辨识力矩：τ_hat = H(q,qd,qdd)·π̂

    Args:
        show: True 时弹出窗口（需图形环境）；False 时仅用 Agg 保存文件。
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tau_meas = np.asarray(data["tau"])
    stats = torque_prediction_rmse(tau_meas, H_stack, pi_identified, dof=dof)
    tau_pred = stats["tau_predicted"]
    n_samples = len(tau_meas)
    time = data.get("time")
    if time is None:
        time = np.arange(n_samples, dtype=float)
    else:
        time = np.asarray(time).ravel()

    fig, axes = plt.subplots(dof, 1, sharex=True, figsize=(10, 2.2 * dof))
    if dof == 1:
        axes = [axes]
    for j in range(dof):
        ax = axes[j]
        ax.plot(time, tau_meas[:, j], label=r"$\tau$ measured", linewidth=0.8, alpha=0.9)
        ax.plot(time, tau_pred[:, j], label=r"$\hat{\tau}=H\hat{\pi}$", linewidth=0.8, alpha=0.9)
        ax.set_ylabel(f"j{j + 1} (Nm)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Torque: measured vs identified (same trajectory)", fontsize=11, y=1.002)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show(block=True)
    plt.close(fig)

    if verbose:
        print(f"  力矩对比图: {out_path}" + ("（已显示）" if show else ""))
        print(f"  力矩 RMSE (全关节堆叠): {stats['rmse_all']:.6e} Nm")
        for j in range(dof):
            print(f"    关节 {j + 1} RMSE: {stats['rmse_per_joint'][j]:.6e} Nm")

    return stats
