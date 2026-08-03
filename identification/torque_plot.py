"""轨迹状态与力矩对比绘图（辨识 / 验证）"""

from __future__ import annotations

import numpy as np


# 关节配色：偏工程感的离散色，避开默认蓝橙堆叠
_JOINT_COLORS = [
    "#1f4e79",
    "#c45c26",
    "#2a9d8f",
    "#6d597a",
    "#b56576",
    "#4a7c59",
    "#d4a373",
]
_MEAS_COLOR = "#1f4e79"
_PRED_COLOR = "#c45c26"


def _apply_plot_style(plt):
    plt.rcParams.update(
        {
            "figure.facecolor": "#f7f5f2",
            "axes.facecolor": "#fffcf8",
            "axes.edgecolor": "#cfc8bf",
            "axes.labelcolor": "#2b2b2b",
            "axes.titlecolor": "#1a1a1a",
            "axes.grid": True,
            "grid.color": "#e6e0d8",
            "grid.linewidth": 0.8,
            "xtick.color": "#4a4a4a",
            "ytick.color": "#4a4a4a",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "lines.solid_capstyle": "round",
            "savefig.facecolor": "#f7f5f2",
            "savefig.bbox": "tight",
            "savefig.dpi": 180,
        }
    )


def _style_ax(ax, *, hide_top_right: bool = True):
    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b8b0a6")
    ax.spines["bottom"].set_color("#b8b0a6")
    ax.tick_params(length=3, width=0.8)


def _get_time(data: dict, n_samples: int) -> np.ndarray:
    time = data.get("time")
    if time is None:
        return np.arange(n_samples, dtype=float)
    return np.asarray(time).ravel()


def plot_trajectory_states(
    data: dict,
    dof: int = 7,
    out_path: str | None = "trajectory_states.png",
    show: bool = True,
    verbose: bool = True,
    title: str | None = None,
):
    """绘制轨迹 q / qd / qdd。"""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_plot_style(plt)

    q = np.asarray(data["q"])[:, :dof]
    qd = np.asarray(data["qd"])[:, :dof]
    qdd = np.asarray(data["qdd"])[:, :dof]
    time = _get_time(data, len(q))

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11.5, 8.2), constrained_layout=True)
    series = [
        (q, "position $q$ (rad)"),
        (qd, "velocity $\\dot{q}$ (rad/s)"),
        (qdd, "acceleration $\\ddot{q}$ (rad/s$^2$)"),
    ]
    for ax, (arr, ylabel) in zip(axes, series):
        for j in range(dof):
            ax.plot(
                time,
                arr[:, j],
                color=_JOINT_COLORS[j % len(_JOINT_COLORS)],
                linewidth=1.35,
                alpha=0.92,
                label=f"j{j + 1}",
            )
        ax.set_ylabel(ylabel)
        _style_ax(ax)

    axes[-1].set_xlabel("time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=dof,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.2,
        handlelength=1.8,
    )
    fig.suptitle(title or "Trajectory states", fontsize=13, fontweight="semibold", y=1.04)

    if out_path:
        fig.savefig(out_path)
    if show:
        plt.show(block=False)
        plt.pause(0.001)
    else:
        plt.close(fig)

    if verbose:
        print(f"  轨迹状态图: {out_path}" + ("（已显示）" if show else ""))


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
    title: str | None = None,
) -> dict:
    """
    对比测量力矩与辨识预测力矩 τ̂ = H·π̂。
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_plot_style(plt)

    tau_meas = np.asarray(data["tau"])
    stats = torque_prediction_rmse(tau_meas, H_stack, pi_identified, dof=dof)
    tau_pred = stats["tau_predicted"]
    time = _get_time(data, len(tau_meas))

    fig, axes = plt.subplots(
        dof,
        1,
        sharex=True,
        figsize=(11.5, 1.55 * dof + 1.0),
        constrained_layout=True,
    )
    if dof == 1:
        axes = [axes]

    for j in range(dof):
        ax = axes[j]
        ax.plot(
            time,
            tau_meas[:, j],
            color=_MEAS_COLOR,
            linewidth=1.45,
            alpha=0.88,
            label=r"measured $\tau$",
            zorder=2,
        )
        ax.plot(
            time,
            tau_pred[:, j],
            color=_PRED_COLOR,
            linewidth=1.55,
            alpha=0.95,
            label=r"predicted $H\hat{\pi}$",
            zorder=3,
        )
        ax.set_ylabel(f"j{j + 1}\n(Nm)", rotation=0, ha="right", va="center", labelpad=18)
        rmse_j = stats["rmse_per_joint"][j]
        ax.text(
            0.99,
            0.90,
            f"RMSE {rmse_j:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#5a5a5a",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "#ffffffcc",
                "edgecolor": "#ddd5cb",
                "linewidth": 0.6,
            },
        )
        _style_ax(ax)

    axes[-1].set_xlabel("time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=1.6,
        handlelength=2.2,
    )
    if title is None:
        title = "Torque: measured vs identified"
    fig.suptitle(
        f"{title}\noverall RMSE = {stats['rmse_all']:.4f} Nm",
        fontsize=13,
        fontweight="semibold",
        y=1.06,
    )

    if out_path:
        fig.savefig(out_path)
    if show:
        plt.show(block=False)
        plt.pause(0.001)
    else:
        plt.close(fig)

    if verbose:
        print(f"  力矩对比图: {out_path}" + ("（已显示）" if show else ""))
        print(f"  力矩 RMSE (全关节堆叠): {stats['rmse_all']:.6e} Nm")
        for j in range(dof):
            print(f"    关节 {j + 1} RMSE: {stats['rmse_per_joint'][j]:.6e} Nm")

    return stats
