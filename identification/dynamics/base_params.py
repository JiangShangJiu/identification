"""基参数与完整参数转换"""

import numpy as np


def convert_base_to_full_params(pi_base, base_params_info):
    """基参数 -> 完整参数"""
    pi_base = np.array(pi_base).flatten()
    Pb = np.array(base_params_info["Pb"])
    Pd = np.array(base_params_info["Pd"])
    Kd = np.array(base_params_info["Kd"])
    pi_dep = Kd.T @ pi_base
    return Pb @ pi_base + Pd @ pi_dep


def convert_full_to_base_params(pi_full, base_params_info):
    """完整参数 -> 基参数（投影）"""
    pi_full = np.array(pi_full).flatten()
    Pb = np.array(base_params_info["Pb"])
    Pd = np.array(base_params_info["Pd"])
    Kd = np.array(base_params_info["Kd"])
    P = Pb + Pd @ Kd.T
    try:
        return np.linalg.solve(P.T @ P, P.T @ pi_full)
    except Exception:
        return np.linalg.pinv(P) @ pi_full
