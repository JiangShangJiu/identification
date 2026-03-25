"""动力学回归矩阵 H(q,qd,qdd)，满足 τ = H·π"""

import numpy as np

_regressor_func = None
_base_regressor_func = None
_base_params_info = None
_n_params = None


def _load_regressor(sympybotics_path=None):
    global _regressor_func, _base_regressor_func, _base_params_info, _n_params
    if _regressor_func is not None:
        return
    from .model import build_dynamics_model, get_dynparms_symbols
    rbt, rbtdef, reg, base_reg, base_info = build_dynamics_model(verbose=False, sympybotics_path=sympybotics_path)
    _regressor_func = reg
    _base_regressor_func = base_reg
    _base_params_info = base_info
    _n_params = len(get_dynparms_symbols(rbtdef)) if base_info is None else base_info["n_base"]


def get_regressor(use_base=True, sympybotics_path=None):
    """返回 (H_func, n_params)"""
    _load_regressor(sympybotics_path)
    if use_base and _base_regressor_func is not None:
        return _base_regressor_func, _base_params_info["n_base"]
    return _regressor_func, _n_params


def get_base_params_info():
    _load_regressor()
    return _base_params_info


def build_H_stack(regressor_func, data: dict, n_params: int, dof: int = 7) -> tuple:
    """构建 H_stack·π = tau_stack"""
    q, qd, qdd, tau = data["q"], data["qd"], data["qdd"], data["tau"]
    n_samples = len(q)
    H_stack = np.zeros((n_samples * dof, n_params))
    tau_stack = np.zeros(n_samples * dof)
    for i in range(n_samples):
        H_i = np.asarray(regressor_func(q[i], qd[i], qdd[i])).reshape(dof, -1)
        if H_i.shape[1] != n_params:
            H_i = H_i[:, :n_params]
        H_stack[i * dof:(i + 1) * dof, :] = H_i
        tau_stack[i * dof:(i + 1) * dof] = tau[i][:dof]
    return H_stack, tau_stack
