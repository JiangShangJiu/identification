"""辨识结果 vs 真实参数对比"""

import numpy as np


def compare(pi_identified: np.ndarray, pi_true: np.ndarray, verbose: bool = True) -> dict:
    """
    Returns:
        dict: rmse, max_rel_error, mean_rel_error, abs_error, relative_error
    """
    n = min(len(pi_identified), len(pi_true))
    pi_id = np.asarray(pi_identified[:n])
    pi_tr = np.asarray(pi_true[:n])
    abs_err = np.abs(pi_id - pi_tr)
    denom = np.where(np.abs(pi_tr) < 1e-12, 1.0, np.abs(pi_tr))
    rel_err = abs_err / denom
    rmse = np.sqrt(np.mean((pi_id - pi_tr) ** 2))
    max_rel = float(np.max(rel_err)) if n > 0 else 0.0
    mean_rel = float(np.mean(rel_err)) if n > 0 else 0.0
    if verbose:
        print(f"Comparison: RMSE={rmse:.6e}, max_rel={max_rel:.2%}, mean_rel={mean_rel:.2%}")
    return {"rmse": rmse, "max_rel_error": max_rel, "mean_rel_error": mean_rel, "abs_error": abs_err, "relative_error": rel_err}
