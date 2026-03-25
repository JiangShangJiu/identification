"""最小二乘辨识 π = (H'H)^{-1} H'τ"""

import numpy as np
from scipy.linalg import lstsq


def identify(H_stack: np.ndarray, tau_stack: np.ndarray, verbose: bool = True) -> dict:
    """
    最小二乘辨识 π = pinv(H) @ tau

    Returns:
        dict: pi_identified, residuals, rmse, rank
    """
    try:
        pi, _, rank, _ = lstsq(H_stack, tau_stack, lapack_driver="gelsd")
    except Exception:
        pi = np.linalg.pinv(H_stack) @ tau_stack
        rank = np.linalg.matrix_rank(H_stack)
    residuals = tau_stack - H_stack @ pi
    rmse = np.sqrt(np.mean(residuals**2))
    if verbose:
        print(f"Identification: rank={rank}/{H_stack.shape[1]}, RMSE={rmse:.6e}")
    return {"pi_identified": pi, "residuals": residuals, "rmse": rmse, "rank": rank}
