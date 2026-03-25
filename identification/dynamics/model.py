"""SymPyBotics 动力学模型与回归矩阵生成"""

import sys
from pathlib import Path

import numpy as np

SYMPYBOTICS_AVAILABLE = False
_sympybotics = None


def _add_sympybotics_path(path=None):
    if path is not None:
        p = Path(path)
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
        return
    candidates = [
        Path(__file__).resolve().parents[2] / "model" / "SymPyBotics",
        Path("/home/xiaomeng/code/learn_robot/model/SymPyBotics"),
    ]
    for p in candidates:
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            break


def create_panda_robotdef():
    """创建 Franka Panda 的 SymPyBotics RobotDef"""
    if _sympybotics is None:
        raise ImportError("请先调用 build_dynamics_model，并设置 sympybotics_path")
    import sympy
    dh_params = [
        ("0", "0", "0.333", "q"),
        ("-pi/2", "0", "0", "q"),
        ("pi/2", "0", "0.316", "q"),
        ("pi/2", "0.0825", "0", "q"),
        ("-pi/2", "-0.0825", "0.384", "q"),
        ("pi/2", "0", "0", "q"),
        ("pi/2", "0.088", "0.107", "q"),
    ]
    rbtdef = _sympybotics.RobotDef("Franka Emika Panda", dh_params, dh_convention="modified")
    rbtdef.frictionmodel = {"Coulomb", "viscous"}
    rbtdef.gravityacc = sympy.Matrix([0.0, 0.0, -9.81])
    return rbtdef


def build_dynamics_model(verbose=False, sympybotics_path=None):
    """建立 Panda 动力学模型，返回 (rbt, rbtdef, regressor_func, base_regressor_func, base_params_info)"""
    global _sympybotics
    _add_sympybotics_path(sympybotics_path)
    if _sympybotics is None:
        try:
            import sympybotics as sb
            _sympybotics = sb
        except ImportError:
            raise ImportError("SymPyBotics 未安装。请设置 sympybotics_path 指向 model/SymPyBotics 目录")
    sb = _sympybotics
    rbtdef = create_panda_robotdef()
    rbt = sb.RobotDynCode(rbtdef, verbose=verbose)
    exec_globals = {
        "numpy": np, "math": __import__("math"),
        "sin": np.sin, "cos": np.cos, "sign": np.sign,
        "__builtins__": __builtins__,
    }
    exec_locals = {}
    exec(
        sb.robotcodegen.robot_code_to_func("python", rbt.H_code, "regressor", "regressor_func", rbtdef),
        exec_globals, exec_locals
    )
    regressor_func = exec_locals["regressor_func"]
    base_regressor_func = None
    base_params_info = None
    try:
        rbt.calc_base_parms(verbose=verbose)
        import sympy
        base_params_info = {
            "n_base": rbt.dyn.n_base,
            "n_full": len(rbtdef.dynparms()),
            "Pb": np.array(rbt.dyn.Pb).astype(float),
            "Pd": np.array(rbt.dyn.Pd).astype(float),
            "Kd": np.array(rbt.dyn.Kd).astype(float),
            "baseparms": rbt.dyn.baseparms,
        }
        if hasattr(rbt, "Hb_code") and rbt.Hb_code:
            exec(
                sb.robotcodegen.robot_code_to_func("python", rbt.Hb_code, "regressor", "base_regressor_func", rbtdef),
                exec_globals, exec_locals
            )
            base_regressor_func = exec_locals["base_regressor_func"]
    except Exception:
        pass
    n_params = len(rbtdef.dynparms()) if base_params_info is None else base_params_info["n_base"]
    return rbt, rbtdef, regressor_func, base_regressor_func, base_params_info


def get_dynparms_symbols(rbtdef):
    return rbtdef.dynparms()
