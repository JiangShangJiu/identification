"""从 MuJoCo XML 提取真实动力学参数"""

from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def extract_from_xml(xml_path: str | Path) -> list[dict]:
    """从 panda.xml 提取惯性参数"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    inertials = root.findall(".//inertial")[:7]
    params = []
    for i, inertial in enumerate(inertials):
        mass = float(inertial.get("mass"))
        pos = [float(x) for x in inertial.get("pos").split()]
        fullinertia = [float(x) for x in inertial.get("fullinertia").split()]
        ixx, iyy, izz, ixy, ixz, iyz = fullinertia
        params.append({"link": i, "mass": mass, "com": pos, "inertia_com": [ixx, iyy, izz, ixy, ixz, iyz]})
    return params


def to_sympybotics_format(params: list, fv=None, fc=None) -> np.ndarray:
    """转换为 SymPyBotics 参数格式"""
    dynparms = []
    for i, p in enumerate(params):
        m = p["mass"]
        r = np.array(p["com"])
        I_com = np.array([
            [p["inertia_com"][0], p["inertia_com"][3], p["inertia_com"][4]],
            [p["inertia_com"][3], p["inertia_com"][1], p["inertia_com"][5]],
            [p["inertia_com"][4], p["inertia_com"][5], p["inertia_com"][2]],
        ])
        r_skew = _skew(r)
        L = I_com + m * (r_skew.T @ r_skew)
        L_list = [L[0, 0], L[0, 1], L[0, 2], L[1, 1], L[1, 2], L[2, 2]]
        l_list = (m * r).tolist()
        dynparms.extend(L_list)
        dynparms.extend(l_list)
        dynparms.append(m)
        dynparms.append(fv[i] if fv and i < len(fv) else 0.0)
        dynparms.append(fc[i] if fc and i < len(fc) else 0.0)
    return np.array(dynparms)


def extract_ground_truth(xml_path: str | Path | None = None, verbose: bool = True) -> dict:
    """从 panda.xml 提取真实参数"""
    if xml_path is None:
        candidates = [
            Path(__file__).resolve().parents[1] / "franka_emika_panda" / "panda.xml",
            Path(__file__).resolve().parents[2] / "franka_emika_panda" / "panda.xml",
            Path("/home/xiaomeng/code/learn_robot/mujoco/franka_emika_panda/panda.xml"),
        ]
        for p in candidates:
            if Path(p).exists():
                xml_path = p
                break
        else:
            raise FileNotFoundError("未找到 panda.xml，请指定 xml_path")
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"panda.xml not found: {xml_path}")
    raw = extract_from_xml(xml_path)
    dynparms = to_sympybotics_format(raw)
    if verbose:
        print(f"Ground truth: {xml_path}, {len(dynparms)} params")
    return {"dynparms_array": dynparms, "raw_params": raw}
