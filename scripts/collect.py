#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""采集辨识数据"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from identification.simulation import collect_data, save_data
from identification.simulation.config import SimulationConfig


def main():
    parser = argparse.ArgumentParser(description="采集 MuJoCo 辨识数据")
    parser.add_argument("--harmonic", action="store_true", help="多谐波激励轨迹")
    parser.add_argument("--noise", action="store_true", help="力矩加高斯噪声")
    parser.add_argument("--model-root", type=str, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--n-periods", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--save", "-o", type=str, required=True, help="输出 .npz 路径")
    args = parser.parse_args()

    cfg = SimulationConfig(model_root=args.model_root)
    data = collect_data(
        config=cfg,
        duration=args.duration,
        dt=args.dt,
        use_harmonic=args.harmonic,
        n_periods=args.n_periods,
        add_noise=args.noise,
        verbose=True,
    )
    save_data(data, args.save)
    print(f"已保存: {args.save}")


if __name__ == "__main__":
    main()
