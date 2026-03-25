# 机器人动力学参数辨识

从 `learn_robot/mujoco/identification` 抽象出的 Franka Panda 动力学参数辨识模块，包含 MuJoCo 仿真数据采集与辨识流程。

## 目录结构

```
identification/
├── identification/           # 主包
│   ├── simulation/          # MuJoCo 仿真数据采集
│   │   ├── config.py         # 仿真配置
│   │   ├── loader.py         # 模型加载（可配置路径）
│   │   └── collector.py      # 轨迹仿真 + 数据采集 (q, qd, qdd, tau)
│   ├── trajectory/           # 激励轨迹
│   │   ├── generators.py     # 正弦/多项式/随机/多谐波
│   │   └── collect.py       # 采集用轨迹接口
│   ├── dynamics/             # 动力学
│   │   ├── model.py          # SymPyBotics 动力学模型
│   │   ├── regressor.py     # 回归矩阵 H
│   │   └── base_params.py   # 基参数转换
│   ├── solver.py            # 最小二乘辨识
│   ├── compare.py            # 辨识 vs 真实参数对比
│   └── ground_truth.py       # 从 XML 提取真实参数
├── scripts/                  # CLI 入口
│   ├── main.py               # 一键流程
│   └── collect.py            # 仅采集数据
├── requirements.txt
└── README.md
```

## 依赖

- `numpy`, `scipy`, `sympy`, `mujoco`
- **SymPyBotics**：用于回归矩阵，需将 `model/SymPyBotics` 路径加入 `sys.path` 或安装 `sympybotics`
- **Franka 模型**：需 `franka_emika_panda` 目录（含 `scene.xml`, `panda.xml`）

## 配置路径

若模型或 SymPyBotics 不在默认位置，可：

1. 复制 `franka_emika_panda` 到项目根目录
2. 复制 `model/SymPyBotics` 到项目根目录
3. 或通过 `--model-root` 指定模型根目录

默认会尝试：
- `identification/franka_emika_panda`
- `identification/../mujoco/franka_emika_panda`
- `/home/xiaomeng/code/learn_robot/mujoco/franka_emika_panda`

## 运行

```bash
# 进入项目根目录
cd /home/xiaomeng/code/identification

# 一键流程（采集 + 辨识 + 对比）
python scripts/main.py --harmonic [--noise] [--save-data data.npz] [--load-data data.npz]

# 仅采集数据
python scripts/collect.py -o data.npz --harmonic [--noise]
```

## 力矩说明

- **τ 来源**：`mj_inverse` 得到的 `qfrc_inverse`（逆动力学理想力矩）
- 适合参数辨识，与含控制器/摩擦的 `actuator_force` 不同
