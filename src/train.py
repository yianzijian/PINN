"""
端到端 PINN 训练：二维磁矢势 A 的泊松方程 ∇²A = -μ₀J，归一化域 [0,1]²。
流程：Adam 大规模迭代 → L-BFGS 精修 → 保存权重与 JSON 清单 → 绘制 A 与 |H|。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import deepxde as dde

from training_metadata import (
    collect_environment,
    save_manifest,
    summarize_losshistory,
    summarize_train_state,
)

# 项目根目录下的 models/ 存放 .pt 与训练清单，路径与当前工作目录无关
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = os.path.join(_ROOT, "models")
os.makedirs(_MODEL_DIR, exist_ok=True)

os.environ["DDE_BACKEND"] = "pytorch"

# --- 物理常数（与无量纲化 PDE 中的 J0 等配合使用）---
mu0 = 4 * np.pi * 1e-9  # 真空磁导率，单位与文献中 J、L 的取法一致
L = 1  # 特征长度，用于后处理把无量纲坐标换回物理尺度
J0 = 1000.0  # 参考电流密度，源项 f_model 的输出会除以 J0 进入方程
A0 = mu0 * J0 * L ** 2  # 特征磁矢势，用于把网络输出还原到物理量

# float32：Adam + GPU 上通常更快；若损失震荡可改为 float64（会更慢）
dde.config.set_default_float("float32")

# --- 计算域：单位正方形，坐标为无量纲 x,y ∈ [0,1] ---
geom = dde.geometry.Rectangle([0, 0], [1, 1])


def boundary_left(x, on_boundary):   return on_boundary and dde.utils.isclose(x[0], 0.0)
def boundary_right(x, on_boundary):  return on_boundary and dde.utils.isclose(x[0], 1.0)
def boundary_top(x, on_boundary):    return on_boundary and dde.utils.isclose(x[1], 1.0)
def boundary_bottom(x, on_boundary): return on_boundary and dde.utils.isclose(x[1], 0.0)


class HardRectangleStep(nn.Module):
    """矩形区域内为均匀电流密度（阶跃近似），区域外为 0；模拟绕组/导体窗口一类源项。"""

    def __init__(self, x_min=0.1, x_max=0.2, y_min=0.4, y_max=0.6, scale=1000.0):
        super().__init__()
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.scale = scale

    def forward(self, x):
        x_c, y_c = x[:, 0], x[:, 1]
        mask = (x_c >= self.x_min) & (x_c <= self.x_max) & \
               (y_c >= self.y_min) & (y_c <= self.y_max)
        return (mask.float() * self.scale).unsqueeze(1).to(x.device)


f_model = HardRectangleStep()


def pde(x, y):
    # y 为网络输出的标量场（磁矢势的无量纲形式）；残差对应 ∇²y + f/J0 = 0
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    f_val = f_model(x) / J0
    return -(dy_xx + dy_yy) - f_val


# 四边 Dirichlet：边界上 A=0（理想磁导体近似或远场归零类边界）
bcs = [
    dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top),
    dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom),
    dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left),
    dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right),
]

# 域内配点数 / 边界配点数 / 测试点：越大通常越稳但更慢
data = dde.data.PDE(geom, pde, bcs, num_domain=10000, num_boundary=5000, num_test=5000)

# 全连接：输入 (x,y)，输出 1 维；swish 在 PINN 中常用
net = dde.nn.FNN([2] + [128] * 6 + [1], "swish", "Glorot normal")
model = dde.Model(data, net)

# DeepXDE 在 GPU 上采点与反传；L-BFGS 阶段会切回 CPU（二阶优化器显存与实现限制）
if torch.cuda.is_available():
    model.net.to(torch.device("cuda"))
    torch.set_float32_matmul_precision('high')
    print(f"✓ 使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("警告：GPU 不可用，使用 CPU")

# 超参集中写出，便于改迭代次数或 L-BFGS 容忍度时只动这一处
ADAM_LR = 0.001
ADAM_ITERATIONS = 20000
# maxiter：L-BFGS 内层迭代上限；gtol/ftol：梯度与函数值停止准则（0 表示不按该项停）
LBFGS_OPTIONS = {"maxcor": 100, "ftol": 0, "gtol": 1e-13, "maxiter": 30000}
train_device = "cuda" if torch.cuda.is_available() else "cpu"

model.compile("adam", lr=ADAM_LR)

# --- 阶段 1：Adam 全局搜索，步数多、对学习率不极端敏感 ---
print(f"开始 Adam {ADAM_ITERATIONS} 次迭代...")
losshistory_adam, train_state_adam = model.train(iterations=ADAM_ITERATIONS)
torch.save(model.net.state_dict(), os.path.join(_MODEL_DIR, "eshape_adam.pt"))

# --- 阶段 2：L-BFGS 在解附近快速下降残差，常能压到更小 PDE 损失 ---
print("切回 CPU 准备 L-BFGS...")
model.net.to(torch.device("cpu"))

dde.optimizers.config.set_LBFGS_options(**LBFGS_OPTIONS)
model.compile("L-BFGS")
losshistory_lbfgs, train_state_lbfgs = model.train()
torch.save(model.net.state_dict(), os.path.join(_MODEL_DIR, "eshape_final.pt"))
print("✓ 所有训练任务已完成！")

# 将优化器、网络结构、采样数、最终损失等写入 models/eshape_train_manifest.json，便于复现与对比实验
_manifest_path = save_manifest(
    _MODEL_DIR,
    "eshape_train_manifest.json",
    {
        "run": {
            "script": "train.py",
            "description": "E-core Poisson / magnetic vector potential PINN (Adam + L-BFGS)",
        },
        "environment": collect_environment(torch, dde),
        "deepxde": {"default_float": "float32", "backend": os.environ.get("DDE_BACKEND", "")},
        "physics": {"mu0": mu0, "L": L, "J0": J0, "A0": A0},
        "geometry": {"type": "Rectangle", "xmin": [0, 0], "xmax": [1, 1]},
        "source_term": {
            "type": "HardRectangleStep",
            "x_min": f_model.x_min,
            "x_max": f_model.x_max,
            "y_min": f_model.y_min,
            "y_max": f_model.y_max,
            "scale": f_model.scale,
        },
        "network": {
            "class": "FNN",
            "layer_sizes": [2] + [128] * 6 + [1],
            "activation": "swish",
            "kernel_initializer": "Glorot normal",
        },
        "training_data": {
            "num_domain": 10000,
            "num_boundary": 5000,
            "num_test": 5000,
        },
        "boundary_conditions": "Dirichlet A=0 on all four sides",
        "optimization": {
            "stages": [
                {
                    "optimizer": "Adam",
                    "learning_rate": ADAM_LR,
                    "iterations": ADAM_ITERATIONS,
                    "device": train_device,
                    "losshistory": summarize_losshistory(losshistory_adam),
                    "train_state": summarize_train_state(train_state_adam),
                },
                {
                    "optimizer": "L-BFGS",
                    "lbfgs_options": LBFGS_OPTIONS,
                    "device": "cpu",
                    "losshistory": summarize_losshistory(losshistory_lbfgs),
                    "train_state": summarize_train_state(train_state_lbfgs),
                },
            ],
        },
        "artifacts": {
            "checkpoints": {
                "after_adam": "eshape_adam.pt",
                "final": "eshape_final.pt",
            },
            "manifest": "eshape_train_manifest.json",
        },
    },
)
print(f"✓ 训练参数已保存: {_manifest_path}")


def predict_and_plot():
    """由 A 求 H：二维下 H_x = (1/μ₀)∂A/∂y，H_y = -(1/μ₀)∂A/∂x；再画 |H|。"""
    print("生成磁场分布图...")
    model.net.to(torch.device("cpu"))

    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T.astype(np.float32)

    y_pred = model.predict(points)

    # *10000*A0：与当前无量纲标定一致，把网络输出换到 Wb/m 量级便于和 COMSOL 对比
    A_phys = y_pred.reshape(X.shape) * 10000 * A0
    dA_dy, dA_dx = np.gradient(A_phys, y * L, x * L)
    H_mag = np.sqrt((dA_dy / mu0) ** 2 + (-dA_dx / mu0) ** 2)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.contourf(X * L, Y * L, A_phys, levels=300, cmap="jet")
    plt.colorbar(label="|A| (Wb/m)")
    plt.title("Magnetic Vector Potential A")

    plt.subplot(1, 2, 2)
    plt.contourf(X * L, Y * L, H_mag, levels=500, cmap="jet")
    plt.colorbar(label="|H| (A/m)")
    plt.title("Magnetic Field |H|")
    plt.tight_layout()
    plt.show()


predict_and_plot()