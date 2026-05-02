"""
第二阶段：在已有权重上做 L-BFGS 精修（本脚本为 float64 + OperatorBC + [80]*4 网络）。
与 train.py（[128]*6、仅 Dirichlet）不是同一套结构；eshapeadam5.pt 必须与下面 net 定义一致。
"""
import os
import deepxde as dde
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from training_metadata import (
    collect_environment,
    save_manifest,
    summarize_losshistory,
    summarize_train_state,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = os.path.join(_ROOT, "models")
os.makedirs(_MODEL_DIR, exist_ok=True)


# --- 物理参数（含义同 train.py）---
mu0 = 4 * np.pi * 1e-9  # 真空磁导率 (H/cm)
L = 1                # 特征长度 (m)
J0 = 1000.0              # 特征电流密度 (A/cm^2)
A0 = mu0 * J0 * L**2    # 特征磁矢势 (Wb/m)
# 二阶优化与病态 Hessian 时常用 float64，减少数值噪声
dde.config.set_default_float("float64")
geom = dde.geometry.Rectangle([0, 0], [1, 1])

# 各边的几何判定：用于把边界条件挂到 x=0/1、y=0/1 上
def boundary_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.0)
def boundary_right(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1.0)
def boundary_top(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 1.0)
def boundary_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0.0)


class HardRectangleStep(nn.Module):
    """矩形载流区域；与 train.py 中类等价，写法略不同。"""

    def __init__(self, x_min=0.1, x_max=0.2, y_min=0.4, y_max=0.6, scale=1000.0):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.scale = scale
    
    def forward(self, x):
        x_coord = x[:, 0]
        y_coord = x[:, 1]
        fx = ((x_coord >= self.x_min) & (x_coord <= self.x_max)).float()
        fy = ((y_coord >= self.y_min) & (y_coord <= self.y_max)).float()
        fxy = self.scale * fx * fy
        return fxy.unsqueeze(1)
        
f_model = HardRectangleStep()


def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    f_val = f_model(x) / J0
    return -(dy_xx + dy_yy) - f_val


def h_x(x, y, X):
    # H 的 x 分量与 ∂A/∂y 相关（2D z 向 A）
    da_y = dde.grad.jacobian(y, x, i=0, j=1)
    return da_y


def h_y(x, y, X):
    da_x = dde.grad.jacobian(y, x, i=0, j=0)
    return da_x


def h_yx(x, y, X):
    # 预留：若需在边界约束 ∂H/∂x 等可用
    da_xx = dde.grad.hessian(y, x, i=0, j=0)
    return da_xx

# Dirichlet：四边 A=0；OperatorBC：在边上对 H 相关算子施加约束（具体物理含义依赖你的建模）
a_top_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top)
a_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left)
a_right_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right)
a_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom)
hx_bc_r = dde.OperatorBC(geom, h_x, boundary_right)
hx_bc_l = dde.OperatorBC(geom, h_x, boundary_left)
hy_bc_t = dde.OperatorBC(geom, h_y, boundary_top)
hy_bc_b = dde.OperatorBC(geom, h_y, boundary_bottom)


bcs = [a_top_bc, a_left_bc, a_right_bc, a_bottom_bc, hx_bc_r, hx_bc_l, hy_bc_t, hy_bc_b]

data = dde.data.PDE(geom, pde, bcs, num_domain=3000, num_boundary=100, num_test=1500)
net = dde.nn.FNN([2] + [80] * 4 + [1], "Swish", "Glorot normal")
model = dde.Model(data, net)

# 必须已有同名权重；若你从 train.py 只得到 eshape_adam.pt，需复制/改名为 eshapeadam5.pt 或改此处文件名
print("正在加载模型...")
model.net.load_state_dict(torch.load(os.path.join(_MODEL_DIR, "eshapeadam5.pt")))
print("✓ 模型加载成功！")

# maxcor：L-BFGS 保留的曲率对数量；maxfun/maxls：函数求值与线搜索次数上限
LBFGS_OPTIONS = {
    "maxcor": 100,
    "ftol": 0,
    "gtol": 1e-13,
    "maxiter": 40000,
    "maxfun": 100000,
    "maxls": 50,
}
dde.optimizers.config.set_LBFGS_options(**LBFGS_OPTIONS)
model.compile("L-BFGS")
losshistory_lbfgs, train_state_lbfgs = model.train()
# 在当前工作目录生成 loss 曲线等图；与 models/ 下清单无关
dde.saveplot(losshistory_lbfgs, train_state_lbfgs)

print("正在保存模型...")
OUT_CKPT = "eshapeadamLBFGS50005.pt"
torch.save(model.net.state_dict(), os.path.join(_MODEL_DIR, OUT_CKPT))
print("✓ 模型保存成功！")

_manifest_path = save_manifest(
    _MODEL_DIR,
    "eshape_lbfgs_manifest.json",
    {
        "run": {
            "script": "train lbfgs.py",
            "description": "L-BFGS fine-tune (loads Adam checkpoint; extended BCs)",
        },
        "environment": collect_environment(torch, dde),
        "deepxde": {"default_float": "float64"},
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
            "layer_sizes": [2] + [80] * 4 + [1],
            "activation": "Swish",
            "kernel_initializer": "Glorot normal",
        },
        "training_data": {
            "num_domain": 3000,
            "num_boundary": 100,
            "num_test": 1500,
        },
        "boundary_conditions": "Dirichlet A=0 + OperatorBC for H on boundaries",
        "initialization": {
            "checkpoint": "eshapeadam5.pt",
            "path": os.path.join(_MODEL_DIR, "eshapeadam5.pt"),
        },
        "optimization": {
            "stages": [
                {
                    "optimizer": "L-BFGS",
                    "lbfgs_options": LBFGS_OPTIONS,
                    "losshistory": summarize_losshistory(losshistory_lbfgs),
                    "train_state": summarize_train_state(train_state_lbfgs),
                },
            ],
        },
        "artifacts": {
            "checkpoints": {"output": OUT_CKPT},
            "manifest": "eshape_lbfgs_manifest.json",
        },
    },
)
print(f"✓ 训练参数已保存: {_manifest_path}")

print("模型已准备就绪，可以进行预测...")


def predict_and_plot():
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    y_pred = model.predict(points)
    # 与 train.py 相同的标定，使 A 处于可与 COMSOL 对比的量级
    A_physical = y_pred.reshape(X.shape) * 10000 * A0
    X_physical = X * L
    Y_physical = Y * L

    dA_dy, dA_dx = np.gradient(A_physical, y * L, x * L)
    H_x = dA_dy / mu0
    H_y = -dA_dx / mu0
    H_magnitude = np.sqrt(H_x**2 + H_y**2)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.contourf(X_physical, Y_physical, A_physical, levels=300, cmap="jet")
    plt.colorbar(label="|A| (wb/m)")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.title("Magnetic Vector Potential A")
    plt.axis("equal")
    
    plt.subplot(1, 2, 2)
    plt.contourf(X_physical, Y_physical, H_magnitude, levels=500, cmap="jet")
    plt.colorbar(label="|H| (A/m)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Magnetic Field |H|")
    plt.axis("equal")
    
    plt.tight_layout()
    plt.show()

predict_and_plot()

# 域内一点 smoke test，检查网络输出是否为有限值
test_point = np.array([[0.5, 0.5]])  # 中心点
prediction = model.predict(test_point)
print(f"在点(0.5, 0.5)处的预测值: {prediction[0][0]:.6f}")