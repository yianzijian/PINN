import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import deepxde as dde
os.environ["DDE_BACKEND"] = "pytorch"
# 物理常数
mu0 = 4 * np.pi * 1e-9
L = 1
J0 = 1000.0
A0 = mu0 * J0 * L ** 2

dde.config.set_default_float("float32")

# --- 几何定义 ---
geom = dde.geometry.Rectangle([0, 0], [1, 1])


def boundary_left(x, on_boundary):   return on_boundary and dde.utils.isclose(x[0], 0.0)
def boundary_right(x, on_boundary):  return on_boundary and dde.utils.isclose(x[0], 1.0)
def boundary_top(x, on_boundary):    return on_boundary and dde.utils.isclose(x[1], 1.0)
def boundary_bottom(x, on_boundary): return on_boundary and dde.utils.isclose(x[1], 0.0)


class HardRectangleStep(nn.Module):
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
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    f_val = f_model(x) / J0
    return -(dy_xx + dy_yy) - f_val


bcs = [
    dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top),
    dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom),
    dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left),
    dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right),
]

data = dde.data.PDE(geom, pde, bcs, num_domain=10000, num_boundary=5000, num_test=5000)

# 构建网络
net = dde.nn.FNN([2] + [128] * 6 + [1], "swish", "Glorot normal")
model = dde.Model(data, net)

# 迁移到 GPU
if torch.cuda.is_available():
    model.net.to(torch.device("cuda"))
    torch.set_float32_matmul_precision('high')
    print(f"✓ 使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("警告：GPU 不可用，使用 CPU")

# 编译
model.compile("adam", lr=0.001)

# --- Adam 阶段 ---
print("开始 Adam 20000 次迭代...")
losshistory, train_state = model.train(iterations=20000)
torch.save(model.net.state_dict(), "eshape_adam.pt")

# --- L-BFGS 阶段 ---
print("切回 CPU 准备 L-BFGS...")
model.net.to(torch.device("cpu"))

dde.optimizers.config.set_LBFGS_options(
    maxcor=100, ftol=0, gtol=1e-13, maxiter=30000
)
model.compile("L-BFGS")
losshistory, train_state = model.train()
torch.save(model.net.state_dict(), "eshape_final.pt")
print("✓ 所有训练任务已完成！")


# --- 可视化 ---
def predict_and_plot():
    print("生成磁场分布图...")
    model.net.to(torch.device("cpu"))

    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T.astype(np.float32)

    y_pred = model.predict(points)

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