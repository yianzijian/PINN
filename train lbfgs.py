#读取参数加载网络，使用 lbfgs训练的第二个程序，这里的网络形状和 adam 训练的网络形状是一样的
#前面的注释见另外一个train文件
import deepxde as dde
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



# 物理参数
mu0 = 4 * np.pi * 1e-9  # 真空磁导率 (H/cm)
L = 1                # 特征长度 (m)
J0 = 1000.0              # 特征电流密度 (A/cm^2)
A0 = mu0 * J0 * L**2    # 特征磁矢势 (Wb/m)
dde.config.set_default_float("float64")
# 归一化几何定义
geom = dde.geometry.Rectangle([0, 0], [1, 1])

# 边界函数（省略，已定义）
def boundary_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.0)
def boundary_right(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1.0)
def boundary_top(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 1.0)
def boundary_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0.0)


class HardRectangleStep(nn.Module):
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
# PDE 定义（省略，已定义）
def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    f_val = f_model(x) / J0
    return -(dy_xx + dy_yy) - f_val
def h_x(x, y, X):
    da_y = dde.grad.jacobian(y, x, i=0, j=1)
    return da_y
def h_y(x, y, X):
    da_x = dde.grad.jacobian(y, x, i=0, j=0)
    return da_x
def h_yx(x, y, X):
    da_xx = dde.grad.hessian(y, x, i=0, j=0)
    return da_xx

# 边界条件（省略，已定义）dirichlet
a_top_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top)
a_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left)
a_right_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right)
a_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom)
hx_bc_r = dde.OperatorBC(geom, h_x, boundary_right)
hx_bc_l = dde.OperatorBC(geom, h_x, boundary_left)
hy_bc_t = dde.OperatorBC(geom, h_y, boundary_top)
hy_bc_b = dde.OperatorBC(geom, h_y, boundary_bottom)


bcs = [a_top_bc, a_left_bc, a_right_bc, a_bottom_bc, hx_bc_r, hx_bc_l,hy_bc_t,hy_bc_b]

# 数据和模型（省略，已定义）
data = dde.data.PDE(geom, pde, bcs, num_domain=3000, num_boundary=100, num_test=1500)
net = dde.nn.FNN([2] + [80] * 4 + [1], "Swish", "Glorot normal")
model = dde.Model(data, net)


print("正在加载模型...")
model.net.load_state_dict(torch.load("eshapeadam5.pt"))
print("✓ 模型加载成功！")



dde.optimizers.config.set_LBFGS_options(
    maxcor=100,
    ftol=0,
    gtol=1e-13,
    maxiter=40000,
    maxfun=100000,
    maxls=50
)
#这里使用 L-BFGS 优化器
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state)


print("正在保存模型...")
torch.save(model.net.state_dict(), "eshapeadamLBFGS50005.pt")
print("✓ 模型保存成功！")

# 现在可以使用加载的模型进行预测
print("模型已准备就绪，可以进行预测...")

# 示例：进行预测
def predict_and_plot():
    # 计算 A 和 H
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    y_pred = model.predict(points)
    A_physical = y_pred.reshape(X.shape) * 10000*A0
    X_physical = X * L
    Y_physical = Y * L

    # 计算磁场
    dA_dy, dA_dx = np.gradient(A_physical, y * L, x * L)
    H_x = dA_dy / mu0
    H_y = -dA_dx / mu0
    H_magnitude = np.sqrt(H_x**2 + H_y**2)

    # 绘图
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

# 运行预测和可视化
predict_and_plot()

# 测试单点预测
test_point = np.array([[0.5, 0.5]])  # 中心点
prediction = model.predict(test_point)
print(f"在点(0.5, 0.5)处的预测值: {prediction[0][0]:.6f}")