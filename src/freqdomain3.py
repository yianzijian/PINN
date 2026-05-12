"""
端到端 PINN 训练：二维磁矢势 A 的频域亥姆霍兹方程
物理模型：多材料分布（硅钢磁芯 + 铜绕组 + 空气）
训练域：厘米单位制 (cm)
输出域：国际标准公制 (m, Wb/m)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import deepxde as dde
from datetime import datetime


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = os.path.join(_ROOT, "models")
_RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(_MODEL_DIR, exist_ok=True)
os.makedirs(_RESULTS_DIR, exist_ok=True)


# =============================================
# 物理常数与材料属性 (单位: cm)
# =============================================
mu0_cm = 4 * np.pi * 1e-9          # H/cm
L_cm = 1.0                         # 特征长度 1 cm
J0_cm = 1000.0                     # 源电流密度 1000 A/cm²
A0_cm = mu0_cm * J0_cm * (L_cm**2) # 特征磁矢势 T·cm (Wb/cm)

freq = 100
omega = 2 * np.pi * freq

# 材料电导率 (S/m 转换为 S/cm: 除以 100)
sigma_core_cm = 2e4   # 硅钢: 2e6 S/m -> 2e4 S/cm
sigma_coil_cm = 5.8e5 # 铜线: 5.8e7 S/m -> 5.8e5 S/cm

dde.config.set_default_float("float32")

# =============================================
# 几何域与材料分布掩码
# =============================================
geom = dde.geometry.Rectangle([0, 0], [1, 1])

def get_sigma_distribution(x):
    """
    根据坐标返回空间的电导率分布 (单位: S/cm)
    """
    x_c, y_c = x[:, 0:1], x[:, 1:2]

    # 1. 铁芯 (Core): 外框 [0, 0.4]x[0.3, 0.7] 扣除 内孔 (0.1, 0.3)x(0.4, 0.6)
    in_outer = (x_c >= 0.0) & (x_c <= 0.4) & (y_c >= 0.3) & (y_c <= 0.7)
    in_inner = (x_c > 0.1) & (x_c < 0.3) & (y_c > 0.4) & (y_c < 0.6)
    is_core = in_outer & (~in_inner)

    # 2. 铜绕组 (Coil): 位于 [0.1, 0.2]x[0.4, 0.6]
    is_coil = (x_c >= 0.1) & (x_c <= 0.2) & (y_c >= 0.4) & (y_c <= 0.6)

    # 3. 计算空间各点的电导率
    sigma_val = is_core.float() * sigma_core_cm + is_coil.float() * sigma_coil_cm
    return sigma_val

class HardRectangleStep(nn.Module):
    def __init__(self, x_min=0.1, x_max=0.2, y_min=0.4, y_max=0.6, scale=1000.0):
        super().__init__()
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.scale = scale # 铜线区域内施加源电流

    def forward(self, x):
        x_c, y_c = x[:, 0], x[:, 1]
        mask = (x_c >= self.x_min) & (x_c <= self.x_max) & \
               (y_c >= self.y_min) & (y_c <= self.y_max)
        return (mask.float() * self.scale).unsqueeze(1).to(x.device)

f_model = HardRectangleStep(scale=J0_cm)

# =============================================
# 频域 PDE 定义
# =============================================
def pde(x, y):
    Ar, Ai = y[:, 0:1], y[:, 1:2]

    dAr_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dAr_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)

    dAi_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dAi_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

    # 归一化源电流
    f_val = f_model(x) / J0_cm

    # 核心修改：获取随空间变化的局域耦合系数 C(x, y)
    sigma_x = get_sigma_distribution(x)
    C_local = omega * mu0_cm * sigma_x * (L_cm ** 2)

    # 耦合方程组 (不同材质内涡流效应强度不同)
    res_real = (dAr_xx + dAr_yy) + C_local * Ai + f_val
    res_imag = (dAi_xx + dAi_yy) - C_local * Ar

    return [res_real, res_imag]

# =============================================
# 边界条件 (四周磁绝缘 A=0)
# =============================================
def boundary_all(x, on_boundary): return on_boundary
# ==========================================
# 1. 边界位置判断函数 (对应图片中的 O, A, B, C 四条边)
# =============================================
# 底部 O-A (y=0)
def boundary_bottom(x, on_boundary): return on_boundary and dde.utils.isclose(x[1], 0.0)
# 右侧 A-B (x=1)
def boundary_right(x, on_boundary):  return on_boundary and dde.utils.isclose(x[0], 1.0)
# 顶部 B-C (y=1
def boundary_top(x, on_boundary):    return on_boundary and dde.utils.isclose(x[1], 1.0)
# 左侧 O-C (x=0)
def boundary_left(x, on_boundary):   return on_boundary and dde.utils.isclose(x[0], 0.0)


# =============================================
# 2. Operator 算子定义：将所有物理条件转换为残差
# y维度: [N, 2] -> i=0(实部), i=1(虚部)
# x维度: [N, 2] -> j=0(x坐标), j=1(y坐标)
# =============================================
# 2.1 Dirichlet 算子：A = 0
def dirichlet_real(x, y, _): return y[:, 0:1]
def dirichlet_imag(x, y, _): return y[:, 1:2]
# 2.2 一阶导数：dA/dx = 0 (对应 Hy = 0)
def dAdx_real(x, y, _): return dde.grad.jacobian(y, x, i=0, j=0)
def dAdx_imag(x, y, _): return dde.grad.jacobian(y, x, i=1, j=0)
# 2.3 一阶导数：dA/dy = 0 (对应 Hx = 0)
def dAdy_real(x, y, _): return dde.grad.jacobian(y, x, i=0, j=1)
def dAdy_imag(x, y, _): return dde.grad.jacobian(y, x, i=1, j=1)
# 2.4 二阶导数：d²A/dx² = 0 (对应 dHy/dx = 0)
def d2Adx2_real(x, y, _): return dde.grad.hessian(y, x, component=0, i=0, j=0)
def d2Adx2_imag(x, y, _): return dde.grad.hessian(y, x, component=1, i=0, j=0)


# =============================================
# 3. 组装所有边界条件到 PINN 训练集中
# =============================================
bcs = [
    # ---------------------------------------------------------
    # 第一行公式：四周 A = 0
    # A|_{O-A} = A|_{A-B} = A|_{B-C} = A|_{O-C} = 0
    # ---------------------------------------------------------
    dde.icbc.OperatorBC(geom, dirichlet_real, boundary_bottom),  # O-A
    dde.icbc.OperatorBC(geom, dirichlet_imag, boundary_bottom),

    dde.icbc.OperatorBC(geom, dirichlet_real, boundary_right),  # A-B
    dde.icbc.OperatorBC(geom, dirichlet_imag, boundary_right),

    dde.icbc.OperatorBC(geom, dirichlet_real, boundary_top),  # B-C
    dde.icbc.OperatorBC(geom, dirichlet_imag, boundary_top),

    dde.icbc.OperatorBC(geom, dirichlet_real, boundary_left),  # O-C
    dde.icbc.OperatorBC(geom, dirichlet_imag, boundary_left),

    # ---------------------------------------------------------
    # 第二行公式：导数约束 Hx = 0 和 Hy = 0
    # ---------------------------------------------------------
    # Hx|_{O-C} = Hx|_{A-B} = 0  => 左右边界 dA/dy = 0
    dde.icbc.OperatorBC(geom, dAdy_real, boundary_left),  # O-C
    dde.icbc.OperatorBC(geom, dAdy_imag, boundary_left),
    dde.icbc.OperatorBC(geom, dAdy_real, boundary_right),  # A-B
    dde.icbc.OperatorBC(geom, dAdy_imag, boundary_right),

    # Hy|_{O-A} = Hy|_{B-C} = 0  => 上下边界 dA/dx = 0
    dde.icbc.OperatorBC(geom, dAdx_real, boundary_bottom),  # O-A
    dde.icbc.OperatorBC(geom, dAdx_imag, boundary_bottom),
    dde.icbc.OperatorBC(geom, dAdx_real, boundary_top),  # B-C
    dde.icbc.OperatorBC(geom, dAdx_imag, boundary_top),

]



# =============================================
# 模型构建
# =============================================
# 多材料边界会产生强烈的物理场跳变，需要足够的采样点来捕获

data = dde.data.PDE(geom, pde, bcs, num_domain=1000, num_boundary=300, num_test=300)
net = dde.nn.FNN([2] + [64] * 6 + [2], "tanh", "Glorot normal")
model = dde.Model(data, net)

# =============================================
# 训练阶段
# =============================================
print("✓ 多材料电导率分布初始化完成 (空气/硅钢/铜)")
model.compile("adam", lr=0.002)
model.train(iterations=1000)
# 生成时间戳（格式：YYYYMMDD_HHMMSS）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
adam_model_name = f"eshape_adam_{timestamp}.pt"
final_model_name = f"eshape_final_{timestamp}.pt"

torch.save(model.net.state_dict(), os.path.join(_MODEL_DIR, adam_model_name))  # 保存 Adam 训练结果
print(f"✓ Adam 阶段模型已保存: {adam_model_name}")


# =============================================
# Residual-based Adaptive Refinement (RAR)
# =============================================
# Residual-based Adaptive Refinement (RAR) 修正版
# =============================================
def residual_based_adaptive_refinement(model, geom, rounds=3, n_new_points=500, n_candidates=5000):
    print(f"\n--- 启动 Residual-based Adaptive Refinement (共 {rounds} 轮) ---")

    for i in range(rounds):
        # 1. 随机生成候选海选点
        X_candidates = geom.random_points(n_candidates)

        # 2. 预测 PDE 残差 (关键：使用 predict 并指定 operator)
        f_res = model.predict(X_candidates, operator=pde)

        # 3. 计算综合残差分值
        f_res_np = np.array(f_res)
        err_score = np.sum(np.abs(f_res_np), axis=0).flatten()

        # 4. 筛选残差最大的点
        err_indices = np.argsort(err_score)[-n_new_points:]
        new_points = X_candidates[err_indices]

        # 5. 添加新点到训练集
        model.data.add_anchors(new_points)

        print(f"RAR 轮次 {i + 1}/{rounds}: 识别并加密了 {len(new_points)} 个复杂场区域点.")

        # 6. 微调训练
        model.compile("adam", lr=0.0005)
        model.train(iterations=1000)
# 建议在基础训练完成后执行，此时模型已初步掌握场分布
residual_based_adaptive_refinement(model, geom, rounds=10, n_new_points=500)

# 3. 最终 L-BFGS 压平残差
model.compile("L-BFGS")
model.train()
dde.optimizers.config.set_LBFGS_options(maxiter=1000)
model.compile("L-BFGS")
model.train()
torch.save(model.net.state_dict(), os.path.join(_MODEL_DIR, final_model_name))  # 保存最终模型
print(f"✓ L-BFGS 阶段模型已保存: {final_model_name}")
print("✓ 所有训练任务已完成！")
# =============================================
# 预测与可视化
# =============================================
def predict_and_plot():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res = 150 # 提高作图分辨率以便观察材料边界

    x_cm = np.linspace(0, 1, res)
    y_cm = np.linspace(0, 1, res)
    X_cm, Y_cm = np.meshgrid(x_cm, y_cm)
    points = np.vstack([X_cm.ravel(), Y_cm.ravel()]).T.astype(np.float64)

    y_pred = model.predict(points)

    Ar_cm = y_pred[:, 0].reshape(X_cm.shape) * A0_cm
    Ai_cm = y_pred[:, 1].reshape(X_cm.shape) * A0_cm

    # 坐标 cm -> m, 磁矢势 Wb/cm -> Wb/m
    X_m = X_cm / 100.0
    Y_m = Y_cm / 100.0
    Ar_si = Ar_cm * 100.0
    Ai_si = Ai_cm * 100
    A_abs_si = np.sqrt(Ar_si ** 2 + Ai_si ** 2)

    # 绘制
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Real Part $A_r$ (Wb/m)", "Imaginary Part $A_i$ (Wb/m)", "Magnitude $|A|$ (Wb/m)"]
    datasets = [Ar_si, Ai_si, A_abs_si]

    for ax, data, title in zip(axes, datasets, titles):
        c = ax.contourf(X_m, Y_m, data, levels=100, cmap="jet")
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(c, ax=ax, format='%.2e')

        # 用虚线在图上勾勒出铁芯和铜线的轮廓线，方便观察
        # 画铁芯外框
        ax.plot([0, 0.004, 0.004, 0, 0], [0.003, 0.003, 0.007, 0.007, 0.003], 'w--', alpha=0.5)
        # 画铜线框
        ax.plot([0.001, 0.002, 0.002, 0.001, 0.001], [0.004, 0.004, 0.006, 0.006, 0.004], 'k--', alpha=0.8)

    plt.tight_layout()
    img_save_path = os.path.join(_RESULTS_DIR, f"multi_material_SI_{timestamp}.png")
    plt.savefig(img_save_path, dpi=300)
    print(f"\n--- 公制单位输出结果 (SI) ---")
    print(f"✓ 最大磁矢势幅值: {np.max(A_abs_si):.4e} Wb/m")
    print(f"✓ 图像已保存至: {img_save_path}")
    plt.show()

predict_and_plot()