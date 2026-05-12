import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import deepxde as dde
from datetime import datetime

# ============================================================
# 使用 float64
# 对 Helmholtz / 高频 PDE 非常重要
# ============================================================
dde.config.set_default_float("float64")

# ============================================================
# 保存路径
# ============================================================
_ROOT = os.path.dirname(os.path.abspath(__file__))

_MODEL_DIR = os.path.join(_ROOT, "models")
_RESULTS_DIR = os.path.join(_ROOT, "results")

os.makedirs(_MODEL_DIR, exist_ok=True)
os.makedirs(_RESULTS_DIR, exist_ok=True)

# ============================================================
# 物理参数（cm 制）
# ============================================================

mu0_cm = 4 * np.pi * 1e-9

L_cm = 1.0
J0_cm = 1000.0

# 特征磁矢势
A0_cm = mu0_cm * J0_cm * L_cm**2

freq = 100
omega = 2 * np.pi * freq

# 电导率 (S/cm)
sigma_core_cm = 2e4
sigma_coil_cm = 5.8e5

# ============================================================
# 几何区域
# ============================================================

geom = dde.geometry.Rectangle([0, 0], [1, 1])

# ============================================================
# 电导率分布
# ============================================================

def get_sigma_distribution(x):

    x_c = x[:, 0:1]
    y_c = x[:, 1:2]

    # 铁芯区域
    in_outer = (
        (x_c >= 0.0) &
        (x_c <= 0.4) &
        (y_c >= 0.3) &
        (y_c <= 0.7)
    )

    in_inner = (
        (x_c > 0.1) &
        (x_c < 0.3) &
        (y_c > 0.4) &
        (y_c < 0.6)
    )

    is_core = in_outer & (~in_inner)

    # 铜线区域
    is_coil = (
        (x_c >= 0.1) &
        (x_c <= 0.2) &
        (y_c >= 0.4) &
        (y_c <= 0.6)
    )

    sigma_val = (
        is_core.float() * sigma_core_cm
        + is_coil.float() * sigma_coil_cm
    )

    return sigma_val

# ============================================================
# 电流源
# ============================================================

class HardRectangleStep(nn.Module):

    def __init__(
        self,
        x_min=0.1,
        x_max=0.2,
        y_min=0.4,
        y_max=0.6,
        scale=1000.0
    ):
        super().__init__()

        self.x_min = x_min
        self.x_max = x_max

        self.y_min = y_min
        self.y_max = y_max

        self.scale = scale

    def forward(self, x):

        x_c = x[:, 0]
        y_c = x[:, 1]

        mask = (
            (x_c >= self.x_min)
            & (x_c <= self.x_max)
            & (y_c >= self.y_min)
            & (y_c <= self.y_max)
        )

        return (mask.float() * self.scale).unsqueeze(1)

f_model = HardRectangleStep(scale=J0_cm)

# ============================================================
# PDE
# ============================================================

def pde(x, y):

    Ar = y[:, 0:1]
    Ai = y[:, 1:2]

    # --------------------------------------------------------
    # Hessian
    # --------------------------------------------------------

    dAr_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dAr_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)

    dAi_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dAi_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

    # --------------------------------------------------------
    # source term
    # --------------------------------------------------------

    f_val = f_model(x) / J0_cm

    # --------------------------------------------------------
    # 空间变化耦合系数
    # --------------------------------------------------------

    sigma_x = get_sigma_distribution(x)

    C_local = omega * mu0_cm * sigma_x * L_cm**2

    eps = 1e-8

    # ========================================================
    # PDE 重缩放
    #
    # 原方程:
    # ∇²A + C A = f
    #
    # 改写:
    # (1/C)∇²A + A = f/C
    #
    # 能明显改善 PINN 训练稳定性
    # ========================================================

    res_real = (
        (dAr_xx + dAr_yy) / (C_local + eps)
        + Ai
        + f_val / (C_local + eps)
    )

    res_imag = (
        (dAi_xx + dAi_yy) / (C_local + eps)
        - Ar
    )

    return [res_real, res_imag]

# ============================================================
# 边界条件
#
# 这里只保留 Dirichlet:
# A = 0
#
# 避免过约束
# ============================================================

def boundary_all(x, on_boundary):
    return on_boundary

def dirichlet_real(x, y, _):
    return y[:, 0:1]

def dirichlet_imag(x, y, _):
    return y[:, 1:2]

bcs = [

    dde.icbc.OperatorBC(
        geom,
        dirichlet_real,
        boundary_all
    ),

    dde.icbc.OperatorBC(
        geom,
        dirichlet_imag,
        boundary_all
    ),
]

# ============================================================
# 数据集
# ============================================================

data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=3000,
    num_boundary=500,
    num_test=1000
)

# ============================================================
# 网络
#
# tanh 对二阶 PDE 更稳定
# ============================================================

net = dde.nn.FNN(
    [2] + [64] * 6 + [2],
    "tanh",
    "Glorot normal"
)

model = dde.Model(data, net)

# ============================================================
# loss weighting
#
# 前两个:
# PDE real / imag
#
# 后两个:
# BC real / imag
# ============================================================

loss_weights = [
    1.0,
    1.0,
    10.0,
    10.0,
]

# ============================================================
# Adam
# ============================================================

model.compile(
    "adam",
    lr=1e-3,
    loss_weights=loss_weights
)

losshistory, train_state = model.train(
    iterations=20000
)

# ============================================================
# RAR
# ============================================================

def residual_based_adaptive_refinement(
    model,
    geom,
    rounds=3,
    n_new_points=500,
    n_candidates=5000
):

    print("\n========== 开始 RAR ==========\n")

    for i in range(rounds):

        X_candidates = geom.random_points(n_candidates)

        # PDE residual
        f_res = model.predict(
            X_candidates,
            operator=pde
        )

        f_res = np.array(f_res)

        # ====================================================
        # 使用真正 L2 residual
        # ====================================================

        err_score = np.sqrt(
            f_res[0, :, 0]**2
            +
            f_res[1, :, 0]**2
        )

        err_indices = np.argsort(err_score)[-n_new_points:]

        new_points = X_candidates[err_indices]

        model.data.add_anchors(new_points)

        print(
            f"RAR round {i+1}: "
            f"added {len(new_points)} points"
        )

        model.compile(
            "adam",
            lr=5e-4,
            loss_weights=loss_weights
        )

        model.train(iterations=1000)

# 执行 RAR
residual_based_adaptive_refinement(model, geom)

# ============================================================
# L-BFGS
# ============================================================

dde.optimizers.config.set_LBFGS_options(
    maxiter=1000
)

model.compile(
    "L-BFGS",
    loss_weights=loss_weights
)

model.train()

# ============================================================
# 保存模型
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

save_path = os.path.join(
    _MODEL_DIR,
    f"pinn_{timestamp}.pt"
)

torch.save(
    model.net.state_dict(),
    save_path
)

print(f"\n模型已保存:\n{save_path}")

# ============================================================
# 预测与可视化
# ============================================================

def predict_and_plot(model):

    print("\n========== 开始预测与绘图 ==========\n")

    # --------------------------------------------------------
    # 网格分辨率
    # --------------------------------------------------------

    res = 200

    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)

    X, Y = np.meshgrid(x, y)

    points = np.vstack(
        [X.ravel(), Y.ravel()]
    ).T

    # --------------------------------------------------------
    # PINN 预测
    # --------------------------------------------------------

    pred = model.predict(points)

    Ar = pred[:, 0].reshape(res, res)
    Ai = pred[:, 1].reshape(res, res)

    # --------------------------------------------------------
    # 恢复物理量纲
    #
    # 当前网络输出是无量纲:
    #
    # A_hat = A / A0
    #
    # 所以:
    #
    # A = A_hat * A0
    # --------------------------------------------------------

    Ar = Ar * A0_cm
    Ai = Ai * A0_cm

    # cm -> m
    Ar = Ar * 100.0
    Ai = Ai * 100.0

    # --------------------------------------------------------
    # 模值
    # --------------------------------------------------------

    A_abs = np.sqrt(
        Ar**2 + Ai**2
    )

    # ========================================================
    # 计算磁场
    #
    # Hx = dA/dy / mu
    # Hy = -dA/dx / mu
    # ========================================================

    dx = (x[1] - x[0]) / 100.0
    dy = (y[1] - y[0]) / 100.0

    dAr_dy, dAr_dx = np.gradient(
        Ar,
        dy,
        dx
    )

    dAi_dy, dAi_dx = np.gradient(
        Ai,
        dy,
        dx
    )

    # --------------------------------------------------------
    # 这里只使用 mu0
    #
    # 如果未来考虑磁芯高磁导率
    # 需要做空间 mu 分布
    # --------------------------------------------------------

    Hx_r = dAr_dy / (4 * np.pi * 1e-7)
    Hy_r = -dAr_dx / (4 * np.pi * 1e-7)

    Hx_i = dAi_dy / (4 * np.pi * 1e-7)
    Hy_i = -dAi_dx / (4 * np.pi * 1e-7)

    # 磁场模值

    H_abs = np.sqrt(
        Hx_r**2 + Hy_r**2
        +
        Hx_i**2 + Hy_i**2
    )

    # ========================================================
    # 转换坐标到 m
    # ========================================================

    X_m = X / 100.0
    Y_m = Y / 100.0

    # ========================================================
    # 绘图
    # ========================================================

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10)
    )

    titles = [

        "Real Part $A_r$ (Wb/m)",
        "Imaginary Part $A_i$ (Wb/m)",
        "Magnitude $|A|$ (Wb/m)",

        "Real Magnetic Field $H_r$ (A/m)",
        "Imaginary Magnetic Field $H_i$ (A/m)",
        "Magnitude $|H|$ (A/m)",
    ]

    datasets = [

        Ar,
        Ai,
        A_abs,

        np.sqrt(Hx_r**2 + Hy_r**2),
        np.sqrt(Hx_i**2 + Hy_i**2),
        H_abs
    ]

    # ========================================================
    # 逐图绘制
    # ========================================================

    for ax, data, title in zip(
        axes.flatten(),
        datasets,
        titles
    ):

        c = ax.contourf(
            X_m,
            Y_m,
            data,
            levels=100,
            cmap="jet"
        )

        fig.colorbar(
            c,
            ax=ax,
            format="%.2e"
        )

        ax.set_title(title)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        # ====================================================
        # 绘制铁芯区域
        # ====================================================

        ax.plot(
            [0, 0.004, 0.004, 0, 0],
            [0.003, 0.003, 0.007, 0.007, 0.003],
            "w--",
            linewidth=1.5
        )

        # ====================================================
        # 绘制铜线区域
        # ====================================================

        ax.plot(
            [0.001, 0.002, 0.002, 0.001, 0.001],
            [0.004, 0.004, 0.006, 0.006, 0.004],
            "k--",
            linewidth=1.5
        )

    plt.tight_layout()

    # ========================================================
    # 保存图片
    # ========================================================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path = os.path.join(
        _RESULTS_DIR,
        f"field_result_{timestamp}.png"
    )

    plt.savefig(
        save_path,
        dpi=300
    )

    print(f"\n图像已保存:\n{save_path}")

    # ========================================================
    # 输出最大值
    # ========================================================

    print("\n========== 场最大值 ==========")

    print(f"max |A| = {np.max(A_abs):.4e} Wb/m")

    print(f"max |H| = {np.max(H_abs):.4e} A/m")

    plt.show()

# ============================================================
# 执行绘图
# ============================================================

predict_and_plot(model)