"""
第二阶段 L-BFGS 精细优化脚本
功能说明：
- 在已有 Adam 训练权重的基础上，使用 L-BFGS 算法进一步精细优化
- 本脚本采用 float64 + OperatorBC + [80]*4 网络结构（与 train.py 的 [128]*6、仅 Dirichlet 不同）
- 加载权重文件 eshapeadam5.pt（需与下方 net 定义一致）
"""

# =============================================
# 导入必要的库
# =============================================
import os                      # 文件路径操作
import deepxde as dde          # 物理信息神经网络库
import numpy as np             # 数值计算
import torch                   # PyTorch 深度学习框架
import torch.nn as nn          # 神经网络模块
import matplotlib.pyplot as plt # 绘图可视化

# 从 training_metadata 模块导入训练元数据相关函数
from training_metadata import (
    collect_environment,
    save_manifest,
    summarize_losshistory,
    summarize_train_state,
)

# =============================================
# 路径设置
# =============================================
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
_MODEL_DIR = os.path.join(_ROOT, "models")  # 模型保存目录
os.makedirs(_MODEL_DIR, exist_ok=True)  # 确保模型目录存在

# =============================================
# 物理参数定义（与 train.py 保持一致）
# =============================================
mu0 = 4 * np.pi * 1e-9  # 真空磁导率 (H/cm)
L = 1                   # 特征长度 (m)
J0 = 1000.0            # 特征电流密度 (A/cm^2)
A0 = mu0 * J0 * L**2   # 特征磁矢势 (Wb/m)

# float64：二阶优化与病态 Hessian 时常用，减少数值噪声
dde.config.set_default_float("float64")
geom = dde.geometry.Rectangle([0, 0], [1, 1])  # 定义单位正方形计算域

# =============================================
# 边界条件函数定义
# 用于把边界条件挂到 x=0/1、y=0/1 上
# =============================================
def boundary_left(x, on_boundary):
    """左边界 x=0"""
    return on_boundary and dde.utils.isclose(x[0], 0.0)

def boundary_right(x, on_boundary):
    """右边界 x=1"""
    return on_boundary and dde.utils.isclose(x[0], 1.0)

def boundary_top(x, on_boundary):
    """上边界 y=1"""
    return on_boundary and dde.utils.isclose(x[1], 1.0)

def boundary_bottom(x, on_boundary):
    """下边界 y=0"""
    return on_boundary and dde.utils.isclose(x[1], 0.0)


# =============================================
# 电流源模型：矩形载流区域
# 与 train.py 中类等价，写法略有不同
# =============================================
class HardRectangleStep(nn.Module):
    """矩形区域内为均匀电流密度（阶跃近似），区域外为 0"""

    def __init__(self, x_min=0.1, x_max=0.2, y_min=0.4, y_max=0.6, scale=1000.0):
        """
        初始化矩形电流源模型
        参数:
            x_min, x_max: 矩形区域在 x 方向的起止位置
            y_min, y_max: 矩形区域在 y 方向的起止位置
            scale: 电流密度缩放因子
        """
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.scale = scale

    def forward(self, x):
        """
        前向传播：计算各点的电流密度
        参数:
            x: 输入坐标，形状为 (N, 2)，每行包含 (x, y) 坐标
        返回:
            电流密度值，在矩形区域内为 scale * fx * fy，区域外为 0
        """
        x_coord = x[:, 0]  # x 坐标
        y_coord = x[:, 1]  # y 坐标
        # 分别判断 x 和 y 是否在各自范围内
        fx = ((x_coord >= self.x_min) & (x_coord <= self.x_max)).float()  # x 方向掩码
        fy = ((y_coord >= self.y_min) & (y_coord <= self.y_max)).float()  # y 方向掩码
        fxy = self.scale * fx * fy  # 二维掩码：需同时满足 x 和 y 都在范围内
        return fxy.unsqueeze(1)  # 调整为 (N, 1) 形状
        
f_model = HardRectangleStep()  # 实例化电流源模型


# =============================================
# 泊松方程 PDE 定义：∇²A = -μ₀J
# =============================================
def pde(x, y):
    """
    计算泊松方程的残差（偏微分方程的弱形式）
    参数:
        x: 空间坐标，形状为 (N, 2)
        y: 神经网络预测的磁矢势值，形状为 (N, 1)
    返回:
        PDE 残差，应为 0（当解满足方程时）
    """
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)  # ∂²y/∂x²
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)  # ∂²y/∂y²
    f_val = f_model(x) / J0  # 归一化的电流密度源项
    return -(dy_xx + dy_yy) - f_val  # 泊松方程残差


# =============================================
# 磁场强度 H 的分量计算函数
# 二维情况下：A 是标量（z 方向），H = ∇ × A
# H_x = ∂A/∂y，H_y = -∂A/∂x
# =============================================
def h_x(x, y, X):
    """
    计算磁场强度 H 的 x 分量
    H 的 x 分量与 ∂A/∂y 相关（2D z 向 A）
    """
    da_y = dde.grad.jacobian(y, x, i=0, j=1)  # ∂y/∂y，即 ∂A/∂y
    return da_y


def h_y(x, y, X):
    """
    计算磁场强度 H 的 y 分量
    H 的 y 分量与 -∂A/∂x 相关
    """
    da_x = dde.grad.jacobian(y, x, i=0, j=0)  # ∂y/∂x，即 ∂A/∂x
    return da_x


def h_yx(x, y, X):
    """
    预留：若需在边界约束 ∂H/∂x 等可用
    计算 ∂²A/∂x²
    """
    da_xx = dde.grad.hessian(y, x, i=0, j=0)  # ∂²y/∂x²
    return da_xx

# =============================================
# 边界条件定义
# - Dirichlet 条件：四边 A=0
# - OperatorBC：对 H 相关算子施加约束（具体物理含义依赖建模）
# =============================================
# Dirichlet 边界条件：磁矢势 A=0
a_top_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top)     # 上边界 A=0
a_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left)   # 左边界 A=0
a_right_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right)  # 右边界 A=0
a_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom)  # 下边界 A=0

# OperatorBC：磁场分量边界约束
hx_bc_r = dde.OperatorBC(geom, h_x, boundary_right)  # 右边界 H_x 约束
hx_bc_l = dde.OperatorBC(geom, h_x, boundary_left)   # 左边界 H_x 约束
hy_bc_t = dde.OperatorBC(geom, h_y, boundary_top)     # 上边界 H_y 约束
hy_bc_b = dde.OperatorBC(geom, h_y, boundary_bottom)  # 下边界 H_y 约束

# 汇总所有边界条件
bcs = [a_top_bc, a_left_bc, a_right_bc, a_bottom_bc, hx_bc_r, hx_bc_l, hy_bc_t, hy_bc_b]

# =============================================
# 构建 PINN 训练数据
# num_domain: 域内配点数，num_boundary: 边界配点数，num_test: 测试点数
# =============================================
data = dde.data.PDE(geom, pde, bcs, num_domain=3000, num_boundary=100, num_test=1500)

# =============================================
# 构建神经网络模型
# 5层隐藏层，每层80神经元（与 train.py 的 [128]*6 不同）
# =============================================
net = dde.nn.FNN([2] + [80] * 4 + [1], "Swish", "Glorot normal")  # 定义网络结构
model = dde.Model(data, net)  # 创建 DeepXDE 模型

# =============================================
# 加载预训练的 Adam 模型权重
# 注意：必须已有同名权重文件 eshapeadam5.pt
# 若你从 train.py 只得到 eshape_adam.pt，需复制/改名为 eshapeadam5.pt
# =============================================
print("正在加载模型...")
model.net.load_state_dict(torch.load(os.path.join(_MODEL_DIR, "eshapeadam5.pt")))
print("✓ 模型加载成功！")

# =============================================
# L-BFGS 优化器参数设置
# maxcor: L-BFGS 保留的曲率对数量
# gtol/ftol: 收敛阈值
# maxiter/maxfun/maxls: 最大迭代/函数求值/线搜索次数
# =============================================
LBFGS_OPTIONS = {
    "maxcor": 100,     # 保留的曲率对数量
    "ftol": 0,          # 函数值变化阈值（0表示不按该项停）
    "gtol": 1e-13,      # 梯度范数阈值
    "maxiter": 40000,   # 最大迭代次数
    "maxfun": 100000,   # 最大函数求值次数
    "maxls": 50,        # 最大线搜索步数
}

# 配置 L-BFGS 并开始训练
dde.optimizers.config.set_LBFGS_options(**LBFGS_OPTIONS)
model.compile("L-BFGS")  # 使用 L-BFGS 编译模型
losshistory_lbfgs, train_state_lbfgs = model.train()  # 开始训练

# 保存 loss 曲线等图到当前工作目录
dde.saveplot(losshistory_lbfgs, train_state_lbfgs)

# =============================================
# 保存精修后的模型权重
# =============================================
print("正在保存模型...")
OUT_CKPT = "eshapeadamLBFGS50005.pt"  # 输出权重文件名
torch.save(model.net.state_dict(), os.path.join(_MODEL_DIR, OUT_CKPT))
print("✓ 模型保存成功！")

# =============================================
# 保存训练清单与元数据
# =============================================
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


# =============================================
# 预测与可视化函数
# =============================================
def predict_and_plot():
    """
    生成磁场分布图
    包括磁矢势 A 和磁场强度 |H| 的等高线图
    与 train.py 相同的标定，使 A 处于可与 COMSOL 对比的量级
    """
    # 创建 100x100 的网格点用于预测
    x = np.linspace(0, 1, 100)  # x 方向采样点
    y = np.linspace(0, 1, 100)  # y 方向采样点
    X, Y = np.meshgrid(x, y)  # 生成网格
    points = np.vstack([X.ravel(), Y.ravel()]).T  # 展平为 (10000, 2) 的坐标矩阵

    # 进行预测
    y_pred = model.predict(points)  # 形状为 (10000, 1)

    # 将网络输出转换为物理量级的磁矢势
    A_physical = y_pred.reshape(X.shape) * 10000 * A0
    X_physical = X * L  # 转换为物理坐标
    Y_physical = Y * L

    # 计算磁场强度分量
    dA_dy, dA_dx = np.gradient(A_physical, y * L, x * L)  # 计算 A 的梯度
    H_x = dA_dy / mu0   # H_x = (1/μ₀)∂A/∂y
    H_y = -dA_dx / mu0  # H_y = -(1/μ₀)∂A/∂x
    H_magnitude = np.sqrt(H_x**2 + H_y**2)  # |H| = sqrt(H_x² + H_y²)

    # 绘制图形
    plt.figure(figsize=(12, 5))

    # 左图：磁矢势 A 的等高线图
    plt.subplot(1, 2, 1)
    plt.contourf(X_physical, Y_physical, A_physical, levels=300, cmap="jet")  # 300 级等高线
    plt.colorbar(label="|A| (wb/m)")  # 颜色条
    plt.xlabel("x (cm)")  # x 轴标签
    plt.ylabel("y (cm)")  # y 轴标签
    plt.title("Magnetic Vector Potential A")  # 标题
    plt.axis("equal")  # 等比例坐标轴

    # 右图：磁场强度 |H| 的等高线图
    plt.subplot(1, 2, 2)
    plt.contourf(X_physical, Y_physical, H_magnitude, levels=500, cmap="jet")  # 500 级等高线
    plt.colorbar(label="|H| (A/m)")  # 颜色条
    plt.xlabel("x (m)")  # x 轴标签
    plt.ylabel("y (m)")  # y 轴标签
    plt.title("Magnetic Field |H|")  # 标题
    plt.axis("equal")  # 等比例坐标轴

    plt.tight_layout()  # 调整布局
    plt.show()  # 显示图像


# 执行预测与绘图
predict_and_plot()

# =============================================
# 烟雾测试：检查网络输出是否为有限值
# 在中心点 (0.5, 0.5) 处进行预测
# =============================================
test_point = np.array([[0.5, 0.5]])  # 中心点测试坐标
prediction = model.predict(test_point)  # 进行预测
print(f"在点(0.5, 0.5)处的预测值: {prediction[0][0]:.6f}")  # 打印预测结果