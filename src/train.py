"""
端到端 PINN 训练：二维磁矢势 A 的泊松方程 ∇²A = -μ₀J，归一化域 [0,1]²。
流程：Adam 大规模迭代 → L-BFGS 精修 → 保存权重与 JSON 清单 → 绘制 A 与 |H|。
"""

# =============================================
# 导入必要的库
# =============================================
import os                      # 文件路径操作
import numpy as np             # 数值计算
import torch                   # PyTorch 深度学习框架
import torch.nn as nn          # 神经网络模块
import matplotlib.pyplot as plt # 绘图可视化
import deepxde as dde          # 物理信息神经网络库

# 从 training_metadata 模块导入训练元数据相关函数：
# - collect_environment: 收集运行环境信息
# - save_manifest: 保存训练清单到 JSON 文件
# - summarize_losshistory: 汇总损失历史
# - summarize_train_state: 汇总训练状态
from training_metadata import (
    collect_environment,
    save_manifest,
    summarize_losshistory,
    summarize_train_state,
)

# 项目根目录下的 models/ 存放 .pt 与训练清单，路径与当前工作目录无关
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 获取项目根目录路径
_MODEL_DIR = os.path.join(_ROOT, "models")  # 模型保存目录
os.makedirs(_MODEL_DIR, exist_ok=True)  # 确保模型目录存在，不存在则创建

# 设置 DeepXDE 后端为 PyTorch
os.environ["DDE_BACKEND"] = "pytorch"

# =============================================
# 物理常数定义（与无量纲化 PDE 中的 J0 等配合使用）
# =============================================
mu0 = 4 * np.pi * 1e-9  # 真空磁导率，单位与文献中 J、L 的取法一致
L = 1                   # 特征长度，用于后处理把无量纲坐标换回物理尺度
J0 = 1000.0             # 参考电流密度，源项 f_model 的输出会除以 J0 进入方程
A0 = mu0 * J0 * L ** 2  # 特征磁矢势，用于把网络输出还原到物理量

# float32：Adam + GPU 上通常更快；若损失震荡可改为 float64（会更慢）
dde.config.set_default_float("float32")

# =============================================
# 计算域定义：单位正方形，坐标为无量纲 x,y ∈ [0,1]
# =============================================
geom = dde.geometry.Rectangle([0, 0], [1, 1])

# =============================================
# 边界条件函数定义
# 定义四个边界：左边界(x=0)、右边界(x=1)、上边界(y=1)、下边界(y=0)
# =============================================
def boundary_left(x, on_boundary):   return on_boundary and dde.utils.isclose(x[0], 0.0)   # 左边界 x=0
def boundary_right(x, on_boundary):  return on_boundary and dde.utils.isclose(x[0], 1.0)  # 右边界 x=1
def boundary_top(x, on_boundary):    return on_boundary and dde.utils.isclose(x[1], 1.0)  # 上边界 y=1
def boundary_bottom(x, on_boundary): return on_boundary and dde.utils.isclose(x[1], 0.0)  # 下边界 y=0


# =============================================
# 电流源模型：矩形区域内均匀电流密度（阶跃近似）
# 模拟绕组/导体窗口一类源项
# =============================================
class HardRectangleStep(nn.Module):
    """矩形区域内为均匀电流密度（阶跃近似），区域外为 0；模拟绕组/导体窗口一类源项。"""

    def __init__(self, x_min=0.1, x_max=0.2, y_min=0.4, y_max=0.6, scale=1000.0):
        """
        初始化矩形电流源模型
        参数:
            x_min, x_max: 矩形区域在 x 方向的起止位置
            y_min, y_max: 矩形区域在 y 方向的起止位置
            scale: 电流密度缩放因子
        """
        super().__init__()
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.scale = scale

    def forward(self, x):
        """
        前向传播：计算各点的电流密度
        参数:
            x: 输入坐标，形状为 (N, 2)，每行包含 (x, y) 坐标
        返回:
            电流密度值，在矩形区域内为 scale，区域外为 0
        """
        x_c, y_c = x[:, 0], x[:, 1]  # 分离 x, y 坐标
        # 创建掩码：点在矩形区域内为 True，区域外为 False
        mask = (x_c >= self.x_min) & (x_c <= self.x_max) & \
               (y_c >= self.y_min) & (y_c <= self.y_max)
        # 返回电流密度：区域内为 scale，区域外为 0
        return (mask.float() * self.scale).unsqueeze(1).to(x.device)


f_model = HardRectangleStep()  # 实例化电流源模型


# =============================================
# 泊松方程 PDE 定义：∇²A = -μ₀J
# 其中 A 是磁矢势，J 是电流密度
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
    # y 为网络输出的标量场（磁矢势的无量纲形式）
    # 残差对应 ∇²y + f/J0 = 0
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)  # ∂²y/∂x²
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)  # ∂²y/∂y²
    f_val = f_model(x) / J0  # 归一化的电流密度源项
    return -(dy_xx + dy_yy) - f_val  # 泊松方程残差


# =============================================
# 边界条件定义：四边 Dirichlet 条件 A=0
# 理想磁导体近似或远场归零类边界条件
# =============================================
bcs = [
    dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top),    # 上边界 A=0
    dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom), # 下边界 A=0
    dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left),   # 左边界 A=0
    dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right),  # 右边界 A=0
]

# =============================================
# 构建 PINN 训练数据
# 域内配点数 / 边界配点数 / 测试点：越大通常越稳但更慢
# =============================================
data = dde.data.PDE(geom, pde, bcs, num_domain=10000, num_boundary=5000, num_test=5000)

# =============================================
# 构建神经网络模型：全连接前馈网络
# 输入: (x, y) 二维坐标，输出: 1 维磁矢势 A
# swish 激活函数在 PINN 中常用
# =============================================
net = dde.nn.FNN([2] + [128] * 6 + [1], "swish", "Glorot normal")  # 7层隐藏层，每层128神经元
model = dde.Model(data, net)  # 创建 DeepXDE 模型

# =============================================
# GPU 配置与训练设备选择
# DeepXDE 在 GPU 上采点与反传；L-BFGS 阶段会切回 CPU
# =============================================
if torch.cuda.is_available():
    model.net.to(torch.device("cuda"))  # 将模型移到 GPU
    torch.set_float32_matmul_precision('high')  # 设置矩阵乘法精度
    print(f"✓ 使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("警告：GPU 不可用，使用 CPU")

# =============================================
# 训练超参数设置
# =============================================
ADAM_LR = 0.001          # Adam 优化器学习率
ADAM_ITERATIONS = 20000  # Adam 迭代次数

# L-BFGS 参数说明：
# maxcor: L-BFGS 保留的曲率对数量
# gtol/ftol: 梯度与函数值停止准则（0 表示不按该项停）
# maxiter: L-BFGS 内层迭代上限
LBFGS_OPTIONS = {"maxcor": 100, "ftol": 0, "gtol": 1e-13, "maxiter": 30000}
train_device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================
# 阶段1：Adam 优化器全局搜索
# 步数多、对学习率不极端敏感，适合大规模迭代
# =============================================
model.compile("adam", lr=ADAM_LR)  # 使用 Adam 优化器编译模型
print(f"开始 Adam {ADAM_ITERATIONS} 次迭代...")
losshistory_adam, train_state_adam = model.train(iterations=ADAM_ITERATIONS)  # 开始训练
torch.save(model.net.state_dict(), os.path.join(_MODEL_DIR, "eshape_adam.pt"))  # 保存 Adam 训练结果

# =============================================
# 阶段2：L-BFGS 精细优化
# 在 Adam 解的基础上快速下降残差，常能压到更小 PDE 损失
# =============================================
print("切回 CPU 准备 L-BFGS...")
model.net.to(torch.device("cpu"))  # L-BFGS 在 CPU 上运行

# 配置 L-BFGS 优化器参数
dde.optimizers.config.set_LBFGS_options(**LBFGS_OPTIONS)
model.compile("L-BFGS")  # 使用 L-BFGS 编译模型
losshistory_lbfgs, train_state_lbfgs = model.train()  # 继续训练
torch.save(model.net.state_dict(), os.path.join(_MODEL_DIR, "eshape_final.pt"))  # 保存最终模型
print("✓ 所有训练任务已完成！")

# =============================================
# 保存训练清单与元数据
# 将优化器、网络结构、采样数、最终损失等写入 models/eshape_train_manifest.json
# 便于复现与对比实验
# =============================================
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


# =============================================
# 预测与可视化函数
# 由磁矢势 A 计算磁场强度 H：二维下 H_x = (1/μ₀)∂A/∂y，H_y = -(1/μ₀)∂A/∂x
# 然后绘制 A 与 |H| 的分布图
# =============================================
def predict_and_plot():
    """生成磁场分布图，包括磁矢势 A 和磁场强度 |H|"""
    print("生成磁场分布图...")
    model.net.to(torch.device("cpu"))  # 确保模型在 CPU 上

    # 创建 100x100 的网格点用于预测
    x = np.linspace(0, 1, 100)  # x 方向采样点
    y = np.linspace(0, 1, 100)  # y 方向采样点
    X, Y = np.meshgrid(x, y)  # 生成网格
    points = np.vstack([X.ravel(), Y.ravel()]).T.astype(np.float32)  # 展平为 (10000, 2) 的坐标矩阵

    # 进行预测
    y_pred = model.predict(points)  # 形状为 (10000, 1)

    # *10000*A0：与当前无量纲标定一致，把网络输出换到 Wb/m 量级便于和 COMSOL 对比
    A_phys = y_pred.reshape(X.shape) * 10000 * A0  # 转换为物理量级的磁矢势
    dA_dy, dA_dx = np.gradient(A_phys, y * L, x * L)  # 计算 A 的梯度
    H_mag = np.sqrt((dA_dy / mu0) ** 2 + (-dA_dx / mu0) ** 2)  # 计算磁场强度大小 |H|

    # 绘制图形
    plt.figure(figsize=(12, 5))

    # 左图：磁矢势 A 的等高线图
    plt.subplot(1, 2, 1)
    plt.contourf(X * L, Y * L, A_phys, levels=300, cmap="jet")  # 300 级等高线
    plt.colorbar(label="|A| (Wb/m)")  # 颜色条
    plt.title("Magnetic Vector Potential A")  # 标题

    # 右图：磁场强度 |H| 的等高线图
    plt.subplot(1, 2, 2)
    plt.contourf(X * L, Y * L, H_mag, levels=500, cmap="jet")  # 500 级等高线
    plt.colorbar(label="|H| (A/m)")  # 颜色条
    plt.title("Magnetic Field |H|")  # 标题

    plt.tight_layout()  # 调整布局
    plt.show()  # 显示图像


# 执行预测与绘图
predict_and_plot()