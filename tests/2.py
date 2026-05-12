import matplotlib.pyplot as plt
import numpy as np

# =============================================
# 1. 数据提取与预处理 (根据最新日志)
# =============================================
# 提取步骤
steps = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
         11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000,
         20000, 21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000,
         29000, 30000, 30656]

# 训练损失 (PDE + 各项 BC 的总和)
train_loss = [1.97e-02, 3.65e-03, 3.56e-03, 3.49e-03, 3.56e-03, 3.53e-03,
              3.46e-03, 3.51e-03, 3.36e-03, 2.53e-03, 1.57e-03, 1.23e-03,
              7.05e-04, 1.44e-03, 1.55e-03, 5.01e-04, 2.66e-04, 2.66e-04,
              1.96e-05, 1.55e-05, 6.19e-04, 3.69e-04, 1.16e-04, 3.73e-05,
              1.20e-05, 1.42e-05, 2.97e-05, 5.25e-06, 2.85e-06, 1.76e-06,
              4.64e-06, 5.14e-07]

# 测试损失 (用于评估泛化能力)
test_loss = [2.51e-02, 5.16e-03, 5.04e-03, 5.15e-03, 5.08e-03, 5.11e-03,
             4.99e-03, 5.73e-03, 5.16e-03, 3.55e-03, 1.19e-03, 9.17e-04,
             1.81e-04, 1.20e-03, 8.23e-04, 2.39e-04, 5.51e-04, 2.29e-03,
             1.44e-03, 1.56e-03, 1.98e-04, 2.04e-04, 1.64e-04, 2.73e-04,
             2.86e-04, 3.26e-04, 2.58e-04, 3.28e-04, 2.69e-04, 2.60e-04,
             2.21e-04, 2.84e-04]

# =============================================
# 2. 统一风格绘图 (DeepXDE Style)
# =============================================
plt.figure(figsize=(10, 7), dpi=100)

# 绘制曲线：蓝色实线-训练集，红色虚线-测试集
plt.semilogy(steps, train_loss, color="#0000FF", linestyle="-", marker="o",
             markersize=3, label="Train loss", linewidth=1.2)


# 设置 Adam 与 L-BFGS 分隔线 (30000 步切换)
plt.axvline(x=30000, color="black", linestyle=":", linewidth=1.5)
plt.text(15000, plt.gca().get_ylim()[1]*0.3, "Adam Stage",
         fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.6))
plt.text(30328, plt.gca().get_ylim()[1]*0.3, "L-BFGS",
         fontsize=10, ha='left', color='darkred', fontweight='bold')

# 图形细节配置
plt.xlabel("Steps", fontsize=13)
plt.ylabel("Loss", fontsize=13)
plt.title("Convergence History (Multi-material PINN)", fontsize=15)
plt.legend(loc="upper right", frameon=True, shadow=True, fancybox=True, fontsize=11)
plt.grid(True, which="both", linestyle="--", alpha=0.4)

# 限制坐标轴显示
plt.xlim(-500, 31500)
plt.tick_params(labelsize=11)

plt.tight_layout()
plt.show()