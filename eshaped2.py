#读取参数预测并且绘制出误差的图的程序
#输出结果的主要程序
import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 物理常数和参数
mu0 = 4 * np.pi * 1e-9  # 真空磁导率 (H/cm)
L = 1                # 特征长度 (m)
J0 = 1000.0              # 特征电流密度 (A/cm^2)
A0 = mu0 * J0 * L**2    # 特征磁矢势 (Wb/m)
dde.config.set_default_float("float32")

# 1. 读取COMSOL导出的数据
def read_comsol_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('%') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    az = float(parts[2])
                    data.append([x, y, az])
                except ValueError:
                    continue
    return np.array(data)

# 读取数据
comsol_data = read_comsol_data('eshape.txt')
x_comsol = comsol_data[:, 0]
y_comsol = comsol_data[:, 1]
A_comsol = comsol_data[:, 2]

# 2. 准备神经网络模型
geom = dde.geometry.Rectangle([0, 0], [1, 1])
bcs = []
# 数据和模型（省略，已定义）
data = dde.data.PDE(geom, 0, bcs, num_domain=3000, num_boundary=100, num_test=1500)
net = dde.nn.FNN([2] + [128] * 8 + [1], "Swish", "Glorot normal")
model = dde.Model(data, net)


print("正在加载模型...")
model.net.load_state_dict(torch.load("eshapeadamLBFGS30010.pt", map_location=torch.device('cpu')))
print("✓ 模型加载成功！")

# 3. 创建规则网格（物理单位）
x_phys = np.linspace(min(x_comsol), max(x_comsol), 100)
y_phys = np.linspace(min(y_comsol), max(y_comsol), 100)
X_phys, Y_phys = np.meshgrid(x_phys, y_phys)
points_phys = np.vstack([X_phys.ravel(), Y_phys.ravel()]).T

# 4. 对COMSOL数据进行网格插值
A_comsol_interp = griddata((x_comsol, y_comsol), A_comsol, 
                          (X_phys, Y_phys), method='linear')

# 5. 神经网络预测（归一化坐标 -> 物理单位）
# 归一化坐标 = 物理坐标 / L
points_norm = points_phys / L
A_nn = model.predict(points_norm).reshape(X_phys.shape) * 100 * A0

# 6. 计算误差
error = A_comsol_interp - A_nn
error1 = np.abs(error )

# 7. 创建三子图
plt.figure(figsize=(20, 6))
# 计算统一的最小值和最大值
vmin = min(np.nanmin(A_comsol_interp), np.nanmin(A_nn))
vmax = max(np.nanmax(A_comsol_interp), np.nanmax(A_nn))

# 子图1: COMSOL数据插值结果
plt.subplot(1, 3, 1)
contour1 = plt.contourf(X_phys, Y_phys, A_comsol_interp, 
                       levels=300, 
                       cmap="turbo",
                       vmin=vmin,  # 设置最小值
                       vmax=vmax)  # 设置最大值
cbar1 = plt.colorbar(contour1, label="A (Wb/m)")
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
plt.title("FEM |A| (Wb/m)")
plt.axis('equal')

# 子图2: 神经网络预测结果
plt.subplot(1, 3, 2)
contour2 = plt.contourf(X_phys, Y_phys, A_nn, 
                       levels=300, 
                       cmap="turbo",
                       vmin=vmin,  # 使用相同的min
                       vmax=vmax)  # 使用相同的max
cbar2 = plt.colorbar(contour2, label="A (Wb/m)")
plt.xlabel("x (cm)")
plt.ylabel("")
plt.title("PINN |A| (Wb/m)")
plt.axis('equal')

# 子图3: 误差图
plt.subplot(1, 3, 3)
# 使用对称的颜色映射以更好显示正负误差
contour3 = plt.contourf(X_phys, Y_phys, error1, levels=500, 
                       cmap="turbo",)
plt.colorbar(contour3, label="")
plt.xlabel("x (cm)")
plt.ylabel("")
plt.title("err: FEM - PINN err (Wb/m)")
plt.axis('equal')

# 添加统计信息
max_error = np.nanmax(np.abs(error))
min_error = np.nanmin(error)
mean_error = np.nanmean(error)
rmse = np.sqrt(np.nanmean(error**2))
plt.figtext(0.5, 0.01, 
           f"maxerr: {max_error:.3e} Wb/m | averr: {mean_error:.3e} Wb/m | RMSE: {rmse:.3e} Wb/m",
           ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.7})

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)  # 为底部文本留出空间
plt.savefig('comparison.png', dpi=300)
plt.show()
# 在代码中找到以下部分：

# 计算相对误差用于统计（但不显示在图中）
epsilon = 1e-7
rel_error = error1 / (np.abs(A_comsol_interp) + epsilon)  # 这里已经避免除以零

# 然后修改相对平均误差的计算：
mean_rel_error = np.nanmean(rel_error)  # 使用已经避免除以零的rel_error

# 最后修改打印语句：
print(f"相对平均误差: {mean_rel_error:.2%}")
# 8. 计算并打印统计信息
print("\n误差分析:")
print(f"最大绝对误差: {max_error:.3e} Wb/m")
print(f"最小误差: {min_error:.3e} Wb/m")
print(f"平均误差: {mean_error:.3e} Wb/m")
print(f"均方根误差 (RMSE): {rmse:.3e} Wb/m")
print(f"相对平均误差: {mean_rel_error:.2%}")