#读取comsol文件并且画图的测试程序
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 读取数据文件
data = []
with open('eshape.txt', 'r') as file:
    for line in file:
        # 跳过注释行和空行
        if line.startswith('%') or not line.strip():
            continue
        # 处理数据行
        parts = line.split()
        if len(parts) >= 3:
            try:
                x = float(parts[0])
                y = float(parts[1])
                az = float(parts[2])
                data.append([x, y, az])
            except ValueError:
                continue

# 转换为NumPy数组
data = np.array(data)
x = data[:, 0]
y = data[:, 1]
az = data[:, 2]

# 创建插值网格
xi = np.linspace(min(x), max(x), 200)
yi = np.linspace(min(y), max(y), 200)
X_grid, Y_grid = np.meshgrid(xi, yi)

# 网格插值
Z_grid = griddata((x, y), az, (X_grid, Y_grid), method='cubic')

# 创建图形
plt.figure(figsize=(12, 5))

# 绘制等值线图
plt.subplot(1, 2, 1)
contour = plt.contourf(X_grid, Y_grid, Z_grid, levels=300, cmap="jet")
plt.colorbar(contour, label="|A| (Wb/m)")
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
plt.title("Magnetic Vector Potential A")
plt.axis("equal")

# 添加原始数据点位置图（可选）
plt.subplot(1, 2, 2)
plt.scatter(x, y, c=az, s=5, cmap="jet")
plt.colorbar(label="|A| (Wb/m)")
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
plt.title("Original Measurement Points")
plt.axis("equal")

plt.tight_layout()
plt.show()