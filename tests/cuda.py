import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import deepxde as dde
from scipy.interpolate import griddata
import re

# =============================================
# 1. 物理常数与路径 (保持一致)
# =============================================
L_cm = 1.0
mu0_cm = 4 * np.pi * 1e-9
J0_cm = 1000.0
A0_cm = mu0_cm * J0_cm * (L_cm ** 2)
PHYS_SCALE = A0_cm * 100

_MODEL_PATH = "eshape_final_20260509_011158.pt"
_DATA_PATH = "Untitled.txt"


def get_model():
    return dde.nn.FNN([2] + [64] * 6 + [2], "swish", "Glorot normal")


# =============================================
# 2. 高分辨率预测
# =============================================
def load_and_predict_high_res(model_path, resolution=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_model()

    print(f"正在加载模型并进行高分辨率推理 (Res: {resolution}x{resolution})...")
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint)
    net.to(device).eval()

    # 创建更高密度的网格
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    pts = np.vstack([X.ravel(), Y.ravel()]).T.astype(np.float32)

    pts_tensor = torch.from_numpy(pts).to(device)
    with torch.no_grad():
        # 分批处理以防止大分辨率下的显存溢出
        prediction = []
        batch_size = 50000
        for i in range(0, len(pts_tensor), batch_size):
            batch_pts = pts_tensor[i: i + batch_size]
            prediction.append(net(batch_pts).cpu().numpy())
        prediction = np.vstack(prediction)

    Ar_raw = prediction[:, 0].reshape(resolution, resolution)
    return X, Y, Ar_raw


# =============================================
# 3. 数据解析
# =============================================
def load_comsol_real_only(file_path):
    coords, real_values = [], []
    pattern = re.compile(
        r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('%') or not line.strip(): continue
            match = pattern.search(line)
            if match:
                x, y, real_val = match.groups()
                coords.append([float(x), float(y)])
                real_values.append(float(real_val))
    return np.array(coords), np.array(real_values)


# =============================================
# 4. 高清可视化分析
# =============================================
def analyze_high_res(model_path, data_path):
    # A. 获取高分辨率 PINN 结果
    X, Y, Ar_raw = load_and_predict_high_res(model_path, resolution=500)
    Ar_pinn = Ar_raw * PHYS_SCALE

    # B. 获取 COMSOL 参考值并插值
    ref_coords, ref_real = load_comsol_real_only(data_path)
    # 使用 cubic 插值保持高分辨率下的平滑度
    Ar_ref_interp = griddata(ref_coords, ref_real, (X, Y), method='cubic', fill_value=0)

    # C. 计算误差
    error_abs = np.abs(Ar_pinn - Ar_ref_interp)

    # D. 绘图 (提高 DPI 和 图像尺寸)
    plt.rcParams['figure.dpi'] = 120  # 提高屏幕显示清晰度
    fig, ax = plt.subplots(1, 3, figsize=(22, 6))

    # 设置通用的颜色映射参数，确保对比公平
    v_min = min(Ar_ref_interp.min(), Ar_pinn.min())
    v_max = max(Ar_ref_interp.max(), Ar_pinn.max())

    # 1. COMSOL
    im0 = ax[0].pcolormesh(X, Y, Ar_ref_interp, shading='auto', cmap='jet', vmin=v_min, vmax=v_max)
    ax[0].set_title("COMSOL Reference (High-Res)", fontsize=14)
    fig.colorbar(im0, ax=ax[0])

    # 2. PINN
    im1 = ax[1].pcolormesh(X, Y, Ar_pinn, shading='auto', cmap='jet', vmin=v_min, vmax=v_max)
    ax[1].set_title("PINN Prediction (High-Res)", fontsize=14)
    fig.colorbar(im1, ax=ax[1])

    # 3. 误差图 (使用更敏感的色彩表)
    im2 = ax[2].pcolormesh(X, Y, error_abs, shading='auto', cmap='inferno')
    ax[2].set_title("Absolute Error (Detailed)", fontsize=14)
    fig.colorbar(im2, ax=ax[2])

    for a in ax:
        a.set_xlabel("x (cm)")
        a.set_ylabel("y (cm)")
        a.set_aspect('equal')
        # 细化坐标轴刻度
        a.set_xticks(np.arange(0, 1.1, 0.2))
        a.set_yticks(np.arange(0, 1.1, 0.2))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    if os.path.exists(_MODEL_PATH) and os.path.exists(_DATA_PATH):
        analyze_high_res(_MODEL_PATH, _DATA_PATH)
    else:
        print("请检查文件路径。")