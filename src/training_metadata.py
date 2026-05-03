"""
训练运行清单模块
功能说明：
- 与权重一并写入 models/ 目录，便于复现实验与对接 MLOps
- 生成 JSON 格式的 UTF-8 文件，包含 schema_version 和 run.finished_at (UTC ISO8601)
- collect_environment：记录软件栈，避免"同一代码不同环境结果不可比"
- summarize_*：将 DeepXDE 对象压缩为可 json.dump 的纯量，避免保存完整训练历史

约定：
- 使用 JSON（UTF-8）格式
- 顶层包含 schema_version 字段
- run.finished_at 为 UTC ISO8601 时间戳
"""

from __future__ import annotations

import json          # JSON 数据格式处理
import os            # 文件路径操作
import sys           # 系统信息获取
from datetime import datetime, timezone  # 日期时间处理（带时区支持）
from typing import Any, Dict  # 类型提示

import importlib.metadata as _im  # 获取包版本信息


# =============================================
# 模式版本号
# 若以后字段含义不兼容，递增此版本并在读侧做分支
# =============================================
SCHEMA_VERSION = "1.0"


# =============================================
# 辅助函数：获取包的版本信息
# =============================================
def _pkg_version(name: str) -> str:
    """
    获取 pip 安装的发行版版本
    参数:
        name: 包名称
    返回:
        版本字符串；若未安装或查不到则返回 "unknown"
    """
    try:
        return _im.version(name)  # 尝试获取包版本
    except Exception:
        return "unknown"  # 获取失败时返回 unknown


# =============================================
# collect_environment 函数
# 收集 Python/PyTorch/DeepXDE/CUDA 信息，写入清单的 environment 段
# =============================================
def collect_environment(torch_mod: Any, dde_mod: Any) -> Dict[str, Any]:
    """
    收集运行环境信息，用于记录训练环境配置
    参数:
        torch_mod: PyTorch 模块
        dde_mod: DeepXDE 模块
    返回:
        包含环境信息的字典
    """
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],         # Python 版本
        "platform": sys.platform,                 # 操作系统平台
        "torch": getattr(torch_mod, "__version__", "unknown"),  # PyTorch 版本
        "deepxde": getattr(dde_mod, "__version__", _pkg_version("deepxde")),  # DeepXDE 版本
        "numpy": _pkg_version("numpy"),           # NumPy 版本
        "cuda_available": bool(torch_mod.cuda.is_available()),  # CUDA 是否可用
    }
    # 如果 CUDA 可用，添加 GPU 设备名称
    if torch_mod.cuda.is_available():
        env["cuda_device_name"] = torch_mod.cuda.get_device_name(0)
    return env


# =============================================
# summarize_losshistory 函数
# 从 LossHistory 取各损失曲线的最后值与长度
# 不保存完整曲线以控制文件大小
# =============================================
def summarize_losshistory(losshistory: Any) -> Dict[str, Any]:
    """
    汇总损失历史信息
    参数:
        losshistory: DeepXDE 的 LossHistory 对象
    返回:
        包含损失历史摘要的字典（最后值和步数）
    """
    out: Dict[str, Any] = {}
    if losshistory is None:
        return out

    # 遍历常见的损失键名
    for key in ("loss_train", "loss_test", "loss_test_1"):
        if not hasattr(losshistory, key):
            continue
        seq = getattr(losshistory, key)
        if seq is None:
            continue
        try:
            n = len(seq)
            if n == 0:
                continue
            last = seq[-1]  # 取最后一个值
            out[f"{key}_final"] = float(last)  # 最终损失值
            out[f"{key}_steps"] = int(n)        # 训练步数
        except (TypeError, ValueError):
            continue
    return out


# =============================================
# summarize_train_state 函数
# DeepXDE TrainState 中的最优步、最优损失等
# 属性不存在则跳过
# =============================================
def summarize_train_state(train_state: Any) -> Dict[str, Any]:
    """
    汇总训练状态信息
    参数:
        train_state: DeepXDE 的 TrainState 对象
    返回:
        包含训练状态摘要的字典
    """
    if train_state is None:
        return {}
    out: Dict[str, Any] = {}

    # 遍历常见的状态属性
    for attr in ("best_step", "best_loss", "best_y"):
        if hasattr(train_state, attr):
            v = getattr(train_state, attr)
            try:
                if attr == "best_y" and v is not None:
                    # best_y 可能有多维，取第一个值
                    out[attr] = float(v[0][0]) if hasattr(v, "__len__") and len(v) else None
                elif v is not None:
                    # best_loss 转为 float，best_step 转为 int
                    out[attr] = float(v) if attr == "best_loss" else int(v)
            except (TypeError, ValueError, IndexError):
                pass
    return out


# =============================================
# save_manifest 函数
# 写入 UTF-8 JSON（indent=2）文件
# 自动补 schema_version 与 run.finished_at（UTC）
# =============================================
def save_manifest(model_dir: str, filename: str, manifest: Dict[str, Any]) -> str:
    """
    保存训练清单到 JSON 文件
    参数:
        model_dir: 模型目录路径
        filename: 清单文件名
        manifest: 清单内容字典
    返回:
        保存的文件路径
    """
    os.makedirs(model_dir, exist_ok=True)  # 确保目录存在

    manifest = dict(manifest)  # 复制字典以避免修改原数据
    manifest.setdefault("schema_version", SCHEMA_VERSION)  # 添加 schema_version（如不存在）

    # 处理 run 字段：添加 finished_at 时间戳（如不存在）
    run = dict(manifest.get("run") or {})
    run.setdefault("finished_at", datetime.now(timezone.utc).isoformat())
    manifest["run"] = run

    # 写入 JSON 文件
    path = os.path.join(model_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)  # indent=2 格式化，ensure_ascii=False 支持中文
        f.write("\n")  # 追加换行符
    return path
