"""训练运行清单：与权重一并写入 models/，便于复现实验与对接 MLOps。

约定：
- 使用 JSON（UTF-8），顶层含 schema_version，run.finished_at 为 UTC ISO8601。
- collect_environment：记录软件栈，避免“同一代码不同环境结果不可比”。
- summarize_*：把 DeepXDE 对象压成可 json.dump 的纯量，避免把整个 history 塞进文件。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict

import importlib.metadata as _im


# 若以后字段含义不兼容，递增此版本并在读侧做分支
SCHEMA_VERSION = "1.0"


def _pkg_version(name: str) -> str:
    """pip 安装的发行版版本；未安装或查不到时返回 unknown。"""
    try:
        return _im.version(name)
    except Exception:
        return "unknown"


def collect_environment(torch_mod: Any, dde_mod: Any) -> Dict[str, Any]:
    """Python / PyTorch / DeepXDE / CUDA 信息，写入清单的 environment 段。"""
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "torch": getattr(torch_mod, "__version__", "unknown"),
        "deepxde": getattr(dde_mod, "__version__", _pkg_version("deepxde")),
        "numpy": _pkg_version("numpy"),
        "cuda_available": bool(torch_mod.cuda.is_available()),
    }
    if torch_mod.cuda.is_available():
        env["cuda_device_name"] = torch_mod.cuda.get_device_name(0)
    return env


def summarize_losshistory(losshistory: Any) -> Dict[str, Any]:
    """从 LossHistory 取各损失曲线的最后值与长度；不保存完整曲线以控制文件大小。"""
    out: Dict[str, Any] = {}
    if losshistory is None:
        return out
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
            last = seq[-1]
            out[f"{key}_final"] = float(last)
            out[f"{key}_steps"] = int(n)
        except (TypeError, ValueError):
            continue
    return out


def summarize_train_state(train_state: Any) -> Dict[str, Any]:
    """DeepXDE TrainState 中的最优步、最优损失等；属性不存在则跳过。"""
    if train_state is None:
        return {}
    out: Dict[str, Any] = {}
    for attr in ("best_step", "best_loss", "best_y"):
        if hasattr(train_state, attr):
            v = getattr(train_state, attr)
            try:
                if attr == "best_y" and v is not None:
                    out[attr] = float(v[0][0]) if hasattr(v, "__len__") and len(v) else None
                elif v is not None:
                    out[attr] = float(v) if attr == "best_loss" else int(v)
            except (TypeError, ValueError, IndexError):
                pass
    return out


def save_manifest(model_dir: str, filename: str, manifest: Dict[str, Any]) -> str:
    """写入 UTF-8 JSON（indent=2）；自动补 schema_version 与 run.finished_at（UTC）。"""
    os.makedirs(model_dir, exist_ok=True)
    manifest = dict(manifest)
    manifest.setdefault("schema_version", SCHEMA_VERSION)
    run = dict(manifest.get("run") or {})
    run.setdefault("finished_at", datetime.now(timezone.utc).isoformat())
    manifest["run"] = run

    path = os.path.join(model_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path
