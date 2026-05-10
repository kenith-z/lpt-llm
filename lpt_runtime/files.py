"""运行时文件写入与轻量完整性检查工具。"""

from __future__ import annotations

from pathlib import Path
from time import sleep
from uuid import uuid4
import zipfile

import torch


def _temporary_sibling_path(target_path):
    target_path = Path(target_path)
    suffix = f".tmp.{uuid4().hex}"
    return target_path.with_name(f"{target_path.name}{suffix}")


def _cleanup_file(path):
    path = Path(path)
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _ensure_non_empty(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise OSError(f"文件写入结果为空: {path}")


def atomic_torch_save(payload, path, *, retries=1):
    """先写同目录临时文件，成功后再替换目标文件。"""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(int(retries) + 1):
        temp_path = _temporary_sibling_path(target_path)
        try:
            torch.save(payload, temp_path)
            _ensure_non_empty(temp_path)
            temp_path.replace(target_path)
            return target_path
        except Exception as exc:
            last_error = exc
            _cleanup_file(temp_path)
            if attempt < int(retries):
                sleep(1.0)
    raise RuntimeError(f"原子保存 PyTorch 文件失败: {target_path}") from last_error


def atomic_write_text(path, text, *, encoding="utf-8"):
    """先写临时文本文件，成功后再替换目标文件。"""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_sibling_path(target_path)
    try:
        temp_path.write_text(text, encoding=encoding)
        _ensure_non_empty(temp_path)
        temp_path.replace(target_path)
        return target_path
    except Exception:
        _cleanup_file(temp_path)
        raise


def is_torch_save_file_readable(path):
    """轻量检查 torch.save 默认 zip 文件是否至少有可读中央目录。"""
    target_path = Path(path)
    if not target_path.is_file() or target_path.stat().st_size <= 0:
        return False
    try:
        with zipfile.ZipFile(target_path) as archive:
            return bool(archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False
