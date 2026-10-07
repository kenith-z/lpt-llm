"""大规模 JSONL 数据读取与离线 token cache。

本模块是数据管线的可选旁路，不会改变 :mod:`lpt_data.io` 的旧行为。
它有两个用途：

* 把一个很大的 JSONL 文件按字节范围分给多个 ``spawn`` 进程，避免单个
  Python 主进程承担全部 JSON 解析；
* 把解析后的 token 序列保存为 ragged 分片，训练时以 ``numpy.memmap``
  只读访问，避免每个 epoch 重复解析 JSONL 和分词。

所有 worker 入口都是模块顶层函数，``ProcessPoolExecutor`` 使用显式
``spawn`` 上下文，因此 Windows 和 Linux 都可以 pickle。缓存目录采用
临时目录完成后原子发布，训练中途不会看到半成品 manifest。
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import shutil
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确提示
    np = None
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

from .schema import normalize_dataset_record


TOKEN_CACHE_FORMAT = "lpt_token_cache"
TOKEN_CACHE_SCHEMA_VERSION = 1
DEFAULT_JSONL_CHUNK_BYTES = 32 * 1024 * 1024
DEFAULT_CACHE_SHARD_RECORDS = 100_000
DEFAULT_CACHE_SHARD_TOKENS = 16_000_000


@dataclass(frozen=True, slots=True)
class JSONLChunk:
    """一个不会重复读取记录的 JSONL 字节范围。"""

    source_path: Path
    start_offset: int
    end_offset: int
    file_index: int
    chunk_index: int


@dataclass(frozen=True, slots=True)
class RecordMetadata:
    """记录的稳定来源和长度信息。

    ``record_index`` 是文件内非空、规范化记录的零基序号；它由有序合并器
    修正，因此不会因为 worker 的字节切分而改变。``line_number`` 只在从文件
    开头读取的 chunk 中提供，避免为了计算行号再完整扫描大文件。
    """

    record_id: str
    source_path: str
    record_index: int
    line_number: int | None
    byte_offset: int
    byte_length: int
    text_length: int
    token_length: int | None = None
    # 训练级统计字段：cache 数据集不再回读 JSONL，因此构建时把样本类型和
    # 来源写入每条元数据，供 summary 和进度审计使用。旧 cache 缺省为空。
    record_type: str | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedJSONLRecord:
    """并行 JSONL reader 的可 pickle 返回值。"""

    record: dict[str, Any]
    metadata: RecordMetadata


@dataclass(frozen=True, slots=True)
class TokenizedCacheRecord:
    """cache builder 产出的 token 序列和元数据。"""

    token_ids: tuple[int, ...]
    metadata: RecordMetadata


@dataclass(frozen=True, slots=True)
class _ChunkResult:
    """worker 返回的 chunk 结果，同时保留原始记录数量。"""

    chunk: JSONLChunk
    records: tuple[Any, ...]
    source_record_count: int


@dataclass(frozen=True, slots=True)
class _CacheShardInfo:
    """manifest 中一个 cache 分片的索引信息。"""

    name: str
    records: int
    tokens: int
    min_length: int
    max_length: int


def _require_numpy():
    """延迟检查 numpy，保证只使用 JSONL reader 时仍可轻量导入。"""
    if np is None:  # pragma: no cover - 仅在环境没有 numpy 时执行
        raise RuntimeError(
            "离线 token cache 需要 numpy；请安装 requirements.txt 中的 numpy。"
        ) from _NUMPY_IMPORT_ERROR
    return np


def _normalise_paths(paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]]) -> tuple[Path, ...]:
    """规范化输入路径并保留调用方顺序。"""
    if isinstance(paths, (str, os.PathLike)):
        values = (paths,)
    else:
        values = tuple(paths)
    if not values:
        raise ValueError("paths 不能为空。")
    result = []
    for value in values:
        path = Path(value).expanduser()
        if path.suffix.lower() != ".jsonl":
            raise ValueError(f"只支持 JSONL 文件，当前路径为: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        # absolute() 会消除 ``..``，但不会把 Windows 目录联接改写成另一个盘符；
        # checkpoint/cache 中因此仍能保留项目内的逻辑路径。
        result.append(path.absolute())
    return tuple(result)


def stable_path(path: Path, source_root: Path | None = None) -> str:
    """生成跨平台、可写入 checkpoint 的 POSIX 来源路径。"""
    path = Path(path).absolute()
    if source_root is not None:
        try:
            return path.relative_to(Path(source_root).absolute()).as_posix()
        except ValueError:
            pass
    try:
        return path.relative_to(Path.cwd().absolute()).as_posix()
    except ValueError:
        return path.as_posix()


_stable_path = stable_path


def _record_text_length(record: Mapping[str, Any]) -> int:
    """计算用于分桶的字符长度，不改变训练文本语义。"""
    if record.get("type") == "text":
        return len(str(record.get("text", "")))
    total = 0
    for message in record.get("messages", ()):
        if isinstance(message, Mapping):
            total += len(str(message.get("content", "")))
            total += len(str(message.get("thinking", "")))
    return total


def plan_jsonl_chunks(
    paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    *,
    chunk_bytes: int = DEFAULT_JSONL_CHUNK_BYTES,
) -> tuple[JSONLChunk, ...]:
    """按字节规划 JSONL chunk。

    结束位置会向后移动到完整行的末尾，worker 因而可以独立 ``seek``；超长
    单行会形成一个大于 ``chunk_bytes`` 的 chunk，而不会被截断或重复。
    """
    paths = _normalise_paths(paths)
    chunk_bytes = int(chunk_bytes)
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes 必须为正整数。")

    chunks: list[JSONLChunk] = []
    for file_index, path in enumerate(paths):
        file_size = path.stat().st_size
        start = 0
        chunk_index = 0
        with path.open("rb") as handle:
            while start < file_size:
                target = min(file_size, start + chunk_bytes)
                if target < file_size:
                    handle.seek(target)
                    handle.readline()
                    end = handle.tell()
                else:
                    end = file_size
                # 极端情况下（例如损坏的文件句柄）保护规划器不死循环。
                if end <= start:
                    handle.seek(start)
                    handle.readline()
                    end = handle.tell()
                    if end <= start:
                        raise OSError(f"无法推进 JSONL chunk 边界: {path}:{start}")
                chunks.append(
                    JSONLChunk(
                        source_path=path,
                        start_offset=int(start),
                        end_offset=int(end),
                        file_index=file_index,
                        chunk_index=chunk_index,
                    )
                )
                start = end
                chunk_index += 1
    return tuple(chunks)


def _parse_jsonl_chunk(
    chunk: JSONLChunk,
    *,
    expected_types: tuple[str, ...] | None,
    source_root: Path | None,
) -> tuple[list[ParsedJSONLRecord], int]:
    """读取并规范化一个字节 chunk；该函数由 reader/cache worker 共用。"""
    allowed_types = None if expected_types is None else set(expected_types)
    source_path = Path(chunk.source_path)
    stable_source = _stable_path(source_path, source_root)
    records: list[ParsedJSONLRecord] = []
    source_record_count = 0
    with source_path.open("rb") as handle:
        handle.seek(int(chunk.start_offset))
        if chunk.start_offset > 0:
            # 只有起点落在一行中间时才丢弃首行；若前一个字节是换行，不能误丢整行。
            handle.seek(int(chunk.start_offset) - 1)
            previous_byte = handle.read(1)
            handle.seek(int(chunk.start_offset))
            if previous_byte not in (b"\n", b"\r"):
                handle.readline()

        physical_line_number: int | None = 1 if chunk.start_offset == 0 else None
        while True:
            byte_offset = handle.tell()
            if byte_offset >= int(chunk.end_offset):
                break
            raw_line = handle.readline()
            if not raw_line:
                break
            current_line_number = physical_line_number
            if physical_line_number is not None:
                physical_line_number += 1
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"JSONL 记录解析失败: {source_path}:{current_line_number or byte_offset}"
                ) from exc
            # 使用 byte offset 作为缺省 id，避免非首 chunk 的局部行号造成重复 id。
            default_id = f"{stable_source}@{byte_offset}"
            record = normalize_dataset_record(payload, default_id=default_id)
            if allowed_types is not None and record["type"] not in allowed_types:
                raise ValueError(
                    f"{source_path} 包含不被允许的样本类型: {record['type']}，"
                    f"期望类型为 {sorted(allowed_types)}"
                )
            metadata = RecordMetadata(
                record_id=str(record["id"]),
                source_path=stable_source,
                record_index=source_record_count,
                line_number=current_line_number,
                byte_offset=int(byte_offset),
                byte_length=len(raw_line),
                text_length=_record_text_length(record),
                record_type=str(record["type"]),
                source=record.get("source"),
            )
            records.append(ParsedJSONLRecord(record=dict(record), metadata=metadata))
            source_record_count += 1
    return records, source_record_count


def _read_jsonl_chunk_worker(
    chunk: JSONLChunk,
    expected_types: tuple[str, ...] | None,
    source_root: str | None,
) -> _ChunkResult:
    """spawn worker 入口：只负责读取和 schema 规范化。"""
    records, count = _parse_jsonl_chunk(
        chunk,
        expected_types=expected_types,
        source_root=None if source_root is None else Path(source_root),
    )
    return _ChunkResult(chunk=chunk, records=tuple(records), source_record_count=count)


def _select_resume_chunks(
    chunks: Sequence[JSONLChunk],
    resume_offsets: Mapping[str | os.PathLike[str], int] | None,
) -> tuple[JSONLChunk, ...]:
    """按每个文件的 byte offset 筛选可恢复的 chunk。"""
    if not resume_offsets:
        return tuple(chunks)
    normalised = {str(Path(key).resolve()): max(0, int(value)) for key, value in resume_offsets.items()}
    selected = []
    for chunk in chunks:
        offset = normalised.get(str(Path(chunk.source_path).resolve()), 0)
        if int(chunk.end_offset) <= offset:
            continue
        if int(chunk.start_offset) < offset < int(chunk.end_offset):
            selected.append(replace(chunk, start_offset=offset))
        else:
            selected.append(chunk)
    return tuple(selected)


def _iter_ordered_chunk_results(
    chunks: Sequence[JSONLChunk],
    *,
    worker_count: int,
    max_in_flight: int | None,
    submit,
) -> Iterator[_ChunkResult]:
    """有界、保序地执行 chunk worker；``submit`` 只在主进程调用。"""
    chunks = tuple(chunks)
    if worker_count <= 0:
        for chunk in chunks:
            yield submit(None, chunk)
        return

    worker_count = min(int(worker_count), max(1, int(os.cpu_count() or 1)))
    in_flight = max(worker_count, int(max_in_flight or worker_count * 2))
    context = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=worker_count, mp_context=context)
    pending: deque[Any] = deque()
    chunk_iterator = iter(chunks)

    def submit_next() -> bool:
        try:
            chunk = next(chunk_iterator)
        except StopIteration:
            return False
        pending.append(submit(executor, chunk))
        return True

    try:
        for _ in range(in_flight):
            if not submit_next():
                break
        while pending:
            future = pending.popleft()
            result = future.result()
            submit_next()
            yield result
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def iter_jsonl_records_parallel(
    paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    *,
    expected_types: Iterable[str] | None = None,
    num_workers: int = 0,
    chunk_bytes: int = DEFAULT_JSONL_CHUNK_BYTES,
    max_in_flight: int | None = None,
    resume_offsets: Mapping[str | os.PathLike[str], int] | None = None,
    record_index_offsets: Mapping[str | os.PathLike[str], int] | None = None,
) -> Iterator[ParsedJSONLRecord]:
    """并行读取 JSONL，并按输入文件和字节 chunk 顺序产出记录。

    ``resume_offsets`` 的值应指向下一条记录的字节起点；若落在一行中间，
    该行会被跳过。``record_index_offsets`` 用于恢复时修正文件内记录序号，
    不传时从零开始计数。``num_workers=0`` 是同步实现，便于 CPU smoke test。
    """
    normalised_paths = _normalise_paths(paths)
    chunks = _select_resume_chunks(plan_jsonl_chunks(normalised_paths, chunk_bytes=chunk_bytes), resume_offsets)
    expected = None if expected_types is None else tuple(sorted(set(expected_types)))
    source_root = Path.cwd().resolve()

    # 同步路径不能解析两遍；单独实现一次调用以避免大文件 CPU 浪费。
    if int(num_workers) <= 0:
        counts: dict[str, int] = {}
        offsets = _normalise_record_offsets(record_index_offsets)
        for chunk in chunks:
            parsed, source_count = _parse_jsonl_chunk(
                chunk,
                expected_types=expected,
                source_root=source_root,
            )
            key = str(Path(chunk.source_path).resolve())
            base = counts.get(key, offsets.get(key, 0))
            for item in parsed:
                yield replace(item, metadata=replace(item.metadata, record_index=base + item.metadata.record_index))
            counts[key] = base + source_count
        return

    counts = {}
    offsets = _normalise_record_offsets(record_index_offsets)
    for result in _iter_ordered_chunk_results(
        chunks,
        worker_count=int(num_workers),
        max_in_flight=max_in_flight,
        submit=lambda executor, chunk: executor.submit(
            _read_jsonl_chunk_worker,
            chunk,
            expected,
            str(source_root),
        ),
    ):
        key = str(Path(result.chunk.source_path).resolve())
        base = counts.get(key, offsets.get(key, 0))
        for item in result.records:
            yield replace(item, metadata=replace(item.metadata, record_index=base + item.metadata.record_index))
        counts[key] = base + result.source_record_count


class JSONLReaderPool:
    """可跨 epoch 复用的 JSONL ``spawn`` reader 进程池。

    ``iter_jsonl_records_parallel`` 适合一次性扫描；训练循环若每个 epoch 都
    重建 iterator，应使用此类把进程启动和 tokenizer/schema 模块导入成本摊平。
    该类只保存路径、chunk 计划和 executor，实例本身可 pickle（pickle 后的
    副本会在目标进程中重新创建 pool）。
    """

    def __init__(
        self,
        paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
        *,
        expected_types: Iterable[str] | None = None,
        num_workers: int = 0,
        chunk_bytes: int = DEFAULT_JSONL_CHUNK_BYTES,
        max_in_flight: int | None = None,
        source_root: str | os.PathLike[str] | None = None,
    ):
        self.paths = _normalise_paths(paths)
        self.expected_types = (
            None if expected_types is None else tuple(sorted(set(expected_types)))
        )
        self.num_workers = max(0, int(num_workers))
        self.chunk_bytes = int(chunk_bytes)
        self.max_in_flight = max_in_flight
        self.source_root = (
            Path.cwd().resolve()
            if source_root is None
            else Path(source_root).expanduser().resolve()
        )
        self.chunks = plan_jsonl_chunks(self.paths, chunk_bytes=self.chunk_bytes)
        self._executor = None
        self._closed = False

    def __getstate__(self):
        """executor 不参与 pickle；目标进程会按配置懒创建。"""
        return {
            "paths": self.paths,
            "expected_types": self.expected_types,
            "num_workers": self.num_workers,
            "chunk_bytes": self.chunk_bytes,
            "max_in_flight": self.max_in_flight,
            "source_root": self.source_root,
        }

    def __setstate__(self, state):
        self.__init__(
            state["paths"],
            expected_types=state["expected_types"],
            num_workers=state["num_workers"],
            chunk_bytes=state["chunk_bytes"],
            max_in_flight=state["max_in_flight"],
            source_root=state["source_root"],
        )

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("JSONLReaderPool 已关闭。")
        if self.num_workers <= 0 or self._executor is not None:
            return
        context = multiprocessing.get_context("spawn")
        self._executor = ProcessPoolExecutor(
            max_workers=min(self.num_workers, max(1, int(os.cpu_count() or 1))),
            mp_context=context,
        )

    def iter_records(
        self,
        *,
        paths: Iterable[str | os.PathLike[str]] | None = None,
        resume_offsets: Mapping[str | os.PathLike[str], int] | None = None,
        record_index_offsets: Mapping[str | os.PathLike[str], int] | None = None,
    ) -> Iterator[ParsedJSONLRecord]:
        """迭代一轮数据；返回后 reader pool 仍保持运行。

        ``paths`` 用于在同一个长期存活的进程池中读取 manifest 的单个 entry。
        不传时按构造函数的全部路径读取；传入的路径必须属于该 pool，避免
        训练过程中悄悄扩大 reader 的文件范围。
        """
        self._ensure_open()
        if paths is None:
            chunks = self.chunks
        else:
            selected_paths = _normalise_paths(paths)
            known_paths = {str(Path(path).resolve()) for path in self.paths}
            unknown_paths = [
                path for path in selected_paths
                if str(Path(path).resolve()) not in known_paths
            ]
            if unknown_paths:
                raise ValueError(
                    "JSONLReaderPool 收到未规划的路径: "
                    + ", ".join(str(path) for path in unknown_paths)
                )
            chunks_by_path = {
                str(Path(path).resolve()): [] for path in selected_paths
            }
            for chunk in self.chunks:
                key = str(Path(chunk.source_path).resolve())
                if key in chunks_by_path:
                    chunks_by_path[key].append(chunk)
            chunks = tuple(
                chunk
                for path in selected_paths
                for chunk in chunks_by_path[str(Path(path).resolve())]
            )
        chunks = _select_resume_chunks(chunks, resume_offsets)
        offsets = _normalise_record_offsets(record_index_offsets)
        counts: dict[str, int] = {}
        if self.num_workers <= 0:
            for chunk in chunks:
                parsed, source_count = _parse_jsonl_chunk(
                    chunk,
                    expected_types=self.expected_types,
                    source_root=self.source_root,
                )
                key = str(Path(chunk.source_path).resolve())
                base = counts.get(key, offsets.get(key, 0))
                for item in parsed:
                    yield replace(
                        item,
                        metadata=replace(item.metadata, record_index=base + item.metadata.record_index),
                    )
                counts[key] = base + source_count
            return

        executor = self._executor
        if executor is None:  # 防御性检查，_ensure_open 后不应发生
            raise RuntimeError("JSONLReaderPool executor 尚未初始化。")
        in_flight = max(
            self.num_workers,
            int(self.max_in_flight or self.num_workers * 2),
        )
        pending: deque[Any] = deque()
        chunk_iterator = iter(chunks)

        def submit_next() -> bool:
            try:
                chunk = next(chunk_iterator)
            except StopIteration:
                return False
            pending.append(
                executor.submit(
                    _read_jsonl_chunk_worker,
                    chunk,
                    self.expected_types,
                    str(self.source_root),
                )
            )
            return True

        try:
            for _ in range(in_flight):
                if not submit_next():
                    break
            while pending:
                future = pending.popleft()
                result = future.result()
                submit_next()
                key = str(Path(result.chunk.source_path).resolve())
                base = counts.get(key, offsets.get(key, 0))
                for item in result.records:
                    yield replace(
                        item,
                        metadata=replace(item.metadata, record_index=base + item.metadata.record_index),
                    )
                counts[key] = base + result.source_record_count
        finally:
            # 调用方可以提前 break；已提交但未消费的任务不能继续无限生产结果。
            for future in pending:
                future.cancel()

    def close(self):
        """关闭并等待 worker；重复调用是安全的。"""
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()


# worker 进程只读这两个全局对象；它们由 initializer 设置，避免每条记录重复 pickle。
_CACHE_WORKER_TOKENIZER = None
_CACHE_WORKER_ENCODER = None


def _initialize_token_cache_worker(tokenizer, encode_fn):
    """初始化 token cache worker，并关闭 tokenizer 自身的线程池。"""
    global _CACHE_WORKER_TOKENIZER, _CACHE_WORKER_ENCODER
    _CACHE_WORKER_TOKENIZER = tokenizer
    _CACHE_WORKER_ENCODER = encode_fn
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _extract_token_ids(encoded: Any) -> tuple[int, ...]:
    """兼容 tokenizer 返回 dict、numpy、torch 或普通序列。"""
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if encoded is None:
        raise ValueError("编码器没有返回 input_ids。")
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, Sequence) and encoded and isinstance(encoded[0], Sequence):
        encoded = encoded[0]
    result = []
    for token in encoded:
        if isinstance(token, bool) or not isinstance(token, int):
            try:
                token = int(token)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"token id 不是整数: {token!r}") from exc
        if token < 0 or token > 0xFFFFFFFF:
            raise ValueError(f"token id 超出 uint32 范围: {token}")
        result.append(int(token))
    return tuple(result)


def _encode_record(record: Mapping[str, Any]) -> tuple[int, ...]:
    """在 worker 中调用用户编码器或最小 text tokenizer 适配。"""
    if _CACHE_WORKER_ENCODER is not None:
        return _extract_token_ids(_CACHE_WORKER_ENCODER(dict(record)))
    tokenizer = _CACHE_WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("token cache worker 没有 tokenizer 或 encode_fn。")
    if record.get("type") != "text":
        raise ValueError("chat 样本必须通过 encode_fn 编码，以复用项目 chat 模板。")
    token_ids = _extract_token_ids(
        tokenizer(record["text"], add_special_tokens=False)
    )
    # text 模板在正文后追加 eos 片段，且正文与 eos 都是全监督 answer 通道。
    # cache 只保存 token ids，因此这里补上 eos，保证后续训练可直接复用，
    # 不需要再经过 tokenizer 或模板渲染。
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        token_ids = token_ids + (int(eos_token_id),)
    return token_ids


def _tokenize_jsonl_chunk_worker(
    chunk: JSONLChunk,
    expected_types: tuple[str, ...] | None,
    source_root: str | None,
    max_sequence_length: int | None,
    min_tokens: int,
    drop_short: bool,
) -> _ChunkResult:
    """spawn worker 入口：读取、规范化和分词在同一进程完成。"""
    parsed, source_count = _parse_jsonl_chunk(
        chunk,
        expected_types=expected_types,
        source_root=None if source_root is None else Path(source_root),
    )
    result: list[TokenizedCacheRecord] = []
    for item in parsed:
        token_ids = _encode_record(item.record)
        if max_sequence_length is not None:
            token_ids = token_ids[:max_sequence_length]
        if len(token_ids) < min_tokens:
            if drop_short:
                continue
            raise ValueError(
                f"样本 {item.metadata.record_id} 编码后只有 {len(token_ids)} 个 token，"
                f"小于 min_tokens={min_tokens}。"
            )
        result.append(
            TokenizedCacheRecord(
                token_ids=token_ids,
                metadata=replace(item.metadata, token_length=len(token_ids)),
            )
        )
    return _ChunkResult(chunk=chunk, records=tuple(result), source_record_count=source_count)


def _normalise_record_offsets(offsets: Mapping[str | os.PathLike[str], int] | None) -> dict[str, int]:
    """规范化恢复用的 source path -> record index 映射。"""
    if not offsets:
        return {}
    return {str(Path(key).resolve()): max(0, int(value)) for key, value in offsets.items()}


def _cleanup_stale_cache_tmp(destination: Path) -> None:
    """清理同一目标目录的历史中断残留，保证一个数据集只有一个 cache。

    ``build_token_cache`` 先在目标同级创建 ``<name>_tmp-*`` 临时目录，成功
    后才原子改名为最终目录。进程被强制终止时，异常清理可能来不及执行，
    残留的临时目录会在下次构建前被这里按目标名前缀清除，避免磁盘堆积。
    """
    destination = Path(destination)
    parent = destination.parent
    if not parent.is_dir():
        return
    for item in parent.glob(f"{destination.name}_tmp-*"):
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            elif item.exists():
                item.unlink()
        except OSError:
            pass


def _file_fingerprint(path: Path, sample_bytes: int = 64 * 1024) -> dict[str, Any]:
    """生成不需要完整扫描大文件的输入指纹。"""
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if stat.st_size > sample_bytes:
            handle.seek(max(0, stat.st_size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return {
        "path": _stable_path(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sample_sha256": digest.hexdigest(),
        "algorithm": "stat+head_tail_sha256",
    }


def tokenizer_fingerprint(tokenizer: Any, encode_fn: Callable | None) -> dict[str, Any]:
    """生成可审计的 tokenizer/encoder 指纹，不把对象写进 manifest。"""
    if encode_fn is not None:
        identity = {
            "kind": "encode_fn",
            "module": getattr(encode_fn, "__module__", None),
            "qualname": getattr(encode_fn, "__qualname__", repr(encode_fn)),
        }
    else:
        identity = {
            "kind": "tokenizer",
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "name_or_path": getattr(tokenizer, "name_or_path", None),
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    identity["sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return identity


_tokenizer_fingerprint = tokenizer_fingerprint


def _write_cache_shard(
    shard_dir: Path,
    shard_name: str,
    records: Sequence[TokenizedCacheRecord],
) -> _CacheShardInfo:
    """写入一个 ragged token 分片。"""
    _require_numpy()
    target = shard_dir / shard_name
    target.mkdir(parents=True, exist_ok=False)
    flat: list[int] = []
    offsets = [0]
    lengths = []
    for item in records:
        flat.extend(item.token_ids)
        offsets.append(len(flat))
        lengths.append(len(item.token_ids))

    # Windows 上 np.save 可能延迟关闭文件句柄；显式使用 with 确保立即关闭。
    with open(target / "tokens.npy", "wb") as f:
        np.save(f, np.asarray(flat, dtype=np.uint32), allow_pickle=False)
    with open(target / "offsets.npy", "wb") as f:
        np.save(f, np.asarray(offsets, dtype=np.int64), allow_pickle=False)
    with open(target / "lengths.npy", "wb") as f:
        np.save(f, np.asarray(lengths, dtype=np.int32), allow_pickle=False)

    with (target / "records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in records:
            payload = asdict(item.metadata)
            payload["record_id"] = item.metadata.record_id
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    lengths_array = np.asarray(lengths, dtype=np.int32)
    return _CacheShardInfo(
        name=shard_name,
        records=len(records),
        tokens=len(flat),
        min_length=int(lengths_array.min()) if lengths else 0,
        max_length=int(lengths_array.max()) if lengths else 0,
    )


def build_token_cache(
    paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    cache_dir: str | os.PathLike[str],
    *,
    tokenizer: Any = None,
    encode_fn: Callable[[Mapping[str, Any]], Any] | None = None,
    expected_types: Iterable[str] | None = None,
    max_sequence_length: int | None = None,
    min_tokens: int = 2,
    drop_short: bool = True,
    shard_size_records: int = DEFAULT_CACHE_SHARD_RECORDS,
    shard_size_tokens: int = DEFAULT_CACHE_SHARD_TOKENS,
    num_workers: int = 0,
    chunk_bytes: int = DEFAULT_JSONL_CHUNK_BYTES,
    max_in_flight: int | None = None,
    source_root: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """并行构建离线 token cache，并原子发布到 ``cache_dir``。

    ``encode_fn`` 必须是模块顶层、可 pickle 的函数，签名为 ``record -> token ids``；
    仅处理 text 数据时也可以传 HuggingFace tokenizer。默认不改写已有目录，需
    显式 ``overwrite=True`` 才会替换旧 cache。
    """
    _require_numpy()
    if (tokenizer is None) == (encode_fn is None):
        raise ValueError("tokenizer 与 encode_fn 必须且只能提供一个。")
    if int(min_tokens) < 0:
        raise ValueError("min_tokens 不能为负数。")
    if max_sequence_length is not None and int(max_sequence_length) <= 0:
        raise ValueError("max_sequence_length 必须为正整数。")
    if int(shard_size_records) <= 0 or int(shard_size_tokens) <= 0:
        raise ValueError("shard_size_records 和 shard_size_tokens 必须为正整数。")
    if int(num_workers) < 0:
        raise ValueError("num_workers 不能为负数。")
    normalised_paths = _normalise_paths(paths)
    expected = None if expected_types is None else tuple(sorted(set(expected_types)))
    root = Path.cwd().resolve() if source_root is None else Path(source_root).resolve()
    destination = Path(cache_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"cache 目录已存在：{destination}；如需替换请显式 overwrite=True。")

    # spawn 会 pickle initializer 参数；提前给出比 BrokenProcessPool 更易懂的错误。
    if int(num_workers) > 0:
        try:
            pickle.dumps(tokenizer if encode_fn is None else encode_fn)
        except Exception as exc:
            raise TypeError("tokenizer/encode_fn 必须支持 pickle，才能启用 spawn worker。") from exc

    chunks = plan_jsonl_chunks(normalised_paths, chunk_bytes=chunk_bytes)
    temp_parent = destination.parent
    # 强杀残留的旧临时目录不参与校验，也不能继续累积；构建前先清理。
    _cleanup_stale_cache_tmp(destination)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{destination.name}_tmp-", dir=str(temp_parent)))
    shard_root = temp_dir / "shards"
    shard_root.mkdir()
    shard_infos: list[_CacheShardInfo] = []
    pending_records: list[TokenizedCacheRecord] = []
    pending_tokens = 0
    source_counts: dict[str, int] = {}
    source_ranges: dict[str, dict[str, int]] = {}
    seen_source_keys: set[str] = set()
    file_type_counters: dict[str, Counter[str]] = {}
    file_source_counters: dict[str, Counter[str]] = {}
    total_type_counter: Counter[str] = Counter()
    total_source_counter: Counter[str] = Counter()
    total_records = 0
    total_tokens = 0

    try:
        if int(num_workers) <= 0:
            def submit_sync(_executor, chunk):
                return _tokenize_jsonl_chunk_worker_sync(
                    chunk,
                    expected,
                    str(root),
                    tokenizer,
                    encode_fn,
                    max_sequence_length,
                    int(min_tokens),
                    bool(drop_short),
                )
            results = _iter_ordered_chunk_results(
                chunks,
                worker_count=0,
                max_in_flight=None,
                submit=submit_sync,
            )
        else:
            context = multiprocessing.get_context("spawn")
            # 此分支不用 _iter_ordered_chunk_results 的 initializer 参数，因为 cache
            # worker 需要一次性缓存 tokenizer/encoder。
            executor = ProcessPoolExecutor(
                max_workers=min(int(num_workers), max(1, int(os.cpu_count() or 1))),
                mp_context=context,
                initializer=_initialize_token_cache_worker,
                initargs=(tokenizer, encode_fn),
            )
            results = _iter_ordered_cache_worker_results(
                executor,
                chunks,
                expected,
                str(root),
                max_sequence_length,
                int(min_tokens),
                bool(drop_short),
                max_in_flight=max_in_flight,
            )

        for result in results:
            key = str(Path(result.chunk.source_path).resolve())
            if key not in seen_source_keys:
                seen_source_keys.add(key)
                source_ranges[key] = {"start": sum(source_counts.values())}
            base = source_counts.get(key, 0)
            for item in result.records:
                metadata = replace(
                    item.metadata,
                    record_index=base + item.metadata.record_index,
                )
                file_type_counters.setdefault(key, Counter())[
                    str(metadata.record_type or "<unknown>")
                ] += 1
                file_source_counters.setdefault(key, Counter())[
                    str(metadata.source or "<unknown>")
                ] += 1
                total_type_counter[str(metadata.record_type or "<unknown>")] += 1
                total_source_counter[str(metadata.source or "<unknown>")] += 1
                normalized_item = replace(item, metadata=metadata)
                pending_records.append(normalized_item)
                pending_tokens += len(normalized_item.token_ids)
                total_records += 1
                total_tokens += len(normalized_item.token_ids)
                if (
                    len(pending_records) >= int(shard_size_records)
                    or pending_tokens >= int(shard_size_tokens)
                ):
                    shard_infos.append(
                        _write_cache_shard(
                            shard_root,
                            f"shard-{len(shard_infos):05d}",
                            pending_records,
                        )
                    )
                    pending_records = []
                    pending_tokens = 0
            source_counts[key] = base + result.source_record_count
            source_ranges[key]["count"] = source_counts[key]

        if pending_records:
            shard_infos.append(
                _write_cache_shard(
                    shard_root,
                    f"shard-{len(shard_infos):05d}",
                    pending_records,
                )
            )
        if not shard_infos:
            raise ValueError("没有可写入 token cache 的记录。")

        extra = dict(extra_metadata or {})
        extra["type_counts"] = dict(sorted(total_type_counter.items()))
        extra["source_counts"] = dict(sorted(total_source_counter.items()))
        extra["source_summaries"] = {
            _stable_path(Path(key), root): {
                "type_counts": dict(sorted(file_type_counters.get(key, Counter()).items())),
                "source_counts": dict(sorted(file_source_counters.get(key, Counter()).items())),
            }
            for key in sorted(source_counts)
        }
        extra["source_record_ranges"] = {
            _stable_path(Path(key), root): {
                "start": int(value["start"]),
                "count": int(value["count"]),
            }
            for key, value in sorted(source_ranges.items())
        }

        manifest = {
            "format": TOKEN_CACHE_FORMAT,
            "schema_version": TOKEN_CACHE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "token_dtype": "uint32",
            "max_sequence_length": None if max_sequence_length is None else int(max_sequence_length),
            "min_tokens": int(min_tokens),
            "drop_short": bool(drop_short),
            "total_records": int(total_records),
            "total_tokens": int(total_tokens),
            "source_files": [_file_fingerprint(path) for path in normalised_paths],
            "tokenizer": _tokenizer_fingerprint(tokenizer, encode_fn),
            "shards": [asdict(info) for info in shard_infos],
            "extra_metadata": extra,
        }
        with (temp_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        # Windows 上 executor.shutdown 后进程可能还持有文件句柄；
        # 显式等待并触发垃圾回收，确保所有句柄释放。
        if int(num_workers) > 0:
            time.sleep(1.0)
            gc.collect()
            time.sleep(1.0)

        if destination.exists():
            if not overwrite:
                raise FileExistsError(destination)
            # 先把旧目录原子改名为备份，再发布新目录；避免直接 rmtree 时
            # 目标被占用（杀毒/同步/未释放句柄）导致半删除或改名失败。
            backup = destination.with_name(f"{destination.name}_legacy")
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            destination.rename(backup)
        else:
            backup = None
        try:
            # Windows 上文件占用可能导致 replace 失败；添加重试逻辑。
            max_retries = 10
            last_error = None
            for attempt in range(max_retries):
                try:
                    temp_dir.replace(destination)
                    break
                except (OSError, PermissionError) as exc:
                    last_error = exc
                    if attempt < max_retries - 1:
                        wait_time = 1.0 + (attempt * 0.5)
                        time.sleep(wait_time)
                        gc.collect()
                    else:
                        raise last_error
        except Exception:
            # 发布失败时尽力恢复旧目录，保证下一次仍可复用/重建。
            if backup is not None and backup.exists() and not destination.exists():
                try:
                    backup.rename(destination)
                except OSError:
                    pass
            raise
        if backup is not None and backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return destination


def _tokenize_jsonl_chunk_worker_sync(
    chunk,
    expected_types,
    source_root,
    tokenizer,
    encode_fn,
    max_sequence_length,
    min_tokens,
    drop_short,
):
    """同步 cache 构建路径，不依赖 worker 全局状态。"""
    global _CACHE_WORKER_TOKENIZER, _CACHE_WORKER_ENCODER
    old_tokenizer, old_encoder = _CACHE_WORKER_TOKENIZER, _CACHE_WORKER_ENCODER
    _CACHE_WORKER_TOKENIZER, _CACHE_WORKER_ENCODER = tokenizer, encode_fn
    try:
        return _tokenize_jsonl_chunk_worker(
            chunk,
            expected_types,
            source_root,
            max_sequence_length,
            min_tokens,
            drop_short,
        )
    finally:
        _CACHE_WORKER_TOKENIZER, _CACHE_WORKER_ENCODER = old_tokenizer, old_encoder


def _iter_ordered_cache_worker_results(
    executor,
    chunks,
    expected,
    source_root,
    max_sequence_length,
    min_tokens,
    drop_short,
    *,
    max_in_flight,
):
    """cache worker 的有界保序结果生成器。"""
    worker_count = int(getattr(executor, "_max_workers", 1))
    in_flight = max(worker_count, int(max_in_flight or worker_count * 2))
    pending: deque[Any] = deque()
    iterator = iter(chunks)

    def submit_next():
        try:
            chunk = next(iterator)
        except StopIteration:
            return False
        pending.append(
            executor.submit(
                _tokenize_jsonl_chunk_worker,
                chunk,
                expected,
                source_root,
                max_sequence_length,
                min_tokens,
                drop_short,
            )
        )
        return True

    try:
        for _ in range(in_flight):
            if not submit_next():
                break
        while pending:
            future = pending.popleft()
            result = future.result()
            submit_next()
            yield result
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


class TokenCache:
    """只读 ragged token cache；token 数组通过 mmap 延迟加载。"""

    def __init__(self, cache_dir: str | os.PathLike[str]):
        _require_numpy()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"不是有效 token cache：{manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("format") != TOKEN_CACHE_FORMAT:
            raise ValueError(f"不支持的 token cache 格式：{self.manifest.get('format')!r}")
        if int(self.manifest.get("schema_version", -1)) != TOKEN_CACHE_SCHEMA_VERSION:
            raise ValueError("token cache schema 版本不匹配。")
        self._shards = tuple(self.manifest.get("shards", ()))
        self._cumulative = []
        total = 0
        for shard in self._shards:
            records = int(shard["records"])
            if records <= 0:
                raise ValueError("token cache 分片 records 必须为正数。")
            total += records
            self._cumulative.append(total)
        if total != int(self.manifest.get("total_records", -1)):
            raise ValueError("token cache manifest 的记录数与分片不一致。")
        self._loaded_arrays: dict[int, tuple[Any, Any, Any]] = {}

    def __len__(self):
        return int(self.manifest["total_records"])

    def __getstate__(self):
        """只序列化路径，spawn 子进程中重新打开 mmap。"""
        return {"cache_dir": str(self.cache_dir)}

    def __setstate__(self, state):
        self.__init__(state["cache_dir"])

    def _locate(self, index: int) -> tuple[int, int]:
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self._cumulative, index)
        previous = 0 if shard_index == 0 else self._cumulative[shard_index - 1]
        return shard_index, index - previous

    def _arrays(self, shard_index: int):
        arrays = self._loaded_arrays.get(shard_index)
        if arrays is None:
            shard_path = self.cache_dir / "shards" / self._shards[shard_index]["name"]
            arrays = (
                np.load(shard_path / "tokens.npy", mmap_mode="r", allow_pickle=False),
                np.load(shard_path / "offsets.npy", mmap_mode="r", allow_pickle=False),
                np.load(shard_path / "lengths.npy", mmap_mode="r", allow_pickle=False),
            )
            self._loaded_arrays[shard_index] = arrays
        return arrays

    def get_tokens(self, index: int, *, copy: bool = False):
        """返回单条 token 序列；默认是 mmap view。"""
        shard_index, local_index = self._locate(index)
        tokens, offsets, _lengths = self._arrays(shard_index)
        value = tokens[int(offsets[local_index]) : int(offsets[local_index + 1])]
        return value.copy() if copy else value

    def _read_metadata(self, shard_index: int, local_index: int) -> dict[str, Any]:
        shard_path = self.cache_dir / "shards" / self._shards[shard_index]["name"]
        with (shard_path / "records.jsonl").open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index == local_index:
                    return json.loads(line)
        raise IndexError(local_index)

    def __getitem__(self, index: int) -> TokenizedCacheRecord:
        shard_index, local_index = self._locate(index)
        return TokenizedCacheRecord(
            token_ids=tuple(int(value) for value in self.get_tokens(index)),
            metadata=RecordMetadata(**self._read_metadata(shard_index, local_index)),
        )

    def iter_records(
        self,
        start: int = 0,
        stop: int | None = None,
        *,
        as_numpy: bool = False,
    ) -> Iterator[TokenizedCacheRecord]:
        """顺序遍历 token 记录，适合训练数据 worker。

        ``as_numpy=False`` 保持历史 tuple 语义（``TokenizedCacheRecord.token_ids``
        是 ``tuple[int, ...]``），供序列化或外部只读场景使用；
        ``as_numpy=True`` 时直接返回 ``numpy`` 数组切片（默认是 mmap view，
        不复制数据），让 cache 数据可以在批处理路径上零 Python 循环直达张量。
        调用方需要持有数组引用时自行 ``copy()``，避免 mmap 关闭后访问失效。
        """
        start = max(0, int(start))
        stop = len(self) if stop is None else min(len(self), int(stop))
        if stop <= start:
            return
        global_index = 0
        for shard_index, shard in enumerate(self._shards):
            shard_start = global_index
            shard_end = global_index + int(shard["records"])
            global_index = shard_end
            if shard_end <= start or shard_start >= stop:
                continue
            tokens, offsets, _lengths = self._arrays(shard_index)
            metadata_path = self.cache_dir / "shards" / shard["name"] / "records.jsonl"
            with metadata_path.open("r", encoding="utf-8") as handle:
                for local_index, line in enumerate(handle):
                    absolute_index = shard_start + local_index
                    if absolute_index < start:
                        continue
                    if absolute_index >= stop:
                        break
                    metadata = RecordMetadata(**json.loads(line))
                    begin = int(offsets[local_index])
                    end = int(offsets[local_index + 1])
                    token_ids = (
                        tokens[begin:end]
                        if as_numpy
                        else tuple(int(value) for value in tokens[begin:end])
                    )
                    yield TokenizedCacheRecord(
                        token_ids=token_ids,
                        metadata=metadata,
                    )

    def iter_lengths(self) -> Iterator[int]:
        """流式返回所有 token 长度，不把长度数组拼接到内存。"""
        for shard_index in range(len(self._shards)):
            _tokens, _offsets, lengths = self._arrays(shard_index)
            for value in lengths:
                yield int(value)

    def _extra_metadata(self) -> Mapping[str, Any]:
        """返回 manifest 中的扩展元数据，缺失时返回空字典。"""
        extra = self.manifest.get("extra_metadata")
        return extra if isinstance(extra, Mapping) else {}

    def type_counts(self) -> dict[str, int]:
        """返回构建时统计的样本类型分布。"""
        return dict(self._extra_metadata().get("type_counts") or {})

    def source_counts(self) -> dict[str, int]:
        """返回构建时统计的来源分布。"""
        return dict(self._extra_metadata().get("source_counts") or {})

    def source_record_ranges(self) -> dict[str, dict[str, int]]:
        """返回 stable source path -> (start, count) 的记录区间。

        ``start`` 是 cache 全局记录序号，``count`` 是该源文件实际缓存的记录数。
        训练数据集据此把 manifest entry 定位到 cache 片段，无需再扫描 JSONL。
        """
        ranges = self._extra_metadata().get("source_record_ranges")
        if not isinstance(ranges, Mapping):
            return {}
        result = {}
        for key, value in ranges.items():
            if not isinstance(value, Mapping):
                continue
            try:
                result[str(key)] = {
                    "start": int(value["start"]),
                    "count": int(value["count"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
        return result

    def source_summaries(self) -> dict[str, dict[str, dict[str, int]]]:
        """返回 stable source path -> 类型/来源统计的映射。"""
        summaries = self._extra_metadata().get("source_summaries")
        if not isinstance(summaries, Mapping):
            return {}
        result = {}
        for key, value in summaries.items():
            if not isinstance(value, Mapping):
                continue
            result[str(key)] = {
                "type_counts": dict(value.get("type_counts") or {}),
                "source_counts": dict(value.get("source_counts") or {}),
            }
        return result

    def fingerprint_matches(self, paths) -> bool:
        """校验 cache 对应的源文件与当前磁盘文件一致。

        比较的是 stat + 头尾采样 sha256，不重新扫描大文件，因此启动阶段
        可以在毫秒级判断 cache 是否需要重建。
        """
        try:
            current = [_file_fingerprint(path) for path in _normalise_paths(paths)]
        except (OSError, ValueError):
            return False
        cached = self.manifest.get("source_files")
        if not isinstance(cached, list) or len(cached) != len(current):
            return False
        return all(cached_item == current_item for cached_item, current_item in zip(cached, current))

    def iter_bucketed_batches(
        self,
        batch_size: int,
        *,
        boundaries: Sequence[int] = (1024, 2048, 3072, 4096),
        seed: int = 0,
        epoch: int = 0,
        drop_last: bool = False,
    ) -> Iterator[tuple[int, ...]]:
        """按 token 长度分桶，返回稳定但每 epoch 可变的 index batch。

        每个桶内先按长度排序再切 batch，随后只打乱 batch 的顺序；这样 batch
        内 padding 上限稳定，同时仍保留 epoch 级随机性。最后一个边界之外的
        样本进入溢出桶，调用方可据此发现超过模型窗口的数据。
        """
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数。")
        limits = tuple(int(value) for value in boundaries)
        if any(value <= 0 for value in limits) or tuple(sorted(set(limits))) != limits:
            raise ValueError("boundaries 必须是严格递增的正整数序列。")
        buckets: list[list[tuple[int, int]]] = [[] for _ in range(len(limits) + 1)]
        for index, length in enumerate(self.iter_lengths()):
            buckets[bisect_right(limits, length)].append((length, index))
        rng = random.Random(int(seed) + int(epoch))
        batches: list[tuple[int, ...]] = []
        for bucket in buckets:
            bucket.sort(key=lambda item: (item[0], item[1]))
            bucket_batches = [
                tuple(index for _length, index in bucket[start : start + batch_size])
                for start in range(0, len(bucket), batch_size)
            ]
            if drop_last and bucket_batches and len(bucket_batches[-1]) < batch_size:
                bucket_batches.pop()
            rng.shuffle(bucket_batches)
            batches.extend(bucket_batches)
        rng.shuffle(batches)
        yield from batches

    def close(self):
        """释放已打开的 mmap；对象被垃圾回收时操作系统也会自动释放。"""
        for arrays in self._loaded_arrays.values():
            for array in arrays:
                mmap_obj = getattr(array, "_mmap", None)
                if mmap_obj is not None:
                    mmap_obj.close()
        self._loaded_arrays.clear()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()


__all__ = [
    "DEFAULT_CACHE_SHARD_RECORDS",
    "DEFAULT_CACHE_SHARD_TOKENS",
    "DEFAULT_JSONL_CHUNK_BYTES",
    "JSONLChunk",
    "JSONLReaderPool",
    "ParsedJSONLRecord",
    "RecordMetadata",
    "TOKEN_CACHE_FORMAT",
    "TOKEN_CACHE_SCHEMA_VERSION",
    "TokenCache",
    "TokenizedCacheRecord",
    "build_token_cache",
    "iter_jsonl_records_parallel",
    "plan_jsonl_chunks",
]
