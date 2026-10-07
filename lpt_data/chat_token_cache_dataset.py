"""SFT 与 LoRA 共用的 chat 预编码缓存及 manifest 数据集。

每个源 JSONL 文件独立缓存。缓存保存完整的监督序列，训练时仍按当前
manifest 动态抽样、打乱和 packing；修改抽样权重不会重新分词。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from lpt_config import GlobalConfig
from lpt_protocol import (
    THINKING_MODE_AUTO,
    THINKING_MODE_OFF,
    THINKING_MODE_ON,
    normalize_thinking_mode,
)

from .batching import (
    PRE_TOKENIZED_CHAT_INPUT_IDS,
    PRE_TOKENIZED_CHAT_LABELS,
    PRE_TOKENIZED_CHAT_TARGET_CHANNEL_IDS,
    PRE_TOKENIZED_CHAT_THINKING_MODE_IDS,
    encode_training_sample,
)
from .cache import (
    DEFAULT_JSONL_CHUNK_BYTES,
    RecordMetadata,
    TokenCache,
    _cleanup_stale_cache_tmp,
    _file_fingerprint,
    iter_jsonl_records_parallel,
    stable_path,
    tokenizer_fingerprint,
)
from .io import attach_progress_metadata
from .token_cache_dataset import (
    PRE_TOKENIZED_TOKEN_LENGTH,
    TokenCacheManifestDataset,
    cache_subdir_name,
    ensure_manifest_cache,
)


CHAT_TOKEN_CACHE_FORMAT = "lpt_chat_token_cache"
CHAT_TOKEN_CACHE_SCHEMA_VERSION = 1
CHAT_ENCODING_STRATEGY_VERSION = 1
PRE_TOKENIZED_CHAT_THINKING_SUMMARY = "_lpt_chat_thinking_summary"
_CHAT_ARRAY_FIELDS = (
    PRE_TOKENIZED_CHAT_INPUT_IDS,
    PRE_TOKENIZED_CHAT_LABELS,
    PRE_TOKENIZED_CHAT_THINKING_MODE_IDS,
    PRE_TOKENIZED_CHAT_TARGET_CHANNEL_IDS,
)
_SHARD_MAX_RECORDS = 10_000
_SHARD_MAX_TOKENS = 1_000_000


@dataclass(frozen=True)
class ChatTokenizedCacheRecord:
    """单条 chat 缓存记录，四个序列始终等长。"""

    input_ids: np.ndarray
    labels: np.ndarray
    thinking_mode_ids: np.ndarray
    target_channel_ids: np.ndarray
    metadata: RecordMetadata
    thinking_summary: dict[str, int]


def chat_cache_subdir_name(dataset_path) -> str:
    """与 text token cache 隔离，防止两个格式共用同一子目录。"""
    return f"{cache_subdir_name(dataset_path)}_chat"


def _chat_tokenizer_fingerprint(tokenizer) -> dict:
    """把词表内容纳入指纹，避免同名 tokenizer 改词表后误用旧标签。"""
    identity = tokenizer_fingerprint(tokenizer, None)
    vocab_reader = getattr(tokenizer, "get_vocab", None)
    if callable(vocab_reader):
        vocabulary = vocab_reader()
        payload = json.dumps(vocabulary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        identity["vocab_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and callable(getattr(backend, "to_str", None)):
        identity["backend_sha256"] = hashlib.sha256(
            backend.to_str().encode("utf-8")
        ).hexdigest()
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    identity["sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return identity


def _encoding_signature(*, thinking_mode, template_version, max_sequence_length, expected_types):
    """集中定义影响四个输出序列的全部缓存策略。"""
    return {
        "strategy_version": CHAT_ENCODING_STRATEGY_VERSION,
        "thinking_mode": str(thinking_mode),
        "template_version": str(template_version),
        "max_sequence_length": None if max_sequence_length is None else int(max_sequence_length),
        "expected_types": None if expected_types is None else sorted(set(expected_types)),
    }


def _thinking_summary(record, thinking_mode):
    """构建与训练审计相同的最小统计，不保存原始对话正文。"""
    summary = {
        "assistant_messages": 0,
        "assistant_messages_with_thinking_field": 0,
        "assistant_messages_with_nonempty_thinking": 0,
        "assistant_messages_with_empty_thinking": 0,
        "assistant_messages_thinking_on": 0,
        "assistant_messages_thinking_off": 0,
    }
    mode = normalize_thinking_mode(thinking_mode)
    for message in record.get("messages", ()):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        summary["assistant_messages"] += 1
        has_thinking = "thinking" in message
        nonempty = bool(str(message.get("thinking") or "").strip())
        if has_thinking:
            summary["assistant_messages_with_thinking_field"] += 1
            summary[
                "assistant_messages_with_nonempty_thinking"
                if nonempty else "assistant_messages_with_empty_thinking"
            ] += 1
        resolved = mode if mode in {THINKING_MODE_ON, THINKING_MODE_OFF} else (
            THINKING_MODE_ON if nonempty else THINKING_MODE_OFF
        )
        summary[
            "assistant_messages_thinking_on"
            if resolved == THINKING_MODE_ON else "assistant_messages_thinking_off"
        ] += 1
    return summary


def _write_shard(root, shard_index, records):
    """将 ragged 序列写成共享 offsets 的四个可 mmap 数组。"""
    shard_name = f"shard-{shard_index:05d}"
    destination = root / "shards" / shard_name
    destination.mkdir(parents=True)
    offsets = [0]
    arrays = [[], [], [], []]
    lengths = []
    for record in records:
        values = (record.input_ids, record.labels, record.thinking_mode_ids, record.target_channel_ids)
        length = len(values[0])
        if any(len(value) != length for value in values):
            raise ValueError(f"chat cache 样本 {record.metadata.record_id} 的监督序列长度不一致。")
        for target, value in zip(arrays, values):
            target.extend(value)
        offsets.append(offsets[-1] + length)
        lengths.append(length)
    # 磁盘上用 int32 容纳 token ID 与 -100 掩码，读取时统一转为训练所需 int64。
    dtypes = (np.uint32, np.int32, np.uint8, np.uint8)
    filenames = ("input_ids.npy", "labels.npy", "thinking_mode_ids.npy", "target_channel_ids.npy")
    for values, dtype, filename in zip(arrays, dtypes, filenames):
        with (destination / filename).open("wb") as handle:
            np.save(handle, np.asarray(values, dtype=dtype), allow_pickle=False)
    with (destination / "offsets.npy").open("wb") as handle:
        np.save(handle, np.asarray(offsets, dtype=np.int64), allow_pickle=False)
    with (destination / "lengths.npy").open("wb") as handle:
        np.save(handle, np.asarray(lengths, dtype=np.int32), allow_pickle=False)
    with (destination / "records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            payload = asdict(record.metadata)
            payload["thinking_summary"] = record.thinking_summary
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "name": shard_name,
        "records": len(records),
        "tokens": offsets[-1],
        "min_length": min(lengths),
        "max_length": max(lengths),
    }


def _build_chat_file_cache(
    path,
    destination,
    tokenizer,
    *,
    expected_types,
    thinking_mode,
    template_version,
    max_sequence_length,
    num_workers,
    chunk_bytes,
):
    """构建完整新目录，再替换旧目录；读取期间不会看到半成品。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 沿用 text cache 的中断恢复规则，避免旧临时目录持续占用磁盘。
    _cleanup_stale_cache_tmp(destination)
    temp = Path(tempfile.mkdtemp(prefix=f"{destination.name}_tmp-", dir=destination.parent))
    (temp / "shards").mkdir()
    pending = []
    pending_tokens = 0
    shards = []
    total_records = 0
    total_tokens = 0
    type_counts = Counter()
    source_counts = Counter()
    try:
        for parsed in iter_jsonl_records_parallel(
            [path],
            expected_types=expected_types,
            num_workers=num_workers,
            chunk_bytes=chunk_bytes,
        ):
            # 与在线训练共用唯一编码入口，保持模板、监督掩码和通道完全一致。
            encoded = encode_training_sample(
                parsed.record,
                tokenizer,
                max_length=max_sequence_length,
                thinking_mode=thinking_mode,
            )
            if any(int(value) < 0 or int(value) > np.iinfo(np.uint32).max for value in encoded.input_ids):
                raise ValueError(f"样本 {encoded.sample_id} 的 input_ids 超出 uint32 范围。")
            if any(int(value) < np.iinfo(np.int32).min or int(value) > np.iinfo(np.int32).max for value in encoded.labels):
                raise ValueError(f"样本 {encoded.sample_id} 的 labels 超出 int32 范围。")
            metadata = replace(parsed.metadata, token_length=encoded.length)
            pending.append(ChatTokenizedCacheRecord(
                input_ids=encoded.input_ids,
                labels=encoded.labels,
                thinking_mode_ids=encoded.thinking_mode_ids,
                target_channel_ids=encoded.target_channel_ids,
                metadata=metadata,
                thinking_summary=_thinking_summary(parsed.record, thinking_mode),
            ))
            pending_tokens += encoded.length
            total_records += 1
            total_tokens += encoded.length
            type_counts[str(metadata.record_type or "<unknown>")] += 1
            source_counts[str(metadata.source or "<unknown>")] += 1
            if len(pending) >= _SHARD_MAX_RECORDS or pending_tokens >= _SHARD_MAX_TOKENS:
                shards.append(_write_shard(temp, len(shards), pending))
                pending = []
                pending_tokens = 0
        if pending:
            shards.append(_write_shard(temp, len(shards), pending))
        if not total_records:
            raise ValueError(f"没有可写入 chat token cache 的记录：{path}")
        source_key = stable_path(path)
        manifest = {
            "format": CHAT_TOKEN_CACHE_FORMAT,
            "schema_version": CHAT_TOKEN_CACHE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "max_sequence_length": max_sequence_length,
            "min_tokens": 1,
            "drop_short": False,
            "total_records": total_records,
            "total_tokens": total_tokens,
            "source_files": [_file_fingerprint(path)],
            "tokenizer": _chat_tokenizer_fingerprint(tokenizer),
            "encoding": _encoding_signature(
                thinking_mode=thinking_mode,
                template_version=template_version,
                max_sequence_length=max_sequence_length,
                expected_types=expected_types,
            ),
            "shards": shards,
            "extra_metadata": {
                "type_counts": dict(sorted(type_counts.items())),
                "source_counts": dict(sorted(source_counts.items())),
                "source_record_ranges": {source_key: {"start": 0, "count": total_records}},
                "source_summaries": {source_key: {
                    "type_counts": dict(sorted(type_counts.items())),
                    "source_counts": dict(sorted(source_counts.items())),
                }},
            },
        }
        with (temp / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        if destination.exists():
            # Windows 不允许覆盖非空目录；保留旧目录直到新目录完整写入。
            backup = destination.with_name(f"{destination.name}_previous")
            if backup.exists():
                shutil.rmtree(backup)
            destination.rename(backup)
        else:
            backup = None
        try:
            temp.replace(destination)
        except Exception:
            if backup is not None and not destination.exists():
                backup.rename(destination)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


class ChatTokenCache(TokenCache):
    """四个对齐序列的只读 mmap 缓存；复用 text cache 的统计接口。"""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        with (self.cache_dir / "manifest.json").open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("format") != CHAT_TOKEN_CACHE_FORMAT:
            raise ValueError("chat token cache 格式不匹配。")
        if self.manifest.get("schema_version") != CHAT_TOKEN_CACHE_SCHEMA_VERSION:
            raise ValueError("chat token cache schema 版本不匹配。")
        self._shards = tuple(self.manifest.get("shards", ()))
        self._cumulative = []
        total = 0
        for shard in self._shards:
            count = int(shard["records"])
            if count <= 0:
                raise ValueError("chat token cache 分片记录数必须为正数。")
            total += count
            self._cumulative.append(total)
        if total != int(self.manifest.get("total_records", -1)):
            raise ValueError("chat token cache 分片记录数不匹配。")
        for shard in self._shards:
            root = self.cache_dir / "shards" / shard["name"]
            for filename in (
                "input_ids.npy", "labels.npy", "thinking_mode_ids.npy",
                "target_channel_ids.npy", "offsets.npy", "lengths.npy", "records.jsonl",
            ):
                if not (root / filename).is_file():
                    raise ValueError(f"chat token cache 分片不完整：{root / filename}")
        self._loaded_arrays = {}

    def __getitem__(self, index):
        """按源记录序号读取单条样本，供缓存校验与调试。"""
        shard_index, local_index = self._locate(index)
        return next(self.iter_records(
            start=self._cumulative[shard_index] - int(self._shards[shard_index]["records"]) + local_index,
            stop=self._cumulative[shard_index] - int(self._shards[shard_index]["records"]) + local_index + 1,
            as_numpy=True,
        ))

    def _arrays(self, shard_index):
        arrays = self._loaded_arrays.get(shard_index)
        if arrays is None:
            root = self.cache_dir / "shards" / self._shards[shard_index]["name"]
            arrays = tuple(
                np.load(root / filename, mmap_mode="r", allow_pickle=False)
                for filename in (
                    "input_ids.npy", "labels.npy", "thinking_mode_ids.npy",
                    "target_channel_ids.npy", "offsets.npy", "lengths.npy",
                )
            )
            self._loaded_arrays[shard_index] = arrays
        return arrays

    def iter_records(self, start=0, stop=None, *, as_numpy=False):
        """顺序读取元数据，仅在需要时加载并切片 mmap 数组。"""
        start = max(0, int(start))
        stop = len(self) if stop is None else min(len(self), int(stop))
        if stop <= start:
            return
        global_index = 0
        for shard_index, shard in enumerate(self._shards):
            shard_start = global_index
            shard_end = shard_start + int(shard["records"])
            global_index = shard_end
            if shard_end <= start or shard_start >= stop:
                continue
            input_ids, labels, thinking_ids, channel_ids, offsets, _lengths = self._arrays(shard_index)
            metadata_path = self.cache_dir / "shards" / shard["name"] / "records.jsonl"
            with metadata_path.open("r", encoding="utf-8") as handle:
                for local_index, line in enumerate(handle):
                    absolute_index = shard_start + local_index
                    if absolute_index < start:
                        continue
                    if absolute_index >= stop:
                        break
                    begin = int(offsets[local_index])
                    end = int(offsets[local_index + 1])
                    fields = (value[begin:end] for value in (input_ids, labels, thinking_ids, channel_ids))
                    if not as_numpy:
                        fields = (np.asarray(value, dtype=np.int64) for value in fields)
                    payload = json.loads(line)
                    thinking_summary = payload.pop("thinking_summary", {})
                    yield ChatTokenizedCacheRecord(
                        *fields,
                        metadata=RecordMetadata(**payload),
                        thinking_summary=thinking_summary,
                    )

    def iter_lengths(self):
        """供长度分桶读取预编码后的长度，不重新解析 JSONL。"""
        for shard_index in range(len(self._shards)):
            for value in self._arrays(shard_index)[-1]:
                yield int(value)


def ensure_manifest_chat_token_cache(
    manifest_path,
    cache_dir,
    tokenizer,
    *,
    expected_types=None,
    thinking_mode=THINKING_MODE_AUTO,
    template_version=None,
    max_sequence_length=None,
    num_workers=0,
    chunk_bytes=DEFAULT_JSONL_CHUNK_BYTES,
    overwrite=False,
    skip_paths=None,
    include_zero_weight=False,
):
    """通过通用 manifest 编排器增量构建或复用 chat cache。"""
    if tokenizer is None:
        raise ValueError("构建 chat token cache 必须提供 tokenizer。")
    if max_sequence_length is not None and int(max_sequence_length) <= 0:
        raise ValueError("max_sequence_length 必须为正整数。")
    if int(num_workers) < 0 or int(chunk_bytes) <= 0:
        raise ValueError("num_workers 不能为负数，chunk_bytes 必须为正整数。")
    current_template_version = GlobalConfig.chat_template_version
    template_version = current_template_version if template_version is None else template_version
    if str(template_version) != str(current_template_version):
        raise ValueError("指定模板版本与当前训练编码入口不一致。")
    tokenizer_identity = _chat_tokenizer_fingerprint(tokenizer)
    encoding = _encoding_signature(
        thinking_mode=thinking_mode,
        template_version=template_version,
        max_sequence_length=max_sequence_length,
        expected_types=expected_types,
    )

    def compatibility_checker(cache, paths):
        """检查 chat 源文件、tokenizer 和对话编码策略。"""
        ranges = cache.source_record_ranges()
        return (
            cache.fingerprint_matches(paths)
            and cache.manifest.get("tokenizer", {}).get("sha256")
            == tokenizer_identity["sha256"]
            and cache.manifest.get("encoding") == encoding
            and cache.manifest.get("drop_short") is False
            and len(cache) == ranges.get(stable_path(paths[0]), {}).get("count")
        )

    def build_file(path, destination):
        """构建单个 chat 源文件 cache。"""
        _build_chat_file_cache(
            path,
            destination,
            tokenizer,
            expected_types=expected_types,
            thinking_mode=thinking_mode,
            template_version=template_version,
            max_sequence_length=max_sequence_length,
            num_workers=num_workers,
            chunk_bytes=chunk_bytes,
        )

    def prepare_destination(destination):
        """恢复或清理上次原子发布留下的 chat 备份目录。"""
        backup = destination.with_name(f"{destination.name}_previous")
        if not backup.exists():
            return
        if destination.exists():
            shutil.rmtree(backup)
        else:
            backup.rename(destination)

    return ensure_manifest_cache(
        manifest_path,
        cache_dir,
        cache_class=ChatTokenCache,
        cache_subdir_name_fn=chat_cache_subdir_name,
        compatibility_checker=compatibility_checker,
        build_file=build_file,
        cache_label="chat token cache",
        overwrite=overwrite,
        skip_paths=skip_paths,
        prepare_destination=prepare_destination,
        include_zero_weight=include_zero_weight,
    )


class ChatTokenCacheManifestDataset(TokenCacheManifestDataset):
    """按 text cache 相同的 manifest 采样和进度规则读取 chat 缓存。"""

    _cache_class = ChatTokenCache
    _cache_subdir_name = staticmethod(chat_cache_subdir_name)
    _cache_format = CHAT_TOKEN_CACHE_FORMAT
    _cache_schema_version = CHAT_TOKEN_CACHE_SCHEMA_VERSION
    _cache_min_tokens = 1
    _cache_drop_short = False

    def _build_sample(
        self,
        record,
        entry_plan,
        *,
        record_index,
        sample_ordinal,
        pass_index,
        pass_kind,
    ):
        """复制截断后的四个序列，防止 mmap 关闭后训练样本悬空。"""
        max_length = self.max_sequence_length
        arrays = (
            record.input_ids,
            record.labels,
            record.thinking_mode_ids,
            record.target_channel_ids,
        )
        arrays = tuple(
            np.asarray(value[:max_length], dtype=np.int64).copy()
            if max_length is not None else np.asarray(value, dtype=np.int64).copy()
            for value in arrays
        )
        sample = {
            "id": record.metadata.record_id,
            "type": record.metadata.record_type or "chat",
            "source": record.metadata.source,
            PRE_TOKENIZED_CHAT_INPUT_IDS: arrays[0],
            PRE_TOKENIZED_CHAT_LABELS: arrays[1],
            PRE_TOKENIZED_CHAT_THINKING_MODE_IDS: arrays[2],
            PRE_TOKENIZED_CHAT_TARGET_CHANNEL_IDS: arrays[3],
            PRE_TOKENIZED_CHAT_THINKING_SUMMARY: record.thinking_summary,
            PRE_TOKENIZED_TOKEN_LENGTH: len(arrays[0]),
        }
        return attach_progress_metadata(
            sample,
            entry_plan,
            record_index=record_index,
            sample_ordinal=sample_ordinal,
            pass_index=pass_index,
            pass_kind=pass_kind,
        )


__all__ = [
    "CHAT_TOKEN_CACHE_FORMAT",
    "CHAT_TOKEN_CACHE_SCHEMA_VERSION",
    "ChatTokenCache",
    "ChatTokenCacheManifestDataset",
    "ChatTokenizedCacheRecord",
    "PRE_TOKENIZED_CHAT_INPUT_IDS",
    "PRE_TOKENIZED_CHAT_LABELS",
    "PRE_TOKENIZED_CHAT_THINKING_MODE_IDS",
    "PRE_TOKENIZED_CHAT_TARGET_CHANNEL_IDS",
    "PRE_TOKENIZED_CHAT_THINKING_SUMMARY",
    "chat_cache_subdir_name",
    "ensure_manifest_chat_token_cache",
]
