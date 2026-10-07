"""基于离线 token cache 的 manifest 训练数据集（per-file 增量缓存）。

text 预训练启动阶段如果仍从 JSONL 全量扫描统计和哈希，2GB+ 语料会让
主进程长时间卡在单核解析上。本模块把 manifest 展开计划绑定到按源文件
独立管理的 :class:`TokenCache` mmap 后端，同时保留 manifest 的动态配置
能力：

* ``weight`` / ``sample_limit`` / ``enabled`` 只影响训练计划，不触发 cache
  重建；
* 新增数据集只需构建新文件对应的 cache 子目录，已有文件直接复用；
* 禁用数据集只是从计划中剔除，其 cache 子目录保留，重新启用无需重建；
* 计数、类型/来源统计、文件指纹全部来自各文件 cache manifest，训练启动
  不再回读 JSONL 或调用 tokenizer。

数据集对外暴露与 :class:`lpt_data.io.StreamingManifestDataset` 相同的接口
（``entry_plans``、``iter_records_for_epoch``、进度元数据），因此训练循环、
断点续训和数据进度账本无需感知后端差异。
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from torch.utils.data import IterableDataset

from .cache import (
    DEFAULT_JSONL_CHUNK_BYTES,
    TOKEN_CACHE_FORMAT,
    TOKEN_CACHE_SCHEMA_VERSION,
    TokenCache,
    build_token_cache,
    stable_path,
    tokenizer_fingerprint,
)
from .io import (
    DATA_PROGRESS_RECORD_KEY,
    attach_progress_metadata,
    build_manifest_dataset_plan,
    entry_progress_key,
    iter_buffer_shuffled_records,
    load_manifest_entries,
    progress_path_string,
    resolve_manifest_dataset_path,
)


# 预编码样本在数据集和批处理模块之间共享的字段名。
PRE_TOKENIZED_TOKEN_IDS = "_lpt_token_ids"
PRE_TOKENIZED_TOKEN_LENGTH = "_lpt_token_length"

# text cache 必须保留全部记录，且至少允许 1 token，确保 cache 序号与 JSONL
# 记录序号一一对应，进度账本和 entry plan 的 base/extra 下标才不会错位。
CACHE_MIN_TOKENS = 1
CACHE_DROP_SHORT = False


def _manifest_enabled_paths(manifest_path, *, include_zero_weight=False):
    """返回需要构建 cache 的 JSONL 路径（保持 manifest 顺序去重）。

    ``weight=0`` 的数据集保留在训练计划和进度账本中，但本轮不会产出
    样本。训练启动时默认跳过这类文件，避免为暂不参与训练的数据集分词；
    需要提前预热 cache 时可显式设置 ``include_zero_weight=True``。
    """
    manifest_path, dataset_entries = load_manifest_entries(manifest_path)
    paths = []
    seen = set()
    for entry in dataset_entries:
        if not isinstance(entry, dict) or entry.get("enabled", True) is False:
            continue
        dataset_path = entry.get("path")
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            continue
        weight = entry.get("weight", 1.0)
        if (
            not include_zero_weight
            and isinstance(weight, (int, float))
            and float(weight) == 0.0
        ):
            continue
        resolved_path = resolve_manifest_dataset_path(manifest_path, dataset_path)
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        paths.append(resolved_path)
    if not paths:
        raise ValueError(f"manifest 未包含任何启用的数据集: {manifest_path}")
    return paths


def cache_subdir_name(dataset_path) -> str:
    """为源文件生成稳定且不冲突的 cache 子目录名。

    用 stable path 的 sha256 前缀命名，避免文件名中的盘符、中文、超长路径
    造成目录名不合法；同一源文件在不同 manifest 中复用同一子目录。目录名
    统一使用下划线分隔，避免 Windows/工具链对点开头或带点目录名的特殊处理。
    """
    key = stable_path(Path(dataset_path))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"{digest}_cache"


def _cache_is_compatible(
    cache,
    paths,
    *,
    expected_types,
    max_sequence_length,
    min_tokens,
    drop_short,
    tokenizer,
):
    """判断某个源文件的 cache 是否仍与当前文件、tokenizer 和长度策略一致。"""
    if not cache.fingerprint_matches(paths):
        return False
    if int(cache.manifest.get("max_sequence_length") or 0) != int(
        max_sequence_length or 0
    ):
        return False
    if int(cache.manifest.get("min_tokens", -1)) != int(min_tokens):
        return False
    if bool(cache.manifest.get("drop_short", True)) is not bool(drop_short):
        return False
    if expected_types is not None:
        allowed_types = set(expected_types)
        if set(cache.type_counts().keys()) - allowed_types:
            return False
    current_tokenizer = tokenizer_fingerprint(tokenizer, None)
    cached_tokenizer = cache.manifest.get("tokenizer")
    if (
        not isinstance(cached_tokenizer, dict)
        or cached_tokenizer.get("sha256") != current_tokenizer["sha256"]
    ):
        return False
    ranges = cache.source_record_ranges()
    if not ranges or len(ranges) != len(paths):
        return False
    return True


def ensure_manifest_cache(
    manifest_path,
    cache_dir,
    *,
    cache_class,
    cache_subdir_name_fn,
    compatibility_checker,
    build_file,
    cache_label,
    overwrite=False,
    skip_paths=None,
    prepare_destination=None,
    include_zero_weight=False,
):
    """按源文件增量校验或构建 manifest cache。

    text、SFT 和 LoRA 的 cache 字段不同，但增量编排完全一致：展开
    manifest、跳过已完成文件、校验现有 cache、逐文件构建并输出统计。将
    这段流程集中在这里，调用方只需提供具体 cache 类型、兼容性判断和
    单文件构建函数即可。``build_file`` 接收 ``(source_path,
    destination)`` 两个位置参数；``compatibility_checker`` 接收
    ``(cache, [source_path])`` 两个位置参数。回调抛出的异常会按旧流程
    视为 cache 失效并触发重建。

    ``prepare_destination`` 用于处理后端特有的原子发布残留（例如 chat
    cache 的 ``*_previous`` 备份）。它在打开现有 cache 之前调用，且可
    直接修改目标目录；没有特殊处理时传入 ``None``。
    """
    paths = _manifest_enabled_paths(
        manifest_path,
        include_zero_weight=include_zero_weight,
    )
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    skip_set = {Path(path).resolve() for path in (skip_paths or ())}
    total_paths = len(paths)
    built_count = 0
    reused_count = 0
    skipped_count = 0

    for path_index, path in enumerate(paths, start=1):
        if Path(path).resolve() in skip_set:
            skipped_count += 1
            print(
                f"[{cache_label}] 跳过 {path_index}/{total_paths}: {path.name} "
                "（数据集已完成训练）",
                flush=True,
            )
            continue

        destination = root / cache_subdir_name_fn(path)
        if prepare_destination is not None:
            prepare_destination(destination)

        cache = None
        compatible = False
        if destination.exists() and not overwrite:
            try:
                cache = cache_class(destination)
                compatible = bool(compatibility_checker(cache, [path]))
            except Exception:
                # cache 缺文件、schema 不匹配或指纹失效都统一走重建路径。
                compatible = False
            finally:
                # 无论是否复用都立即释放 mmap，避免重建时旧目录句柄占用。
                if cache is not None:
                    cache.close()
            if compatible:
                reused_count += 1
                continue

        print(
            f"[{cache_label}] 正在构建 {path_index}/{total_paths}: "
            f"{path.name} ...",
            flush=True,
        )
        build_file(path, destination)
        built_count += 1
        print(
            f"[{cache_label}] 完成 {path_index}/{total_paths}: {path.name}",
            flush=True,
        )

    print(
        f"[{cache_label}] 校验完成: 总数={total_paths}, 复用={reused_count}, "
        f"新建={built_count}, 跳过={skipped_count}",
        flush=True,
    )
    return root


def ensure_manifest_token_cache(
    manifest_path,
    cache_dir,
    tokenizer,
    *,
    expected_types=None,
    max_sequence_length=None,
    num_workers=0,
    chunk_bytes=DEFAULT_JSONL_CHUNK_BYTES,
    overwrite=False,
    skip_paths=None,
    include_zero_weight=False,
):
    """按源文件增量校验或构建离线 token cache。

    每个 JSONL 对应 ``cache_dir`` 下的一个独立子目录（见
    :func:`cache_subdir_name`）。已有子目录只有同时满足源文件指纹、
    tokenizer 指纹、长度策略和类型集合一致时才复用；缺失或失效的文件才
    单独重建。因此 manifest 增删数据集、调整 weight/sample_limit/enabled
    都不会触发无关文件的重建。``skip_paths`` 内的文件（已完成训练且
    数据未变动）即使没有 cache 也不会构建。返回值是 cache 容器目录。
    """
    if tokenizer is None:
        raise ValueError("构建 text token cache 必须提供 tokenizer。")
    min_tokens = CACHE_MIN_TOKENS
    drop_short = CACHE_DROP_SHORT

    def compatibility_checker(cache, paths):
        """检查 text cache 的源文件、tokenizer 和编码策略。"""
        return _cache_is_compatible(
            cache,
            paths,
            expected_types=expected_types,
            max_sequence_length=max_sequence_length,
            min_tokens=min_tokens,
            drop_short=drop_short,
            tokenizer=tokenizer,
        )

    def build_file(path, destination):
        """构建单个 text 源文件 cache。"""
        build_token_cache(
            [path],
            destination,
            tokenizer=tokenizer,
            expected_types=expected_types,
            max_sequence_length=max_sequence_length,
            min_tokens=min_tokens,
            drop_short=drop_short,
            num_workers=num_workers,
            chunk_bytes=chunk_bytes,
            overwrite=bool(overwrite or destination.exists()),
        )

    return ensure_manifest_cache(
        manifest_path,
        cache_dir,
        cache_class=TokenCache,
        cache_subdir_name_fn=cache_subdir_name,
        compatibility_checker=compatibility_checker,
        build_file=build_file,
        cache_label="token cache",
        overwrite=overwrite,
        skip_paths=skip_paths,
        include_zero_weight=include_zero_weight,
    )


class TokenCacheManifestDataset(IterableDataset):
    """把 manifest 展开计划绑定到一组 per-file TokenCache 的训练数据集。"""

    # 训练循环据此跳过 JSONL reader pool 和内容哈希扫描。
    uses_token_cache = True
    _cache_class = TokenCache
    _cache_subdir_name = staticmethod(cache_subdir_name)

    # 缓存格式的约束由数据集类型声明。chat/SFT/LoRA 子类只需覆盖这些
    # 类属性即可复用下面的 manifest、进度和采样流程，不再依赖 text 的
    # schema/drop_short 硬编码。
    _cache_format = TOKEN_CACHE_FORMAT
    _cache_schema_version = TOKEN_CACHE_SCHEMA_VERSION
    _cache_min_tokens = CACHE_MIN_TOKENS
    _cache_drop_short = CACHE_DROP_SHORT

    # 这些字段会影响缓存内容或训练语义，纳入进度指纹。子类可通过覆盖
    # 元组补充自己的编码策略字段（例如 chat 的 thinking/template 版本）。
    _cache_fingerprint_strategy_fields = (
        "max_sequence_length",
        "min_tokens",
        "drop_short",
        "encoding",
        "encoding_strategy",
    )

    def __init__(
        self,
        *,
        manifest_path,
        cache_dir,
        expected_types=None,
        shuffle_buffer_size=1024,
        seed=None,
        completed_entries=None,
        max_sequence_length=None,
    ):
        """打开各源文件 cache 并编译 manifest 采样计划。"""
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.cache_root = Path(cache_dir)
        self.expected_types = (
            None if expected_types is None else tuple(sorted(set(expected_types)))
        )
        self.shuffle_buffer_size = max(1, int(shuffle_buffer_size))
        self.seed = seed
        self.max_sequence_length = (
            None if max_sequence_length is None else int(max_sequence_length)
        )
        if self.max_sequence_length is not None and self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length 必须为正整数。")
        self._iteration_index = 0
        self._completed_entries = {
            str(key): int(count)
            for key, count in (completed_entries or {}).items()
        }

        self._caches: dict[str, TokenCache] = {}
        manifest_path, dataset_entries = load_manifest_entries(manifest_path)
        for entry in dataset_entries:
            if not isinstance(entry, dict) or entry.get("enabled", True) is False:
                continue
            weight = entry.get("weight", 1.0)
            if isinstance(weight, (int, float)) and float(weight) == 0.0:
                # 零权重 entry 保留在计划和进度账本中，但不需要 cache 后端。
                continue
            dataset_path = entry.get("path")
            if not isinstance(dataset_path, str) or not dataset_path.strip():
                continue
            resolved_path = resolve_manifest_dataset_path(manifest_path, dataset_path)
            entry_key = entry_progress_key(
                entry.get("name") or resolved_path.stem,
                resolved_path,
            )
            if entry_key in self._completed_entries:
                # 已完成数据集不再参与训练，不需要打开或构建 cache。
                continue
            cache = self._cache_class(
                self.cache_root / self._cache_subdir_name(resolved_path)
            )
            self._validate_cache_manifest(cache)
            self._caches[stable_path(resolved_path)] = cache

        (
            self.entry_plans,
            self.loaded_datasets,
            self.summary_types,
            self.summary_sources,
            self.total_count,
        ) = build_manifest_dataset_plan(
            manifest_path,
            expected_types=expected_types,
            count_provider=self._count_provider,
            summary_provider=self._summary_provider,
        )
        # 已完成数据集从训练计划中完全移除：不产出样本、不进入进度账本，
        # 也不在 source_fingerprints / 迭代路径中引用其 cache。
        kept_plans = []
        kept_loaded = []
        for plan, loaded in zip(self.entry_plans, self.loaded_datasets):
            if progress_path_string(plan.path) in self._completed_entries:
                continue
            kept_plans.append(plan)
            kept_loaded.append(loaded)
        self.entry_plans = tuple(kept_plans)
        self.loaded_datasets = kept_loaded
        self.total_count = sum(
            int(plan.selected_count) for plan in self.entry_plans
        )

    @classmethod
    def _validate_cache_manifest(cls, cache):
        """校验 cache 类型声明，子类可覆盖以增加专用约束。

        ``TokenCache`` 和 ``ChatTokenCache`` 使用不同的磁盘格式，但共享
        manifest 数据集编排。把格式、schema 及记录保留策略集中成类属性后，
        这里不再把 text 的约束误用于 chat 数据集。
        """
        manifest = cache.manifest
        expected_format = cls._cache_format
        if expected_format is not None and manifest.get("format") != expected_format:
            raise ValueError(
                f"{expected_format} cache 格式不匹配："
                f"{manifest.get('format')!r}。"
            )
        expected_schema = cls._cache_schema_version
        if expected_schema is not None and int(
            manifest.get("schema_version", -1)
        ) != int(expected_schema):
            raise ValueError(
                f"{expected_format or 'token'} cache schema 版本不匹配。"
            )
        expected_min_tokens = cls._cache_min_tokens
        if expected_min_tokens is not None and int(
            manifest.get("min_tokens", -1)
        ) != int(expected_min_tokens):
            raise ValueError(
                f"{expected_format or 'token'} cache min_tokens 策略不匹配。"
            )
        expected_drop_short = cls._cache_drop_short
        if expected_drop_short is not None and bool(
            manifest.get("drop_short", True)
        ) is not bool(expected_drop_short):
            if expected_drop_short is False:
                detail = "必须保留全部记录（drop_short=False），否则 cache 序号与记录序号无法一一对应"
            else:
                detail = f"drop_short 必须为 {bool(expected_drop_short)!r}"
            raise ValueError(
                f"{expected_format or 'token'} cache 记录保留策略不匹配：{detail}。"
            )

    @classmethod
    def _cache_fingerprint(cls, cache):
        """基于单个 cache manifest 生成稳定的内容指纹，不重新扫描 JSONL。

        除源文件和 tokenizer 外，格式、schema、长度过滤及编码策略也会
        改变训练样本。全部纳入指纹可避免旧 cache 与新策略共用同一进度账本。
        """
        digest = hashlib.sha256()
        payload = {
            "format": cache.manifest.get("format"),
            "schema_version": cache.manifest.get("schema_version"),
            "source_files": cache.manifest.get("source_files"),
            "tokenizer": cache.manifest.get("tokenizer"),
            "source_record_ranges": cache.source_record_ranges(),
            "source_summaries": cache.source_summaries(),
        }
        for field in cls._cache_fingerprint_strategy_fields:
            payload[field] = cache.manifest.get(field)
        digest.update(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        return digest.hexdigest()

    def _cache_for(self, dataset_path):
        """按 stable path 定位该源文件对应的 per-file cache。"""
        key = stable_path(Path(dataset_path))
        cache = self._caches.get(key)
        if cache is None:
            raise ValueError(
                f"token cache 缺少源文件: {dataset_path}（key={key}）"
            )
        return cache

    def _count_provider(self, dataset_path):
        """从 per-file cache 读取源文件记录数。"""
        key = progress_path_string(dataset_path)
        if key in self._completed_entries:
            return self._completed_entries[key]
        return len(self._cache_for(dataset_path))

    def _summary_provider(self, entry_plan):
        """返回该 entry 的类型/来源统计，全量选择时直接复用构建期聚合值。"""
        if progress_path_string(entry_plan.path) in self._completed_entries:
            return Counter(), Counter()
        if int(entry_plan.selected_count) <= 0:
            return Counter(), Counter()
        cache = self._cache_for(entry_plan.path)
        if int(entry_plan.selected_count) == int(entry_plan.source_record_count):
            return (
                Counter(cache.type_counts()),
                Counter(cache.source_counts()),
            )
        return self._summarize_cache(entry_plan, cache)

    def _summarize_cache(self, entry_plan, cache):
        """按 base/extra 选择规则遍历 cache 记录统计（与 JSONL 后端语义一致）。"""
        type_counter = Counter()
        source_counter = Counter()
        base_index_set = (
            None if entry_plan.base_indices is None else set(entry_plan.base_indices)
        )
        extra_index_set = set(entry_plan.extra_indices)
        for record_index, record in self._iter_selected_records(
            entry_plan,
            selected_indices=None,
        ):
            base_selected = base_index_set is None or record_index in base_index_set
            if entry_plan.repeat_count > 0 and base_selected:
                type_counter[record.metadata.record_type or "<unknown>"] += entry_plan.repeat_count
                source_counter[record.metadata.source or "<unknown>"] += entry_plan.repeat_count
            if record_index in extra_index_set:
                type_counter[record.metadata.record_type or "<unknown>"] += 1
                source_counter[record.metadata.source or "<unknown>"] += 1
        return type_counter, source_counter

    def __len__(self):
        """返回按 manifest weight/sample_limit 展开后的样本数。"""
        return int(self.total_count)

    def _iter_selected_records(self, entry_plan, *, selected_indices):
        """顺序遍历源文件 cache，并按源记录索引过滤。

        token 序列以 numpy 数组切片产出（不复制），避免把 3 亿级 token
        逐 int 转成 Python tuple 造成主进程单核瓶颈。
        """
        cache = self._cache_for(entry_plan.path)
        selected_set = None if selected_indices is None else set(selected_indices)
        for record in cache.iter_records(as_numpy=True):
            record_index = int(record.metadata.record_index)
            if selected_set is None or record_index in selected_set:
                yield record_index, record

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
        """把 cache 记录包装成训练样本，并附加与 JSONL 后端一致的进度元数据。"""
        # 先对 mmap 视图截断，再复制为训练张量所需的 int64。旧 cache 可能
        # 保存数万 token 的原始记录；若先 copy/cast 再由 collator 截断，会
        # 白白放大主进程内存带宽并造成单核忙等。
        if isinstance(record.token_ids, np.ndarray):
            token_view = record.token_ids
            if self.max_sequence_length is not None:
                token_view = token_view[: self.max_sequence_length]
            token_ids = token_view.astype(np.int64, copy=True)
        else:
            token_values = record.token_ids
            if self.max_sequence_length is not None:
                token_values = token_values[: self.max_sequence_length]
            token_ids = tuple(int(value) for value in token_values)
        sample = {
            "id": record.metadata.record_id,
            "type": record.metadata.record_type or "text",
            "source": record.metadata.source,
            PRE_TOKENIZED_TOKEN_IDS: token_ids,
            PRE_TOKENIZED_TOKEN_LENGTH: len(token_ids),
        }
        return attach_progress_metadata(
            sample,
            entry_plan,
            record_index=record_index,
            sample_ordinal=sample_ordinal,
            pass_index=pass_index,
            pass_kind=pass_kind,
        )

    def _iter_manifest_records(self):
        """按 entry plan 顺序产出带进度元数据的预编码样本。"""
        for entry_plan in self.entry_plans:
            sample_ordinal = 0
            if entry_plan.repeat_count > 0:
                for repeat_index in range(entry_plan.repeat_count):
                    for record_index, record in self._iter_selected_records(
                        entry_plan,
                        selected_indices=entry_plan.base_indices,
                    ):
                        yield self._build_sample(
                            record,
                            entry_plan,
                            record_index=record_index,
                            sample_ordinal=sample_ordinal,
                            pass_index=repeat_index,
                            pass_kind="repeat",
                        )
                        sample_ordinal += 1
            if entry_plan.extra_indices:
                for extra_index, (record_index, record) in enumerate(
                    self._iter_selected_records(
                        entry_plan,
                        selected_indices=entry_plan.extra_indices,
                    )
                ):
                    yield self._build_sample(
                        record,
                        entry_plan,
                        record_index=record_index,
                        sample_ordinal=sample_ordinal,
                        pass_index=extra_index,
                        pass_kind="extra",
                    )
                    sample_ordinal += 1

    def _seed_for_epoch(self, epoch_index):
        """根据训练 epoch 生成稳定 shuffle seed，不推进内部迭代计数。"""
        if self.seed is None:
            return random.randrange(0, 2**31)
        return int(self.seed) + int(epoch_index)

    def iter_records_for_epoch(self, epoch_index, *, reader_pool=None):
        """按指定 epoch 产出带来源元数据的训练样本。

        ``reader_pool`` 仅用于兼容 JSONL 后端的调用约定，cache 后端直接忽略。
        """
        worker_seed = self._seed_for_epoch(epoch_index)
        rng = random.Random(worker_seed)
        yield from iter_buffer_shuffled_records(
            self._iter_manifest_records(),
            buffer_size=self.shuffle_buffer_size,
            rng=rng,
        )

    def iter_length_records_for_epoch(self, epoch_index, *, reader_pool=None):
        """只读取缓存长度，重放指定轮次的样本顺序与进度身份。

        动态批处理需要在启动时估算剩余批次数。这里复用缓存的长度数组，
        不读取记录 JSONL、分词或复制 token 内容；长度记录与实际训练记录
        使用相同的 manifest 展开规则及缓冲打乱规则，因此可共用进度过滤
        和批次划分逻辑。对话缓存子类同样提供 ``iter_lengths``，无需额外
        实现。``reader_pool`` 与训练迭代接口一致，缓存后端不使用它。
        """

        def iter_lengths():
            """按真实训练的重复轮次、补充样本顺序生成轻量记录。"""
            for entry_plan in self.entry_plans:
                if int(entry_plan.selected_count) <= 0:
                    continue
                cache = self._cache_for(entry_plan.path)
                sample_ordinal = 0
                passes = (
                    (repeat_index, "repeat", entry_plan.base_indices)
                    for repeat_index in range(entry_plan.repeat_count)
                )
                # 分数权重对应补充集合，始终位于完整重复轮次之后。
                for pass_index, pass_kind, indices in passes:
                    selected_set = None if indices is None else set(indices)
                    for record_index, length in enumerate(cache.iter_lengths()):
                        if selected_set is not None and record_index not in selected_set:
                            continue
                        if self.max_sequence_length is not None:
                            length = min(int(length), self.max_sequence_length)
                        yield attach_progress_metadata(
                            {PRE_TOKENIZED_TOKEN_LENGTH: int(length)},
                            entry_plan,
                            record_index=record_index,
                            sample_ordinal=sample_ordinal,
                            pass_index=pass_index,
                            pass_kind=pass_kind,
                        )
                        sample_ordinal += 1
                if entry_plan.extra_indices:
                    selected_set = set(entry_plan.extra_indices)
                    extra_index = 0
                    for record_index, length in enumerate(cache.iter_lengths()):
                        if record_index not in selected_set:
                            continue
                        if self.max_sequence_length is not None:
                            length = min(int(length), self.max_sequence_length)
                        yield attach_progress_metadata(
                            {PRE_TOKENIZED_TOKEN_LENGTH: int(length)},
                            entry_plan,
                            record_index=record_index,
                            sample_ordinal=sample_ordinal,
                            pass_index=extra_index,
                            pass_kind="extra",
                        )
                        sample_ordinal += 1
                        extra_index += 1

        yield from iter_buffer_shuffled_records(
            iter_lengths(),
            buffer_size=self.shuffle_buffer_size,
            rng=random.Random(self._seed_for_epoch(epoch_index)),
        )

    def __iter__(self):
        """训练迭代入口：对流式记录做 buffer shuffle。"""
        current_iteration = self._iteration_index
        self._iteration_index += 1
        yield from self.iter_records_for_epoch(current_iteration)

    def close(self):
        """释放所有已打开的 mmap；重复调用和垃圾回收时都是安全的。"""
        for cache in self._caches.values():
            cache.close()
        self._caches.clear()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()

    @property
    def source_fingerprints(self):
        """返回 entry_key -> (source_record_count, content_hash)。

        每个 entry 使用对应文件 cache 的轻量摘要，避免 ``DataProgressTracker``
        再对 2GB JSONL 全量计算 sha256。
        """
        result = {}
        for entry_plan in self.entry_plans:
            if int(entry_plan.selected_count) <= 0:
                continue
            key = entry_progress_key(entry_plan.name, entry_plan.path)
            cache = self._cache_for(entry_plan.path)
            result[key] = (
                int(entry_plan.source_record_count),
                self._cache_fingerprint(cache),
            )
        return result


__all__ = [
    "CACHE_DROP_SHORT",
    "CACHE_MIN_TOKENS",
    "PRE_TOKENIZED_TOKEN_IDS",
    "PRE_TOKENIZED_TOKEN_LENGTH",
    "TokenCacheManifestDataset",
    "cache_subdir_name",
    "ensure_manifest_cache",
    "ensure_manifest_token_cache",
]
