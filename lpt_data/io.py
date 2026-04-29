"""结构化数据读写工具。"""

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random

from torch.utils.data import IterableDataset

from .schema import normalize_dataset_record


@dataclass(frozen=True)
class ManifestEntryPlan:
    """描述单个 manifest 条目的流式采样计划。"""

    name: str
    path: Path
    weight: float
    repeat_count: int
    base_indices: tuple[int, ...] | None
    extra_indices: tuple[int, ...]
    selected_count: int


class StreamingManifestDataset(IterableDataset):
    """支持近似随机打乱的流式 manifest 数据集。"""

    def __init__(
        self,
        *,
        manifest_path,
        entry_plans,
        expected_types,
        shuffle_buffer_size,
        loaded_datasets,
        summary_types,
        summary_sources,
        total_count,
        seed=None,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.entry_plans = tuple(entry_plans)
        self.expected_types = None if expected_types is None else tuple(sorted(set(expected_types)))
        self.shuffle_buffer_size = max(1, int(shuffle_buffer_size))
        self.loaded_datasets = list(loaded_datasets)
        self.summary_types = dict(summary_types)
        self.summary_sources = dict(summary_sources)
        self.total_count = int(total_count)
        self.seed = seed
        self._iteration_index = 0

    def __len__(self):
        return self.total_count

    def _next_iteration_seed(self):
        if self.seed is None:
            return random.randrange(0, 2**31)
        current_seed = int(self.seed) + self._iteration_index
        self._iteration_index += 1
        return current_seed

    def _iter_manifest_records(self):
        for entry_plan in self.entry_plans:
            if entry_plan.repeat_count > 0:
                for _ in range(entry_plan.repeat_count):
                    yield from _iter_selected_records(
                        entry_plan.path,
                        entry_plan.base_indices,
                        expected_types=self.expected_types,
                    )
            if entry_plan.extra_indices:
                yield from _iter_selected_records(
                    entry_plan.path,
                    entry_plan.extra_indices,
                    expected_types=self.expected_types,
                )

    def iter_records_for_scan(self):
        """提供不影响训练 shuffle seed 的顺序扫描入口。"""
        yield from self._iter_manifest_records()

    def __iter__(self):
        rng = random.Random(self._next_iteration_seed())
        yield from _iter_buffer_shuffled_records(
            self._iter_manifest_records(),
            buffer_size=self.shuffle_buffer_size,
            rng=rng,
        )


def _iter_dataset_records_with_index(dataset_path, expected_types=None):
    dataset_path = Path(dataset_path)
    if dataset_path.suffix != ".jsonl":
        raise ValueError(f"只支持 JSONL 数据集，当前路径为: {dataset_path}")

    allowed_types = None if expected_types is None else set(expected_types)
    record_count = 0
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line_number, raw_line in enumerate(dataset_file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            payload = json.loads(line)
            default_id = f"{dataset_path.stem}:{line_number}"
            record = normalize_dataset_record(payload, default_id=default_id)
            if allowed_types is not None and record["type"] not in allowed_types:
                raise ValueError(
                    f"{dataset_path} 包含不被允许的样本类型: {record['type']}，"
                    f"期望类型为 {sorted(allowed_types)}"
                )
            yield record_count, record
            record_count += 1

    if record_count == 0:
        raise ValueError(f"数据集为空: {dataset_path}")


def load_dataset_records(dataset_path):
    """从 JSONL 文件读取结构化样本。"""
    return [record for _, record in _iter_dataset_records_with_index(dataset_path)]


def summarize_dataset_types(records):
    """统计不同样本类型的数量。"""
    counter = Counter(record["type"] for record in records)
    return dict(sorted(counter.items()))


def summarize_dataset_sources(records):
    """统计不同来源的样本数量。"""
    counter = Counter(record.get("source", "<unknown>") for record in records)
    return dict(sorted(counter.items()))


def _resolve_manifest_dataset_path(manifest_path, dataset_path):
    resolved_path = Path(dataset_path)
    if resolved_path.is_absolute():
        return resolved_path
    return manifest_path.parent / resolved_path


def _deterministic_pick(records, target_count, seed_text):
    if target_count <= 0:
        return []
    if target_count >= len(records):
        return list(records)
    rng = random.Random(seed_text)
    picked_indices = sorted(rng.sample(range(len(records)), target_count))
    return [records[index] for index in picked_indices]


def _apply_dataset_entry_policy(records, entry, *, seed_text):
    sample_limit = entry.get("sample_limit")
    if sample_limit is not None:
        if not isinstance(sample_limit, int) or sample_limit <= 0:
            raise ValueError(f"sample_limit 非法: {sample_limit}")
        records = _deterministic_pick(records, min(len(records), sample_limit), f"{seed_text}:sample_limit")

    weight = entry.get("weight", 1.0)
    if not isinstance(weight, (int, float)) or weight < 0:
        raise ValueError(f"weight 非法: {weight}")

    if weight == 0:
        return []

    if weight == 1:
        return list(records)

    if weight < 1:
        target_count = max(1, int(round(len(records) * float(weight)))) if records else 0
        return _deterministic_pick(records, target_count, f"{seed_text}:down_weight")

    repeated_records = []
    integer_part = int(weight)
    fractional_part = float(weight) - integer_part
    for _ in range(integer_part):
        repeated_records.extend(records)

    if fractional_part > 0:
        target_count = max(1, int(round(len(records) * fractional_part))) if records else 0
        repeated_records.extend(
            _deterministic_pick(records, target_count, f"{seed_text}:fractional_weight")
        )

    return repeated_records


def _load_manifest_entries(manifest_path):
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    dataset_entries = manifest.get("datasets")
    if not isinstance(dataset_entries, list) or not dataset_entries:
        raise ValueError(f"manifest 缺少非空 datasets 列表: {manifest_path}")
    return manifest_path, dataset_entries


def _normalize_weight(weight):
    if not isinstance(weight, (int, float)) or weight < 0:
        raise ValueError(f"weight 非法: {weight}")
    return float(weight)


def _count_dataset_records(dataset_path, expected_types):
    total_count = 0
    for _, _ in _iter_dataset_records_with_index(dataset_path, expected_types=expected_types):
        total_count += 1
    return total_count


def _pick_indices_from_count(total_count, target_count, seed_text):
    if target_count <= 0:
        return ()
    if target_count >= total_count:
        return tuple(range(total_count))
    rng = random.Random(seed_text)
    return tuple(sorted(rng.sample(range(total_count), target_count)))


def _pick_from_base_selection(base_indices, base_count, total_count, target_count, seed_text):
    if target_count <= 0:
        return ()
    if target_count >= base_count:
        if base_indices is None:
            return tuple(range(total_count))
        return tuple(base_indices)
    if base_indices is None:
        return _pick_indices_from_count(total_count, target_count, seed_text)

    rng = random.Random(seed_text)
    picked_positions = sorted(rng.sample(range(base_count), target_count))
    return tuple(base_indices[position] for position in picked_positions)


def _build_manifest_entry_plan(manifest_path, entry, expected_types):
    dataset_path = entry.get("path")
    if not isinstance(dataset_path, str) or not dataset_path.strip():
        raise ValueError("manifest dataset 配置缺少合法 path。")

    resolved_dataset_path = _resolve_manifest_dataset_path(manifest_path, dataset_path)
    entry_name = entry.get("name") or resolved_dataset_path.stem
    seed_text = f"{manifest_path}:{entry_name}:{resolved_dataset_path}"
    total_count = _count_dataset_records(resolved_dataset_path, expected_types=expected_types)

    sample_limit = entry.get("sample_limit")
    base_indices = None
    base_count = total_count
    if sample_limit is not None:
        if not isinstance(sample_limit, int) or sample_limit <= 0:
            raise ValueError(f"sample_limit 非法: {sample_limit}")
        base_count = min(total_count, sample_limit)
        if base_count < total_count:
            base_indices = _pick_indices_from_count(
                total_count,
                base_count,
                f"{seed_text}:sample_limit",
            )

    weight = _normalize_weight(entry.get("weight", 1.0))
    if weight == 0:
        repeat_count = 0
        extra_indices = ()
    elif weight < 1:
        repeat_count = 0
        target_count = max(1, int(round(base_count * weight))) if base_count else 0
        extra_indices = _pick_from_base_selection(
            base_indices,
            base_count,
            total_count,
            target_count,
            f"{seed_text}:down_weight",
        )
    else:
        repeat_count = int(weight)
        fractional_part = weight - repeat_count
        fractional_count = max(1, int(round(base_count * fractional_part))) if fractional_part > 0 else 0
        extra_indices = _pick_from_base_selection(
            base_indices,
            base_count,
            total_count,
            fractional_count,
            f"{seed_text}:fractional_weight",
        )

    selected_count = base_count * repeat_count + len(extra_indices)
    return ManifestEntryPlan(
        name=entry_name,
        path=resolved_dataset_path,
        weight=weight,
        repeat_count=repeat_count,
        base_indices=base_indices,
        extra_indices=extra_indices,
        selected_count=selected_count,
    )


def _iter_selected_records(dataset_path, selected_indices, expected_types):
    selected_index_set = None if selected_indices is None else set(selected_indices)
    for record_index, record in _iter_dataset_records_with_index(dataset_path, expected_types=expected_types):
        if selected_index_set is None or record_index in selected_index_set:
            yield record


def _summarize_entry_plan(entry_plan, expected_types):
    if entry_plan.selected_count == 0:
        return Counter(), Counter()

    type_counter = Counter()
    source_counter = Counter()
    base_index_set = None if entry_plan.base_indices is None else set(entry_plan.base_indices)
    extra_index_set = set(entry_plan.extra_indices)

    for record_index, record in _iter_dataset_records_with_index(entry_plan.path, expected_types=expected_types):
        base_selected = base_index_set is None or record_index in base_index_set
        if entry_plan.repeat_count > 0 and base_selected:
            type_counter[record["type"]] += entry_plan.repeat_count
            source_counter[record.get("source", "<unknown>")] += entry_plan.repeat_count
        if record_index in extra_index_set:
            type_counter[record["type"]] += 1
            source_counter[record.get("source", "<unknown>")] += 1

    return type_counter, source_counter


def _iter_buffer_shuffled_records(records, buffer_size, rng):
    if buffer_size <= 1:
        yield from records
        return

    buffer = []
    for record in records:
        if len(buffer) < buffer_size:
            buffer.append(record)
            continue

        picked_index = rng.randrange(len(buffer))
        yield buffer[picked_index]
        buffer[picked_index] = record

    rng.shuffle(buffer)
    yield from buffer


def build_streaming_manifest_dataset(
    manifest_path,
    *,
    expected_types=None,
    shuffle_buffer_size=1024,
    seed=None,
):
    """构造流式 manifest 数据集，避免把所有样本一次性加载进内存。"""
    manifest_path, dataset_entries = _load_manifest_entries(manifest_path)

    entry_plans = []
    loaded_datasets = []
    global_type_counter = Counter()
    global_source_counter = Counter()
    total_count = 0

    for entry_index, entry in enumerate(dataset_entries, start=1):
        if not isinstance(entry, dict):
            raise TypeError(f"manifest 第 {entry_index} 个 dataset 配置必须是字典。")
        if entry.get("enabled", True) is False:
            continue

        entry_plan = _build_manifest_entry_plan(
            manifest_path,
            entry,
            expected_types=expected_types,
        )
        entry_type_counter, entry_source_counter = _summarize_entry_plan(
            entry_plan,
            expected_types=expected_types,
        )

        entry_plans.append(entry_plan)
        loaded_datasets.append(
            {
                "name": entry_plan.name,
                "path": str(entry_plan.path),
                "count": entry_plan.selected_count,
                "weight": entry_plan.weight,
            }
        )
        global_type_counter.update(entry_type_counter)
        global_source_counter.update(entry_source_counter)
        total_count += entry_plan.selected_count

    if total_count == 0:
        raise ValueError(f"manifest 未加载到任何样本: {manifest_path}")

    return StreamingManifestDataset(
        manifest_path=manifest_path,
        entry_plans=entry_plans,
        expected_types=expected_types,
        shuffle_buffer_size=shuffle_buffer_size,
        loaded_datasets=loaded_datasets,
        summary_types=dict(sorted(global_type_counter.items())),
        summary_sources=dict(sorted(global_source_counter.items())),
        total_count=total_count,
        seed=seed,
    )


def load_dataset_manifest(manifest_path, expected_types=None):
    """从 manifest 中加载并混合多份结构化数据集。"""
    manifest_path, dataset_entries = _load_manifest_entries(manifest_path)

    records = []
    loaded_datasets = []
    for entry_index, entry in enumerate(dataset_entries, start=1):
        if not isinstance(entry, dict):
            raise TypeError(f"manifest 第 {entry_index} 个 dataset 配置必须是字典。")
        if entry.get("enabled", True) is False:
            continue

        dataset_path = entry.get("path")
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            raise ValueError(f"manifest 第 {entry_index} 个 dataset 缺少合法 path。")

        resolved_dataset_path = _resolve_manifest_dataset_path(manifest_path, dataset_path)
        dataset_records = load_dataset_records(resolved_dataset_path)
        if expected_types is not None:
            invalid_types = sorted(
                {
                    record["type"]
                    for record in dataset_records
                    if record["type"] not in set(expected_types)
                }
            )
            if invalid_types:
                raise ValueError(
                    f"{resolved_dataset_path} 包含不被允许的样本类型: {invalid_types}，"
                    f"期望类型为 {sorted(set(expected_types))}"
                )

        entry_name = entry.get("name") or resolved_dataset_path.stem
        mixed_records = _apply_dataset_entry_policy(
            dataset_records,
            entry,
            seed_text=f"{manifest_path}:{entry_name}:{resolved_dataset_path}",
        )
        records.extend(mixed_records)
        loaded_datasets.append(
            {
                "name": entry_name,
                "path": str(resolved_dataset_path),
                "count": len(mixed_records),
                "weight": entry.get("weight", 1.0),
            }
        )

    if not records:
        raise ValueError(f"manifest 未加载到任何样本: {manifest_path}")

    return records, loaded_datasets
