"""LPT v2 结构化训练样本批处理工具。"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import random

import numpy as np
import torch

from lpt_config import GlobalConfig
from lpt_protocol import (
    DS_BOS_TOKEN,
    DS_EOS_TOKEN,
    DS_PAD_TOKEN,
    TARGET_CHANNEL_PROMPT,
    TARGET_CHANNEL_ANSWER,
    THINKING_MODE_OFF,
    THINKING_MODE_AUTO,
    target_channel_to_id,
    get_template_spec,
    render_training_segments,
    thinking_mode_to_id,
)
from .token_cache_dataset import (
    PRE_TOKENIZED_TOKEN_IDS,
)


# chat 离线缓存的四个监督序列；与 text 缓存使用不同字段，避免误走 text 分支。
PRE_TOKENIZED_CHAT_INPUT_IDS = "_lpt_chat_input_ids"
PRE_TOKENIZED_CHAT_LABELS = "_lpt_chat_labels"
PRE_TOKENIZED_CHAT_THINKING_MODE_IDS = "_lpt_chat_thinking_mode_ids"
PRE_TOKENIZED_CHAT_TARGET_CHANNEL_IDS = "_lpt_chat_target_channel_ids"


REQUIRED_TOKENIZER_TOKENS = (
    DS_BOS_TOKEN,
    DS_EOS_TOKEN,
    DS_PAD_TOKEN,
)


@dataclass(frozen=True)
class EncodedTrainingSample:
    """单条训练样本编码后的 token 结果。"""

    sample_id: str
    # token cache 预编码路径使用 numpy int64 数组，动态分词路径保持 tuple，
    # 两者都支持 len/索引，张量化时按类型走 C 快速路径。
    input_ids: tuple[int, ...] | np.ndarray
    labels: tuple[int, ...] | np.ndarray
    thinking_mode_ids: tuple[int, ...] | np.ndarray
    target_channel_ids: tuple[int, ...] | np.ndarray

    @property
    def length(self):
        """返回编码后 token 长度，packing 时以它作为容量单位。"""
        return len(self.input_ids)


@dataclass(frozen=True)
class PackedTrainingSequence:
    """sequence packing 后的一条训练序列。"""

    input_ids: tuple[int, ...] | np.ndarray
    labels: tuple[int, ...] | np.ndarray
    thinking_mode_ids: tuple[int, ...] | np.ndarray
    target_channel_ids: tuple[int, ...] | np.ndarray
    position_ids: tuple[int, ...] | np.ndarray
    segment_ids: tuple[int, ...] | np.ndarray

    @property
    def length(self):
        """返回 packed row 的真实 token 长度，不含 padding。"""
        return len(self.input_ids)


@dataclass(frozen=True)
class LengthBucketBatch:
    """长度计划中的一个 batch。

    计划只保存原始样本的整数下标，不复制样本内容。这样既能保持样本对象的
    身份，又能在断点续训或多进程之间安全地序列化计划。``bucket_id`` 使用
    ``-1`` 表示该 batch 不属于显式长度桶（例如顺序或全局排序策略）。
    """

    indices: tuple[int, ...]
    lengths: tuple[int, ...]
    bucket_id: int = -1

    @property
    def sample_count(self):
        """返回 batch 中的原始样本数。"""
        return len(self.indices)

    @property
    def token_count(self):
        """返回 batch 中未 padding 的 token 总数。"""
        return sum(self.lengths)

    @property
    def max_sequence_length(self):
        """返回 batch 内最长样本的 token 数。"""
        return max(self.lengths, default=0)

    @property
    def padded_token_count(self):
        """返回按最长样本 padding 后的 token 数。"""
        return self.sample_count * self.max_sequence_length

    @property
    def padding_token_count(self):
        """返回该 batch 因 padding 产生的 token 数。"""
        return self.padded_token_count - self.token_count


@dataclass(frozen=True)
class LengthBucketBatchPlan:
    """可重放的长度分桶 batch 计划。

    ``sample_lengths`` 与传入的样本序列一一对应，``batches`` 中的下标始终
    指向该序列的原始位置。计划本身不持有数据集对象，因此可以写入训练状态
    或发送到 worker；调用 :meth:`iter_batches` 才会将下标还原成原始样本引用。
    """

    batches: tuple[LengthBucketBatch, ...]
    sample_lengths: tuple[int, ...]
    strategy: str
    seed: int
    epoch: int
    batch_size: int
    drop_last: bool
    bucket_boundaries: tuple[int, ...] = ()
    packing_strategy: str = "none"

    @property
    def sample_count(self):
        """返回计划覆盖的样本数。"""
        return len(self.sample_lengths)

    @property
    def batch_count(self):
        """返回计划中的 batch 数。"""
        return len(self.batches)

    @property
    def sample_order(self):
        """返回展平后的原始样本下标顺序。"""
        return tuple(index for batch in self.batches for index in batch.indices)

    def iter_batches(self, samples):
        """按计划还原 batch，保留传入样本对象的身份。

        ``samples`` 可以是支持 ``len`` 和整数下标的序列，也可以是任意可迭代
        对象；后者会在调用时物化一次。每个返回值都是 list，方便直接传给
        ``build_training_batch`` 或 ``build_packed_training_batch``。
        """
        if not hasattr(samples, "__len__") or not hasattr(samples, "__getitem__"):
            samples = tuple(samples)
        if len(samples) != self.sample_count:
            raise ValueError(
                "还原长度计划时样本数量不一致: "
                f"{len(samples)} != {self.sample_count}。"
            )
        for batch in self.batches:
            yield [samples[index] for index in batch.indices]

    def materialize(self, samples):
        """返回由原始样本引用组成的 tuple batch，便于测试和缓存。"""
        return tuple(self.iter_batches(samples))


def prepare_tokenizer(tokenizer):
    """校验 tokenizer 是否满足当前 DS 模板协议。"""
    missing_tokens = [
        token
        for token in REQUIRED_TOKENIZER_TOKENS
        if tokenizer.convert_tokens_to_ids(token) is None
    ]
    if missing_tokens:
        raise ValueError(f"当前 tokenizer 缺少必须的 DS token: {missing_tokens}")

    if tokenizer.bos_token != DS_BOS_TOKEN:
        raise ValueError(
            "当前 tokenizer 的 bos_token 与 DS 模板定义不一致: "
            f"{tokenizer.bos_token} != {DS_BOS_TOKEN}"
        )

    if tokenizer.eos_token is None:
        raise ValueError("当前 tokenizer 未定义 eos_token。")

    template_spec = get_template_spec(GlobalConfig.chat_template_version)
    if tokenizer.eos_token != template_spec.eos_token:
        raise ValueError(
            "当前 tokenizer 的 eos_token 与模板定义不一致: "
            f"{tokenizer.eos_token} != {template_spec.eos_token}"
        )

    pad_token_id = tokenizer.convert_tokens_to_ids(DS_PAD_TOKEN)
    tokenizer.pad_token = DS_PAD_TOKEN
    if tokenizer.pad_token_id != pad_token_id:
        tokenizer.pad_token_id = pad_token_id
    return tokenizer


def _tokenize_rendered_segments(segments, tokenizer):
    """把模板渲染片段编码成 token 序列和原生 thinking 控制序列。"""
    input_ids = []
    labels = []
    thinking_mode_ids = []
    token_channel_ids = []
    for segment in segments:
        encoded = tokenizer(segment.text, add_special_tokens=False)
        segment_ids = encoded["input_ids"]
        if not segment_ids:
            continue
        # 用户提示、系统提示等非监督片段写入 -100，让 cross entropy 忽略这些位置。
        input_ids.extend(segment_ids)
        labels.extend(segment_ids if segment.supervise else [-100] * len(segment_ids))
        thinking_mode_ids.extend([thinking_mode_to_id(segment.thinking_mode)] * len(segment_ids))
        token_channel_ids.extend([target_channel_to_id(segment.target_channel)] * len(segment_ids))
    target_channel_ids = _build_prediction_channel_ids(token_channel_ids)
    return input_ids, labels, thinking_mode_ids, target_channel_ids


def _build_prediction_channel_ids(token_channel_ids):
    """把当前 token 通道转换成预测下一个 token 时使用的目标通道。"""
    if not token_channel_ids:
        return []
    if len(token_channel_ids) == 1:
        return [token_channel_ids[0]]
    return list(token_channel_ids[1:]) + [token_channel_ids[-1]]


def _truncate_sequence(input_ids, labels, thinking_mode_ids, target_channel_ids, max_length):
    """按最大长度同步截断 input_ids、labels 和 thinking 控制张量。"""
    if max_length is None or len(input_ids) <= max_length:
        return input_ids, labels, thinking_mode_ids, target_channel_ids
    return (
        input_ids[:max_length],
        labels[:max_length],
        thinking_mode_ids[:max_length],
        target_channel_ids[:max_length],
    )


def encode_training_sample(sample, tokenizer, max_length=None, *, thinking_mode=THINKING_MODE_AUTO):
    """把单条结构化样本编码成训练 token 序列。"""
    if isinstance(sample, Mapping) and sample.get(PRE_TOKENIZED_CHAT_INPUT_IDS) is not None:
        return _encode_pre_tokenized_chat_sample(sample, max_length)
    if isinstance(sample, Mapping) and sample.get(PRE_TOKENIZED_TOKEN_IDS) is not None:
        return _encode_pre_tokenized_text_sample(
            sample,
            sample[PRE_TOKENIZED_TOKEN_IDS],
            max_length,
        )
    rendered_segments = render_training_segments(
        sample,
        template_version=GlobalConfig.chat_template_version,
        thinking_mode=thinking_mode,
    )
    input_ids, labels, thinking_mode_ids, target_channel_ids = _tokenize_rendered_segments(
        rendered_segments,
        tokenizer,
    )
    input_ids, labels, thinking_mode_ids, target_channel_ids = _truncate_sequence(
        input_ids,
        labels,
        thinking_mode_ids,
        target_channel_ids,
        max_length,
    )

    sample_id = str(sample.get("id", "<unknown>"))
    if len(input_ids) < 2:
        raise ValueError(f"样本 {sample_id} 在截断后不足 2 个 token，无法参与训练。")
    return EncodedTrainingSample(
        sample_id=sample_id,
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        thinking_mode_ids=tuple(thinking_mode_ids),
        target_channel_ids=tuple(target_channel_ids),
    )


def _encode_pre_tokenized_chat_sample(sample, max_length):
    """读取 cache 中已对齐的监督序列，只进行训练窗口截断。"""
    if sample.get("type") != "chat":
        raise ValueError("预编码 chat 样本类型必须为 chat。")
    keys = (
        PRE_TOKENIZED_CHAT_INPUT_IDS,
        PRE_TOKENIZED_CHAT_LABELS,
        PRE_TOKENIZED_CHAT_THINKING_MODE_IDS,
        PRE_TOKENIZED_CHAT_TARGET_CHANNEL_IDS,
    )
    values = [sample[key] for key in keys]
    if max_length is not None:
        max_length = int(max_length)
        if max_length <= 0:
            raise ValueError("max_length 必须是正整数。")
        values = [value[:max_length] for value in values]
    lengths = [len(value) for value in values]
    sample_id = str(sample.get("id", "<unknown>"))
    if len(set(lengths)) != 1:
        raise ValueError(f"样本 {sample_id} 的预编码 chat 序列长度不一致。")
    if lengths[0] < 2:
        raise ValueError(f"样本 {sample_id} 在截断后不足 2 个 token，无法参与训练。")
    return EncodedTrainingSample(
        sample_id=sample_id,
        input_ids=values[0],
        labels=values[1],
        thinking_mode_ids=values[2],
        target_channel_ids=values[3],
    )


def _encode_pre_tokenized_text_sample(sample, token_ids, max_length):
    """把离线 cache 的预编码 text 样本直接转成训练序列。

    cache 构建时已经按 text 模板渲染语义编码：正文 token 后追加 eos，
    正文与 eos 都是全监督 answer 通道、thinking=off。因此这里不再调用
    tokenizer 和模板，只做长度截断和 label/控制张量填充，等价于
    :func:`encode_training_sample` 的 text 分支。
    """
    if sample.get("type") != "text":
        raise ValueError("预编码 token 样本只支持 text 类型。")
    sample_id = str(sample.get("id", "<unknown>"))
    thinking_mode_id = thinking_mode_to_id(THINKING_MODE_OFF)
    answer_channel_id = target_channel_to_id(TARGET_CHANNEL_ANSWER)
    if isinstance(token_ids, np.ndarray):
        # 先截断 mmap/数组视图再转换 dtype，避免旧 cache 中超长记录产生
        # 不会参与训练的整条 int64 临时副本。
        token_array = np.asarray(token_ids)
        if max_length is not None:
            max_length = int(max_length)
            if max_length <= 0:
                raise ValueError("max_length 必须是正整数。")
            token_array = token_array[:max_length]
        token_array = token_array.astype(np.int64, copy=False)
        if token_array.size < 2:
            raise ValueError(
                f"样本 {sample_id} 预编码后不足 2 个 token，无法参与训练。"
            )
        return EncodedTrainingSample(
            sample_id=sample_id,
            input_ids=token_array,
            labels=token_array,
            thinking_mode_ids=np.full(
                token_array.shape,
                thinking_mode_id,
                dtype=np.int64,
            ),
            target_channel_ids=np.full(
                token_array.shape,
                answer_channel_id,
                dtype=np.int64,
            ),
        )
    # 历史 tuple 路径：只出现在测试注入或旧缓存格式中。
    token_ids = tuple(int(value) for value in token_ids)
    if max_length is not None:
        max_length = int(max_length)
        if max_length <= 0:
            raise ValueError("max_length 必须是正整数。")
        token_ids = token_ids[:max_length]
    if len(token_ids) < 2:
        raise ValueError(
            f"样本 {sample_id} 预编码后不足 2 个 token，无法参与训练。"
        )
    return EncodedTrainingSample(
        sample_id=sample_id,
        input_ids=token_ids,
        labels=token_ids,
        thinking_mode_ids=tuple(thinking_mode_id for _ in token_ids),
        target_channel_ids=tuple(answer_channel_id for _ in token_ids),
    )


def estimate_training_sample_length(
    sample,
    tokenizer,
    max_length=None,
    *,
    thinking_mode=THINKING_MODE_AUTO,
):
    """估计结构化样本经过当前模板和 tokenizer 后的 token 长度。

    该函数只用于建立长度计划或离线 token cache，不创建训练张量。长度计算
    与 :func:`encode_training_sample` 使用完全相同的模板片段和截断规则，因而
    不会因为估计器和实际 collator 不一致而产生错误分桶。
    """
    rendered_segments = render_training_segments(
        sample,
        template_version=GlobalConfig.chat_template_version,
        thinking_mode=thinking_mode,
    )
    token_count = 0
    for segment in rendered_segments:
        token_count += len(
            tokenizer(segment.text, add_special_tokens=False)["input_ids"]
        )
    if max_length is not None:
        max_length = int(max_length)
        if max_length <= 0:
            raise ValueError("max_length 必须是正整数。")
        token_count = min(token_count, max_length)
    return token_count


_TOKEN_LENGTH_FIELD_NAMES = (
    "token_length",
    "token_count",
    "num_tokens",
    "input_length",
    "sequence_length",
)


def _coerce_token_length(value, *, sample_index):
    """把长度元数据转换成严格的正整数。"""
    integer_value = int(value)
    if integer_value <= 0:
        raise ValueError(f"样本 {sample_index} 的 token 长度必须为正整数: {value!r}。")
    return integer_value


def _infer_token_length(sample, *, sample_index):
    """从编码对象或结构化样本的常见字段推断 token 长度。"""
    if isinstance(sample, EncodedTrainingSample):
        return sample.length
    if hasattr(sample, "input_ids"):
        input_ids = getattr(sample, "input_ids")
        if input_ids is not None:
            return len(input_ids)
    for attribute_name in ("token_length", "token_count", "num_tokens"):
        if hasattr(sample, attribute_name):
            return getattr(sample, attribute_name)
    if isinstance(sample, Mapping):
        for token_field_name in ("input_ids", "tokens", "token_ids"):
            if token_field_name in sample and sample[token_field_name] is not None:
                return len(sample[token_field_name])
        for field_name in _TOKEN_LENGTH_FIELD_NAMES:
            if field_name in sample:
                return sample[field_name]
    raise ValueError(
        f"无法推断样本 {sample_index} 的 token 长度；请传入 lengths、"
        "length_getter 或 length_estimator。"
    )


def _resolve_token_lengths(
    samples,
    *,
    lengths=None,
    token_lengths=None,
    length_getter: Callable | None = None,
    length_estimator: Callable | None = None,
    max_length=None,
    clip_to_max_length=False,
):
    """解析长度来源并执行统一边界校验。"""
    if lengths is not None and token_lengths is not None:
        raise ValueError("lengths 与 token_lengths 只能提供一个。")
    if lengths is None:
        lengths = token_lengths
    if length_getter is not None and length_estimator is not None:
        raise ValueError("length_getter 与 length_estimator 只能提供一个。")
    if length_getter is None:
        length_getter = length_estimator

    sample_count = len(samples)
    if lengths is not None:
        raw_lengths = tuple(lengths)
        if len(raw_lengths) != sample_count:
            raise ValueError(
                "显式 token 长度数量与样本数量不一致: "
                f"{len(raw_lengths)} != {sample_count}。"
            )
    elif length_getter is not None:
        raw_lengths = tuple(length_getter(sample) for sample in samples)
    else:
        raw_lengths = tuple(
            _infer_token_length(sample, sample_index=index)
            for index, sample in enumerate(samples)
        )

    normalized_max_length = None
    if max_length is not None:
        normalized_max_length = int(max_length)
        if normalized_max_length <= 0:
            raise ValueError("max_length 必须是正整数。")

    normalized_lengths = []
    for index, raw_length in enumerate(raw_lengths):
        normalized_length = _coerce_token_length(raw_length, sample_index=index)
        if normalized_max_length is not None and normalized_length > normalized_max_length:
            if clip_to_max_length:
                normalized_length = normalized_max_length
            else:
                raise ValueError(
                    f"样本 {index} 的 token 长度 {normalized_length} 超过 max_length "
                    f"{normalized_max_length}；如果长度是截断前估计值，请设置 "
                    "clip_to_max_length=True。"
                )
        normalized_lengths.append(normalized_length)
    return tuple(normalized_lengths)


def _normalise_length_strategy(strategy):
    """规范化长度计划策略名称。"""
    normalized = "sequential" if strategy is None else str(strategy).strip().lower()
    aliases = {
        "none": "sequential",
        "keep": "sequential",
        "stable": "sort",
        "stable_sort": "sort",
        "length_sort": "sort",
        "length_bucket": "bucket",
        "bucket_sort": "bucket",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"sequential", "sort", "bucket", "first_fit", "packing"}:
        raise ValueError(
            "strategy 必须是 sequential、sort、bucket、first_fit 或 packing。"
        )
    return normalized


def _normalise_packing_strategy(packing_strategy):
    """规范化固定 batch 与 packing 装箱策略名称。"""
    normalized = "none" if packing_strategy is None else str(packing_strategy).strip().lower()
    aliases = {"fixed": "none", "next-fit": "sequential", "next_fit": "sequential"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"none", "sequential", "first_fit"}:
        raise ValueError("packing_strategy 必须是 none、sequential 或 first_fit。")
    return normalized


def _derive_bucket_boundaries(lengths, max_length, bucket_count):
    """根据最大长度生成稳定的等宽上界。"""
    if bucket_count <= 0:
        raise ValueError("bucket_count 必须是正整数。")
    upper_bound = int(max_length) if max_length is not None else max(lengths, default=1)
    if upper_bound <= 0:
        raise ValueError("无法从空或非正长度集合生成分桶边界。")
    step = max(1, int(math.ceil(upper_bound / bucket_count)))
    boundaries = []
    for bucket_index in range(1, bucket_count + 1):
        boundary = min(upper_bound, step * bucket_index)
        if not boundaries or boundary > boundaries[-1]:
            boundaries.append(boundary)
        if boundary >= upper_bound:
            break
    return tuple(boundaries)


def _validate_bucket_boundaries(bucket_boundaries, max_length):
    """校验分桶上界必须严格递增且不超过训练长度上限。"""
    normalized = tuple(int(value) for value in bucket_boundaries)
    if any(value <= 0 for value in normalized):
        raise ValueError("bucket_boundaries 中的值必须是正整数。")
    if any(left >= right for left, right in zip(normalized, normalized[1:])):
        raise ValueError("bucket_boundaries 必须严格递增。")
    if max_length is not None and normalized and normalized[-1] > int(max_length):
        raise ValueError("bucket_boundaries 不能超过 max_length。")
    return normalized


def _first_fit_length_groups(indices, lengths, max_length, max_samples_per_batch):
    """按 token 容量执行稳定 first-fit 装箱。

    每个新样本按输入顺序尝试已有箱子，放入第一个仍有容量且未超过样本数
    上限的箱子；因此同一输入和长度序列始终得到相同分组。该策略与传统
    ``build_packed_training_batch`` 的 next-fit 默认路径互不影响。
    """
    groups = []
    group_lengths = []
    for index in indices:
        sample_length = lengths[index]
        if sample_length > max_length:
            raise ValueError(
                f"样本 {index} 的 token 长度 {sample_length} 超过 packing 上限 "
                f"{max_length}。"
            )
        placed = False
        for group_index, (group, group_token_count) in enumerate(
            zip(groups, group_lengths)
        ):
            if (
                len(group) < max_samples_per_batch
                and group_token_count + sample_length <= max_length
            ):
                group.append(index)
                group_lengths[group_index] += sample_length
                placed = True
                break
        if not placed:
            groups.append([index])
            group_lengths.append(sample_length)
    return groups


def build_length_bucket_batch_plan(
    samples,
    batch_size,
    *,
    lengths=None,
    token_lengths=None,
    length_getter: Callable | None = None,
    length_estimator: Callable | None = None,
    max_length=None,
    bucket_boundaries=None,
    bucket_count=8,
    strategy="sequential",
    packing_strategy="none",
    seed=0,
    epoch=0,
    shuffle=False,
    drop_last=False,
    sort_within_bucket=True,
    clip_to_max_length=False,
):
    """构建稳定、可重放的长度分桶 batch 计划。

    参数说明：

    * ``lengths``/``token_lengths`` 提供已经计算好的 token 长度；也可以通过
      ``length_getter`` 或 ``length_estimator`` 对每个样本计算一次。
    * ``strategy="sequential"`` 是默认行为，只按原始顺序切成固定大小 batch，
      因而不会改变现有训练行为；``sort`` 做稳定全局长度排序，``bucket`` 按
      ``bucket_boundaries`` 分桶后分别切 batch。
    * ``packing_strategy="sequential"`` 保持现有 next-fit 装箱；
      ``packing_strategy="first_fit"`` 会寻找第一个有剩余容量的箱子。两者都
      使用 ``batch_size`` 作为每箱最大样本数，并要求提供 ``max_length``。
      另外，``strategy="first_fit"``/``"packing"`` 是设置 first-fit 的便捷别名。
    * ``shuffle`` 只使用本地 ``random.Random``，并由 ``seed`` 与 ``epoch`` 决定，
      不会修改全局随机状态。计划保存原始下标，样本对象不会被复制。
    """
    if not hasattr(samples, "__len__") or not hasattr(samples, "__getitem__"):
        samples = tuple(samples)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size 必须是正整数。")
    try:
        seed = int(seed)
        epoch = int(epoch)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("seed 和 epoch 必须是整数。") from exc

    normalized_lengths = _resolve_token_lengths(
        samples,
        lengths=lengths,
        token_lengths=token_lengths,
        length_getter=length_getter,
        length_estimator=length_estimator,
        max_length=max_length,
        clip_to_max_length=clip_to_max_length,
    )
    requested_strategy = _normalise_length_strategy(strategy)
    normalized_strategy = requested_strategy
    normalized_packing = _normalise_packing_strategy(packing_strategy)
    if normalized_strategy in {"first_fit", "packing"}:
        normalized_strategy = "sequential"
        if normalized_packing not in {"none", "first_fit"}:
            raise ValueError(
                "strategy 使用 first_fit/packing 时不能同时指定不同的 packing_strategy。"
            )
        normalized_packing = "first_fit"

    normalized_max_length = None if max_length is None else int(max_length)
    if normalized_packing != "none" and normalized_max_length is None:
        raise ValueError("启用 packing_strategy 时必须提供 max_length。")

    if normalized_strategy == "bucket":
        if bucket_boundaries is None:
            normalized_boundaries = _derive_bucket_boundaries(
                normalized_lengths,
                normalized_max_length,
                int(bucket_count),
            )
        else:
            normalized_boundaries = _validate_bucket_boundaries(
                bucket_boundaries,
                normalized_max_length,
            )
    else:
        normalized_boundaries = ()

    rng = random.Random((seed * 1_000_003 + epoch) & ((1 << 64) - 1))
    all_indices = list(range(len(samples)))
    bucket_groups = []
    if normalized_strategy == "sequential":
        if shuffle:
            rng.shuffle(all_indices)
        bucket_groups = [(all_indices, -1)]
    elif normalized_strategy == "sort":
        if shuffle:
            # 仅打乱相同长度的 tie，仍保持整体按长度升序。
            tie_keys = {index: rng.random() for index in all_indices}
            all_indices.sort(key=lambda index: (normalized_lengths[index], tie_keys[index]))
        else:
            all_indices.sort(key=lambda index: (normalized_lengths[index], index))
        bucket_groups = [(all_indices, -1)]
    else:  # bucket
        grouped = [[] for _ in range(len(normalized_boundaries) + 1)]
        for index in all_indices:
            bucket_id = bisect_left(normalized_boundaries, normalized_lengths[index])
            grouped[bucket_id].append(index)
        for bucket_id, group in enumerate(grouped):
            if not group:
                continue
            if sort_within_bucket:
                group.sort(key=lambda index: (normalized_lengths[index], index))
            if shuffle:
                rng.shuffle(group)
            bucket_groups.append((group, bucket_id))

    planned_groups = []
    for group, bucket_id in bucket_groups:
        if normalized_packing == "first_fit":
            planned_groups.extend(
                (indices, bucket_id)
                for indices in _first_fit_length_groups(
                    group,
                    normalized_lengths,
                    normalized_max_length,
                    batch_size,
                )
            )
        elif normalized_packing == "sequential":
            if normalized_max_length is None:
                raise ValueError("sequential packing 要求 max_length。")
            # next-fit 与现有 _pack_encoded_samples 的顺序语义一致。
            current = []
            current_length = 0
            for index in group:
                sample_length = normalized_lengths[index]
                if sample_length > normalized_max_length:
                    raise ValueError(
                        f"样本 {index} 的 token 长度 {sample_length} 超过 packing 上限 "
                        f"{normalized_max_length}。"
                    )
                if current and (
                    len(current) >= batch_size
                    or current_length + sample_length > normalized_max_length
                ):
                    planned_groups.append((current, bucket_id))
                    current = []
                    current_length = 0
                current.append(index)
                current_length += sample_length
            if current:
                planned_groups.append((current, bucket_id))
        else:
            for start in range(0, len(group), batch_size):
                chunk = group[start : start + batch_size]
                if len(chunk) < batch_size and drop_last:
                    continue
                planned_groups.append((chunk, bucket_id))

    batches = tuple(
        LengthBucketBatch(
            indices=tuple(indices),
            lengths=tuple(normalized_lengths[index] for index in indices),
            bucket_id=int(bucket_id),
        )
        for indices, bucket_id in planned_groups
    )
    return LengthBucketBatchPlan(
        batches=batches,
        sample_lengths=normalized_lengths,
        strategy=requested_strategy,
        seed=seed,
        epoch=epoch,
        batch_size=batch_size,
        drop_last=bool(drop_last),
        bucket_boundaries=normalized_boundaries,
        packing_strategy=normalized_packing,
    )


def _pad_batch(
    batch_input_ids,
    batch_labels,
    batch_thinking_mode_ids,
    batch_target_channel_ids,
    pad_token_id,
    pad_to_multiple_of=None,
):
    """把普通样本 batch padding 成 input_ids/labels/attention_mask 和 thinking 控制张量。"""
    max_sequence_length = max(len(sequence) for sequence in batch_input_ids)
    if pad_to_multiple_of:
        remainder = max_sequence_length % pad_to_multiple_of
        if remainder:
            max_sequence_length += pad_to_multiple_of - remainder

    batch_size = len(batch_input_ids)
    input_ids = torch.full((batch_size, max_sequence_length), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_sequence_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_sequence_length), dtype=torch.long)
    thinking_mode_ids = torch.zeros((batch_size, max_sequence_length), dtype=torch.long)
    target_channel_ids = torch.full(
        (batch_size, max_sequence_length),
        target_channel_to_id(TARGET_CHANNEL_PROMPT),
        dtype=torch.long,
    )

    for row_index, (
        sequence,
        sequence_labels,
        sequence_thinking_mode_ids,
        sequence_target_channel_ids,
    ) in enumerate(
        zip(batch_input_ids, batch_labels, batch_thinking_mode_ids, batch_target_channel_ids)
    ):
        sequence_length = len(sequence)
        input_ids[row_index, :sequence_length] = (
            torch.from_numpy(sequence) if isinstance(sequence, np.ndarray) else torch.tensor(sequence, dtype=torch.long)
        )
        labels[row_index, :sequence_length] = (
            torch.from_numpy(sequence_labels) if isinstance(sequence_labels, np.ndarray) else torch.tensor(sequence_labels, dtype=torch.long)
        )
        attention_mask[row_index, :sequence_length] = 1
        thinking_mode_ids[row_index, :sequence_length] = (
            torch.from_numpy(sequence_thinking_mode_ids) if isinstance(sequence_thinking_mode_ids, np.ndarray) else torch.tensor(sequence_thinking_mode_ids, dtype=torch.long)
        )
        target_channel_ids[row_index, :sequence_length] = (
            torch.from_numpy(sequence_target_channel_ids) if isinstance(sequence_target_channel_ids, np.ndarray) else torch.tensor(sequence_target_channel_ids, dtype=torch.long)
        )
    return input_ids, labels, attention_mask, thinking_mode_ids, target_channel_ids


def _build_packed_sequence(encoded_samples):
    """把多个样本串接成一条 packed row，并保留样本内位置和分段编号。"""
    if encoded_samples and isinstance(encoded_samples[0].input_ids, np.ndarray):
        # numpy 直通：concatenate/arange/full 都在 C 层执行，避免逐 token
        # Python 循环；cache 数据集内所有样本类型一致，按首样本判断即可。
        packed_input_ids = np.concatenate(
            [encoded_sample.input_ids for encoded_sample in encoded_samples],
        )
        packed_labels = np.concatenate(
            [encoded_sample.labels for encoded_sample in encoded_samples],
        )
        packed_thinking_mode_ids = np.concatenate(
            [
                encoded_sample.thinking_mode_ids
                for encoded_sample in encoded_samples
            ],
        )
        packed_target_channel_ids = np.concatenate(
            [
                encoded_sample.target_channel_ids
                for encoded_sample in encoded_samples
            ],
        )
        packed_position_ids = np.concatenate(
            [
                np.arange(encoded_sample.length, dtype=np.int64)
                for encoded_sample in encoded_samples
            ],
        )
        packed_segment_ids = np.concatenate(
            [
                np.full(
                    encoded_sample.length,
                    segment_index,
                    dtype=np.int64,
                )
                for segment_index, encoded_sample in enumerate(
                    encoded_samples,
                    start=1,
                )
            ],
        )
        return PackedTrainingSequence(
            input_ids=packed_input_ids,
            labels=packed_labels,
            thinking_mode_ids=packed_thinking_mode_ids,
            target_channel_ids=packed_target_channel_ids,
            position_ids=packed_position_ids,
            segment_ids=packed_segment_ids,
        )

    packed_input_ids = []
    packed_labels = []
    packed_thinking_mode_ids = []
    packed_target_channel_ids = []
    packed_position_ids = []
    packed_segment_ids = []

    for segment_index, encoded_sample in enumerate(encoded_samples, start=1):
        # position_ids 对每个原始样本从 0 重启，配合 mixed LongRoPE2 保持样本内位置语义。
        packed_input_ids.extend(encoded_sample.input_ids)
        packed_labels.extend(encoded_sample.labels)
        packed_thinking_mode_ids.extend(encoded_sample.thinking_mode_ids)
        packed_target_channel_ids.extend(encoded_sample.target_channel_ids)
        packed_position_ids.extend(range(encoded_sample.length))
        packed_segment_ids.extend([segment_index] * encoded_sample.length)

    return PackedTrainingSequence(
        input_ids=tuple(packed_input_ids),
        labels=tuple(packed_labels),
        thinking_mode_ids=tuple(packed_thinking_mode_ids),
        target_channel_ids=tuple(packed_target_channel_ids),
        position_ids=tuple(packed_position_ids),
        segment_ids=tuple(packed_segment_ids),
    )


def _pack_encoded_samples(encoded_samples, max_length, *, packing_strategy="sequential"):
    """把编码样本装箱到不超过 ``max_length`` 的 packed row。

    默认 ``sequential`` 保留原有 next-fit 行为：只尝试当前 row，遇到放不下
    的样本就开启下一行。显式指定 ``first_fit`` 时才会搜索之前的 row，以提高
    不同长度样本混合时的容量利用率。
    """
    if max_length is None or int(max_length) <= 0:
        raise ValueError("sequence packing 要求 max_length 为正整数。")
    normalized_strategy = _normalise_packing_strategy(packing_strategy)
    if normalized_strategy == "none":
        normalized_strategy = "sequential"

    max_length = int(max_length)
    if normalized_strategy == "first_fit":
        encoded_lengths = tuple(encoded_sample.length for encoded_sample in encoded_samples)
        groups = _first_fit_length_groups(
            list(range(len(encoded_samples))),
            encoded_lengths,
            max_length,
            len(encoded_samples) if encoded_samples else 1,
        )
        return [
            _build_packed_sequence([encoded_samples[index] for index in indices])
            for indices in groups
        ]

    packed_sequences = []
    current_bucket = []
    current_length = 0
    for encoded_sample in encoded_samples:
        if encoded_sample.length > max_length:
            raise ValueError(
                f"样本 {encoded_sample.sample_id} 长度 {encoded_sample.length} 超过 "
                f"packing 上限 {max_length}。"
            )
        projected_length = current_length + encoded_sample.length
        if current_bucket and projected_length > max_length:
            packed_sequences.append(_build_packed_sequence(current_bucket))
            current_bucket = []
            current_length = 0
        current_bucket.append(encoded_sample)
        current_length += encoded_sample.length

    if current_bucket:
        packed_sequences.append(_build_packed_sequence(current_bucket))
    return packed_sequences


def _pad_packed_batch(packed_sequences, pad_token_id, pad_to_multiple_of=None):
    """padding packed row，同时输出 position_ids 和 segment_ids。"""
    max_sequence_length = max(sequence.length for sequence in packed_sequences)
    if pad_to_multiple_of:
        remainder = max_sequence_length % pad_to_multiple_of
        if remainder:
            max_sequence_length += pad_to_multiple_of - remainder

    batch_size = len(packed_sequences)
    input_ids = torch.full((batch_size, max_sequence_length), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_sequence_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_sequence_length), dtype=torch.long)
    thinking_mode_ids = torch.zeros((batch_size, max_sequence_length), dtype=torch.long)
    target_channel_ids = torch.full(
        (batch_size, max_sequence_length),
        target_channel_to_id(TARGET_CHANNEL_PROMPT),
        dtype=torch.long,
    )
    position_ids = torch.zeros((batch_size, max_sequence_length), dtype=torch.long)
    segment_ids = torch.zeros((batch_size, max_sequence_length), dtype=torch.long)

    for row_index, packed_sequence in enumerate(packed_sequences):
        sequence_length = packed_sequence.length
        input_ids[row_index, :sequence_length] = (
            torch.from_numpy(packed_sequence.input_ids) if isinstance(packed_sequence.input_ids, np.ndarray) else torch.tensor(packed_sequence.input_ids, dtype=torch.long)
        )
        labels[row_index, :sequence_length] = (
            torch.from_numpy(packed_sequence.labels) if isinstance(packed_sequence.labels, np.ndarray) else torch.tensor(packed_sequence.labels, dtype=torch.long)
        )
        attention_mask[row_index, :sequence_length] = 1
        thinking_mode_ids[row_index, :sequence_length] = (
            torch.from_numpy(packed_sequence.thinking_mode_ids) if isinstance(packed_sequence.thinking_mode_ids, np.ndarray) else torch.tensor(packed_sequence.thinking_mode_ids, dtype=torch.long)
        )
        target_channel_ids[row_index, :sequence_length] = (
            torch.from_numpy(packed_sequence.target_channel_ids) if isinstance(packed_sequence.target_channel_ids, np.ndarray) else torch.tensor(packed_sequence.target_channel_ids, dtype=torch.long)
        )
        position_ids[row_index, :sequence_length] = (
            torch.from_numpy(packed_sequence.position_ids) if isinstance(packed_sequence.position_ids, np.ndarray) else torch.tensor(packed_sequence.position_ids, dtype=torch.long)
        )
        segment_ids[row_index, :sequence_length] = (
            torch.from_numpy(packed_sequence.segment_ids) if isinstance(packed_sequence.segment_ids, np.ndarray) else torch.tensor(packed_sequence.segment_ids, dtype=torch.long)
        )
    return input_ids, labels, attention_mask, thinking_mode_ids, target_channel_ids, position_ids, segment_ids


def build_training_batch(
    samples,
    tokenizer,
    max_length=None,
    *,
    thinking_mode=THINKING_MODE_AUTO,
    include_thinking_tensors=False,
):
    """把一批结构化样本编码成训练张量。"""
    batch_input_ids = []
    batch_labels = []
    batch_thinking_mode_ids = []
    batch_target_channel_ids = []
    for sample in samples:
        encoded_sample = encode_training_sample(
            sample,
            tokenizer,
            max_length=max_length,
            thinking_mode=thinking_mode,
        )
        batch_input_ids.append(encoded_sample.input_ids)
        batch_labels.append(encoded_sample.labels)
        batch_thinking_mode_ids.append(encoded_sample.thinking_mode_ids)
        batch_target_channel_ids.append(encoded_sample.target_channel_ids)

    pad_to_multiple_of = GlobalConfig.pad_to_multiple_of if GlobalConfig.device.type == "cuda" else None
    padded = _pad_batch(
        batch_input_ids,
        batch_labels,
        batch_thinking_mode_ids,
        batch_target_channel_ids,
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    if include_thinking_tensors:
        return padded
    input_ids, labels, attention_mask, _thinking_mode_ids, _target_channel_ids = padded
    return input_ids, labels, attention_mask


def build_packed_training_batch(
    samples,
    tokenizer,
    max_length,
    *,
    thinking_mode=THINKING_MODE_AUTO,
    include_thinking_tensors=False,
    packing_strategy="sequential",
):
    """把一批样本按 token 序列打包成 packed row 训练张量。

    ``packing_strategy`` 默认值保持历史顺序 next-fit 行为；需要减少碎片时
    可显式传入 ``"first_fit"``。该参数只影响 packed row 的装箱，不改变样本
    的 token 内容、标签或 segment/position 语义。
    """
    encoded_samples = [
        encode_training_sample(
            sample,
            tokenizer,
            max_length=max_length,
            thinking_mode=thinking_mode,
        )
        for sample in samples
    ]
    packed_sequences = _pack_encoded_samples(
        encoded_samples,
        max_length=max_length,
        packing_strategy=packing_strategy,
    )
    pad_to_multiple_of = GlobalConfig.pad_to_multiple_of if GlobalConfig.device.type == "cuda" else None
    (
        input_ids,
        labels,
        attention_mask,
        thinking_mode_ids,
        target_channel_ids,
        position_ids,
        segment_ids,
    ) = _pad_packed_batch(
        packed_sequences,
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    # 返回 len(encoded_samples) 而不是 len(packed_sequences)，训练日志中 sample_count
    # 仍表示原始样本数，便于和非 packing 路径横向对比。
    if include_thinking_tensors:
        return (
            input_ids,
            labels,
            attention_mask,
            thinking_mode_ids,
            target_channel_ids,
            position_ids,
            segment_ids,
            len(encoded_samples),
        )
    return input_ids, labels, attention_mask, position_ids, segment_ids, len(encoded_samples)
