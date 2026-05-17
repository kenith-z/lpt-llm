"""LPT v2 结构化训练样本批处理工具。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lpt_config import GlobalConfig
from lpt_protocol import (
    DS_BOS_TOKEN,
    DS_EOS_TOKEN,
    DS_PAD_TOKEN,
    get_template_spec,
    render_training_segments,
)


REQUIRED_TOKENIZER_TOKENS = (
    DS_BOS_TOKEN,
    DS_EOS_TOKEN,
    DS_PAD_TOKEN,
)


@dataclass(frozen=True)
class EncodedTrainingSample:
    """单条训练样本编码后的 token 结果。"""

    sample_id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]

    @property
    def length(self):
        """返回编码后 token 长度，packing 时以它作为容量单位。"""
        return len(self.input_ids)


@dataclass(frozen=True)
class PackedTrainingSequence:
    """sequence packing 后的一条训练序列。"""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    position_ids: tuple[int, ...]
    segment_ids: tuple[int, ...]

    @property
    def length(self):
        """返回 packed row 的真实 token 长度，不含 padding。"""
        return len(self.input_ids)


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
    """把模板渲染片段编码成 input_ids，并按 supervise 标志构造 labels。"""
    input_ids = []
    labels = []
    for segment in segments:
        encoded = tokenizer(segment.text, add_special_tokens=False)
        segment_ids = encoded["input_ids"]
        if not segment_ids:
            continue
        # 用户提示、系统提示等非监督片段写入 -100，让 cross entropy 忽略这些位置。
        input_ids.extend(segment_ids)
        labels.extend(segment_ids if segment.supervise else [-100] * len(segment_ids))
    return input_ids, labels


def _truncate_sequence(input_ids, labels, max_length):
    """按最大长度同步截断 input_ids 与 labels。"""
    if max_length is None or len(input_ids) <= max_length:
        return input_ids, labels
    return input_ids[:max_length], labels[:max_length]


def encode_training_sample(sample, tokenizer, max_length=None):
    """把单条结构化样本编码成训练 token 序列。"""
    rendered_segments = render_training_segments(
        sample,
        template_version=GlobalConfig.chat_template_version,
    )
    input_ids, labels = _tokenize_rendered_segments(rendered_segments, tokenizer)
    input_ids, labels = _truncate_sequence(input_ids, labels, max_length)

    sample_id = str(sample.get("id", "<unknown>"))
    if len(input_ids) < 2:
        raise ValueError(f"样本 {sample_id} 在截断后不足 2 个 token，无法参与训练。")
    return EncodedTrainingSample(
        sample_id=sample_id,
        input_ids=tuple(input_ids),
        labels=tuple(labels),
    )


def _pad_batch(batch_input_ids, batch_labels, pad_token_id, pad_to_multiple_of=None):
    """把普通样本 batch padding 成 input_ids/labels/attention_mask。"""
    max_sequence_length = max(len(sequence) for sequence in batch_input_ids)
    if pad_to_multiple_of:
        remainder = max_sequence_length % pad_to_multiple_of
        if remainder:
            max_sequence_length += pad_to_multiple_of - remainder

    batch_size = len(batch_input_ids)
    input_ids = torch.full((batch_size, max_sequence_length), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_sequence_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_sequence_length), dtype=torch.long)

    for row_index, (sequence, sequence_labels) in enumerate(zip(batch_input_ids, batch_labels)):
        sequence_length = len(sequence)
        input_ids[row_index, :sequence_length] = torch.tensor(sequence, dtype=torch.long)
        labels[row_index, :sequence_length] = torch.tensor(sequence_labels, dtype=torch.long)
        attention_mask[row_index, :sequence_length] = 1
    return input_ids, labels, attention_mask


def _build_packed_sequence(encoded_samples):
    """把多个样本串接成一条 packed row，并保留样本内位置和分段编号。"""
    packed_input_ids = []
    packed_labels = []
    packed_position_ids = []
    packed_segment_ids = []

    for segment_index, encoded_sample in enumerate(encoded_samples, start=1):
        # position_ids 对每个原始样本从 0 重启，配合 mixed LongRoPE2 保持样本内位置语义。
        packed_input_ids.extend(encoded_sample.input_ids)
        packed_labels.extend(encoded_sample.labels)
        packed_position_ids.extend(range(encoded_sample.length))
        packed_segment_ids.extend([segment_index] * encoded_sample.length)

    return PackedTrainingSequence(
        input_ids=tuple(packed_input_ids),
        labels=tuple(packed_labels),
        position_ids=tuple(packed_position_ids),
        segment_ids=tuple(packed_segment_ids),
    )


def _pack_encoded_samples(encoded_samples, max_length):
    """使用简单 first-fit 顺序装箱，把样本打包到不超过 max_length 的 packed row。"""
    if max_length is None or int(max_length) <= 0:
        raise ValueError("sequence packing 要求 max_length 为正整数。")

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
    position_ids = torch.zeros((batch_size, max_sequence_length), dtype=torch.long)
    segment_ids = torch.zeros((batch_size, max_sequence_length), dtype=torch.long)

    for row_index, packed_sequence in enumerate(packed_sequences):
        sequence_length = packed_sequence.length
        input_ids[row_index, :sequence_length] = torch.tensor(packed_sequence.input_ids, dtype=torch.long)
        labels[row_index, :sequence_length] = torch.tensor(packed_sequence.labels, dtype=torch.long)
        attention_mask[row_index, :sequence_length] = 1
        position_ids[row_index, :sequence_length] = torch.tensor(packed_sequence.position_ids, dtype=torch.long)
        segment_ids[row_index, :sequence_length] = torch.tensor(packed_sequence.segment_ids, dtype=torch.long)
    return input_ids, labels, attention_mask, position_ids, segment_ids


def build_training_batch(samples, tokenizer, max_length=None):
    """把一批结构化样本编码成训练张量。"""
    batch_input_ids = []
    batch_labels = []
    for sample in samples:
        encoded_sample = encode_training_sample(sample, tokenizer, max_length=max_length)
        batch_input_ids.append(encoded_sample.input_ids)
        batch_labels.append(encoded_sample.labels)

    pad_to_multiple_of = GlobalConfig.pad_to_multiple_of if GlobalConfig.device.type == "cuda" else None
    return _pad_batch(
        batch_input_ids,
        batch_labels,
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )


def build_packed_training_batch(samples, tokenizer, max_length):
    """把一批样本按 token 序列打包成 packed row 训练张量。"""
    encoded_samples = [
        encode_training_sample(sample, tokenizer, max_length=max_length)
        for sample in samples
    ]
    packed_sequences = _pack_encoded_samples(encoded_samples, max_length=max_length)
    pad_to_multiple_of = GlobalConfig.pad_to_multiple_of if GlobalConfig.device.type == "cuda" else None
    input_ids, labels, attention_mask, position_ids, segment_ids = _pad_packed_batch(
        packed_sequences,
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    # 返回 len(encoded_samples) 而不是 len(packed_sequences)，训练日志中 sample_count
    # 仍表示原始样本数，便于和非 packing 路径横向对比。
    return input_ids, labels, attention_mask, position_ids, segment_ids, len(encoded_samples)
