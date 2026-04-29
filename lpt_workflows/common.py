"""多训练阶段共享的辅助函数。"""

from dataclasses import dataclass
from pathlib import Path
import random

from transformers import AutoTokenizer

from lpt_config import (
    GenerationConfig,
    GlobalConfig,
    LONGROPE2_EMBEDDING_MODES,
    ModelConfig,
    build_model_config_from_checkpoint,
    load_longrope2_factors_file,
    load_model_config_json,
    model_config_snapshot_path,
)
from lpt_data import (
    build_streaming_manifest_dataset,
    load_dataset_manifest,
    load_dataset_records,
    summarize_dataset_sources,
    summarize_dataset_types,
)
from lpt_runtime import describe_execution_plan, resolve_execution_plan
from lpt_training import load_checkpoint, prepare_tokenizer


TOKENIZER_PATH = Path("./lpt_model/ds_tokenizer")
ARTIFACT_ROOT_DIR = Path("./artifacts/lpt_ds_v1")


@dataclass(frozen=True)
class LongRoPE2WorkflowOptions:
    """工作流层可覆盖的 LongRoPE2 训练策略选项。"""

    train_max_sequence_length: int | None = None
    train_rope_cache_max_sequence_length: int | None = None
    inference_rope_cache_max_sequence_length: int | None = None
    original_window: int | None = None
    target_window: int | None = None
    long_factors_path: Path | None = None
    train_embedding_mode: str | None = None
    inference_embedding_mode: str | None = None
    mixed_original_window: int | None = None
    window_lengths: tuple[int, ...] | None = None
    window_weights: tuple[float, ...] | None = None

    @property
    def has_model_overrides(self):
        return any(
            value is not None
            for value in (
                self.original_window,
                self.target_window,
                self.long_factors_path,
                self.train_embedding_mode,
                self.inference_embedding_mode,
                self.mixed_original_window,
            )
        )

    @property
    def has_training_profile_overrides(self):
        return self.window_lengths is not None or self.window_weights is not None


def _parse_int_tuple(raw_value):
    values = tuple(int(value.strip()) for value in str(raw_value).split(",") if value.strip())
    if not values:
        raise ValueError("至少需要提供一个整数。")
    return values


def _parse_float_tuple(raw_value):
    values = tuple(float(value.strip()) for value in str(raw_value).split(",") if value.strip())
    if not values:
        raise ValueError("至少需要提供一个数值。")
    return values


def add_longrope2_training_arguments(parser):
    """给训练入口注册 LongRoPE2 策略参数。"""
    parser.add_argument(
        "--train-max-sequence-length",
        type=int,
        default=None,
        help="可选：覆盖训练样本截断上限。",
    )
    parser.add_argument(
        "--train-rope-cache-max-sequence-length",
        type=int,
        default=None,
        help="可选：覆盖训练态 RoPE cache 上限。",
    )
    parser.add_argument(
        "--inference-rope-cache-max-sequence-length",
        type=int,
        default=None,
        help="可选：覆盖推理态 RoPE cache 上限。",
    )
    parser.add_argument(
        "--longrope2-original-window",
        type=int,
        default=None,
        help="可选：覆盖 LongRoPE2 原始窗口。",
    )
    parser.add_argument(
        "--longrope2-target-window",
        type=int,
        default=None,
        help="可选：覆盖 LongRoPE2 目标窗口。",
    )
    parser.add_argument(
        "--longrope2-long-factors-path",
        type=Path,
        default=None,
        help="可选：导入 LongRoPE2 搜索因子文件。",
    )
    parser.add_argument(
        "--longrope2-train-embedding-mode",
        choices=LONGROPE2_EMBEDDING_MODES,
        default=None,
        help="可选：训练侧超过原始窗口后的 embedding 模式。",
    )
    parser.add_argument(
        "--longrope2-inference-embedding-mode",
        choices=LONGROPE2_EMBEDDING_MODES,
        default=None,
        help="可选：推理侧超过原始窗口后的 embedding 模式。",
    )
    parser.add_argument(
        "--longrope2-mixed-original-window",
        type=int,
        default=None,
        help="可选：mixed 模式中保留原始 RoPE 的窗口长度。",
    )
    parser.add_argument(
        "--longrope2-window-lengths",
        type=_parse_int_tuple,
        default=None,
        help="可选：训练 batch 混合采样窗口，逗号分隔，例如 2048,7680。",
    )
    parser.add_argument(
        "--longrope2-window-weights",
        type=_parse_float_tuple,
        default=None,
        help="可选：训练 batch 窗口采样权重，逗号分隔。",
    )


def build_longrope2_workflow_options(args):
    """从 argparse 结果构造 LongRoPE2 工作流选项。"""
    return LongRoPE2WorkflowOptions(
        train_max_sequence_length=args.train_max_sequence_length,
        train_rope_cache_max_sequence_length=args.train_rope_cache_max_sequence_length,
        inference_rope_cache_max_sequence_length=args.inference_rope_cache_max_sequence_length,
        original_window=args.longrope2_original_window,
        target_window=args.longrope2_target_window,
        long_factors_path=args.longrope2_long_factors_path,
        train_embedding_mode=args.longrope2_train_embedding_mode,
        inference_embedding_mode=args.longrope2_inference_embedding_mode,
        mixed_original_window=args.longrope2_mixed_original_window,
        window_lengths=args.longrope2_window_lengths,
        window_weights=args.longrope2_window_weights,
    )


def apply_longrope2_runtime_overrides(options: LongRoPE2WorkflowOptions | None):
    """应用只影响本次进程的训练/推理长度覆盖。"""
    if options is None:
        return
    if options.train_max_sequence_length is not None:
        GlobalConfig.train_max_sequence_length = int(options.train_max_sequence_length)
        if options.train_rope_cache_max_sequence_length is None:
            GlobalConfig.train_rope_cache_max_sequence_length = int(options.train_max_sequence_length)
    if options.train_rope_cache_max_sequence_length is not None:
        GlobalConfig.train_rope_cache_max_sequence_length = int(options.train_rope_cache_max_sequence_length)
    if options.inference_rope_cache_max_sequence_length is not None:
        GlobalConfig.inference_rope_cache_max_sequence_length = int(
            options.inference_rope_cache_max_sequence_length
        )


def apply_longrope2_model_config_overrides(model_config, options: LongRoPE2WorkflowOptions | None):
    """把工作流参数合并进 ModelConfig，用于新训练运行。"""
    if options is None or not options.has_model_overrides:
        return model_config

    overrides = {}
    if options.original_window is not None:
        overrides["original_max_len"] = int(options.original_window)
    if options.target_window is not None:
        overrides["longrope2_target_length"] = int(options.target_window)
    if options.long_factors_path is not None:
        overrides["longrope2_long_factors"] = load_longrope2_factors_file(options.long_factors_path)
    if options.train_embedding_mode is not None:
        overrides["longrope2_train_embedding_mode"] = options.train_embedding_mode
    if options.inference_embedding_mode is not None:
        overrides["longrope2_inference_embedding_mode"] = options.inference_embedding_mode
    if options.mixed_original_window is not None:
        overrides["longrope2_mixed_original_window"] = int(options.mixed_original_window)
    return model_config.with_overrides(**overrides)


def build_training_profile_with_longrope2_options(base_profile, options: LongRoPE2WorkflowOptions | None):
    """按命令行覆盖生成临时训练 profile。"""
    if options is None or not options.has_training_profile_overrides:
        return base_profile

    overrides = {}
    if options.window_lengths is not None:
        overrides["longrope2_window_sampling_enabled"] = True
        overrides["longrope2_window_lengths"] = tuple(int(value) for value in options.window_lengths)
    if options.window_weights is not None:
        overrides["longrope2_window_sampling_weights"] = tuple(float(value) for value in options.window_weights)
    return type(
        f"{base_profile.__name__}WithLongRoPE2Options",
        (base_profile,),
        overrides,
    )


def warn_if_longrope2_model_options_ignored(options: LongRoPE2WorkflowOptions | None, reason):
    """提示已有 checkpoint 时不会覆盖模型位置编码配置。"""
    if options is not None and options.has_model_overrides:
        print(f"检测到已有 {reason}，LongRoPE2 模型结构参数以 checkpoint 为准。")


def load_structured_dataset(dataset_path: Path, *, seed=None):
    """加载结构化数据集并做一次随机打乱。"""
    records = load_dataset_records(dataset_path)
    rng = random.Random(seed)
    rng.shuffle(records)
    print("样本类型分布", summarize_dataset_types(records))
    print("样本来源分布", summarize_dataset_sources(records))
    return records


def load_dataset_from_manifest(
    manifest_path: Path,
    *,
    expected_types,
    seed=None,
    shuffle_buffer_size=None,
):
    """从 manifest 加载多语料混合集。"""
    if shuffle_buffer_size is None:
        shuffle_buffer_size = GlobalConfig.dataset_shuffle_buffer_size
    dataset = build_streaming_manifest_dataset(
        manifest_path,
        expected_types=expected_types,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
    )
    print("manifest", manifest_path)
    print("语料清单", dataset.loaded_datasets)
    print("样本类型分布", dataset.summary_types)
    print("样本来源分布", dataset.summary_sources)
    return dataset


def build_local_tokenizer(tokenizer_path: Path = TOKENIZER_PATH):
    """从本地目录加载并校验 tokenizer。"""
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    return prepare_tokenizer(tokenizer)


def load_model_config_from_checkpoint_root(checkpoint_root: Path, *, map_location=None):
    """从 checkpoint 恢复模型结构配置。"""
    checkpoint = load_checkpoint(checkpoint_root, map_location=map_location)
    return build_model_config_from_checkpoint(checkpoint)


def resolve_artifact_model_config(artifact_dir: Path, checkpoint_root: Path | None = None):
    """优先从 artifacts 配置快照恢复模型结构，否则回退到 checkpoint/default。"""
    snapshot_path = model_config_snapshot_path(artifact_dir)
    if snapshot_path.exists():
        return load_model_config_json(snapshot_path)

    if checkpoint_root is not None and checkpoint_root.with_suffix(".pth").exists():
        return load_model_config_from_checkpoint_root(checkpoint_root)

    return ModelConfig()


def resolve_inference_execution_plan(model_config, execution_config=None):
    """按模型结构和执行参数生成推理执行计划。"""
    execution_plan = resolve_execution_plan(
        execution_config,
        num_layers=model_config.num_layers,
    )
    if execution_config is not None and execution_config.print_device_map:
        print(describe_execution_plan(execution_plan))
    return execution_plan


def build_default_generation_config():
    """构造推理阶段使用的默认采样参数。"""
    return GenerationConfig(
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        max_length=1024,
        repetition_penalty=1.2,
    )
