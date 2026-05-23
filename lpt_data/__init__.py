"""LPT 结构化数据包。"""

from importlib import import_module

from .schema import normalize_dataset_record


_LAZY_EXPORT_MODULES = {
    "EncodedTrainingSample": ".batching",
    "PackedTrainingSequence": ".batching",
    "build_packed_training_batch": ".batching",
    "build_training_batch": ".batching",
    "encode_training_sample": ".batching",
    "prepare_tokenizer": ".batching",
    "DATA_PROGRESS_RECORD_KEY": ".io",
    "build_streaming_manifest_dataset": ".io",
    "load_dataset_manifest": ".io",
    "load_dataset_records": ".io",
    "summarize_dataset_sources": ".io",
    "summarize_dataset_types": ".io",
}

__all__ = [
    "normalize_dataset_record",
    *_LAZY_EXPORT_MODULES,
]


def __getattr__(name):
    """按需加载依赖 torch 的训练/流式数据工具，保持 schema 导入轻量。"""
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
