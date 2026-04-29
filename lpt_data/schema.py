"""结构化训练数据 schema 校验。"""

from lpt_protocol import validate_messages


VALID_SAMPLE_TYPES = frozenset({"chat", "text"})


def _normalize_optional_string(record, field_name):
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是字符串。")
    normalized = value.strip()
    return normalized or None


def _normalize_text_record(record):
    text = record.get("text")
    if not isinstance(text, str):
        raise TypeError("text 样本的 text 字段必须是字符串。")
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("text 样本的 text 字段不能为空。")
    normalized_record = dict(record)
    normalized_record["type"] = "text"
    normalized_record["text"] = normalized_text
    return normalized_record


def _normalize_chat_record(record):
    normalized_record = dict(record)
    normalized_record["type"] = "chat"
    normalized_record["messages"] = validate_messages(record.get("messages"))
    return normalized_record


def normalize_dataset_record(record, *, default_id=None):
    """校验并标准化单条数据记录。"""
    if not isinstance(record, dict):
        raise TypeError("数据记录必须是字典。")

    sample_type = record.get("type")
    if sample_type not in VALID_SAMPLE_TYPES:
        raise ValueError(f"不支持的样本类型: {sample_type}")

    normalized_record = (
        _normalize_chat_record(record)
        if sample_type == "chat"
        else _normalize_text_record(record)
    )

    normalized_record["id"] = _normalize_optional_string(record, "id") or default_id
    if not normalized_record["id"]:
        raise ValueError("数据记录缺少可用的 id。")

    for field_name in ("source", "language", "split"):
        normalized_value = _normalize_optional_string(record, field_name)
        if normalized_value is not None:
            normalized_record[field_name] = normalized_value
        else:
            normalized_record.pop(field_name, None)

    return normalized_record

