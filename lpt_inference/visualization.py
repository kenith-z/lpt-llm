"""可视化与调试辅助模块。"""

from pathlib import Path

from PIL import Image
from matplotlib import pyplot as plt

from lpt_config import GlobalConfig


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_ATTENTION_IMAGE_PATTERN = "Time_*.png"
_TOKEN_TABLE_IMAGE = "token_position_table.png"
_TEXT_PREVIEW_IMAGE = "text.png"


def display_checkpoint_summary(checkpoint):
    """以较友好的形式打印检查点中的关键信息。"""
    candidate_fields = [
        ("checkpoint_schema_version", "checkpoint_schema_version"),
        ("model_config_schema_version", "model_config_schema_version"),
        ("model_abbr", "model_abbr"),
        ("model_name_en", "model_name_en"),
        ("model_name_zh", "model_name_zh"),
        ("training_mode", "training_mode"),
        ("training_stage", "training_stage"),
        ("source_manifest", "source_manifest"),
        ("hidden_size", "hidden_size"),
        ("num_heads", "num_heads"),
        ("num_kv_heads", "num_kv_heads"),
        ("head_size", "head_size"),
        ("head_dim", "head_dim"),
        ("num_layers", "num_layers"),
        ("D", "hidden_size"),
        ("h", "num_heads"),
        ("H", "head_size"),
        ("num_blocks", "num_layers"),
        ("tokenizer_category", "tokenizer_category"),
        ("batch_size", "batch_size"),
        ("learning_rate", "learning_rate"),
        ("current_learning_rate", "current_learning_rate"),
        ("warmup_ratio", "warmup_ratio"),
        ("weight_decay", "weight_decay"),
        ("gradient_accumulation_steps", "gradient_accumulation_steps"),
        ("max_grad_norm", "max_grad_norm"),
        ("random_seed", "random_seed"),
        ("loss", "loss"),
        ("latest_eval_loss", "latest_eval_loss"),
        ("latest_eval_ppl", "latest_eval_ppl"),
        ("global_step", "global_step"),
        ("optimizer_step", "optimizer_step"),
        ("tokens_seen", "tokens_seen"),
        ("epoch", "epochs_done"),
        ("total_epochs", "epochs_total"),
    ]
    visible_items = []
    shown_labels = set()
    for key, label in candidate_fields:
        if key in checkpoint and label not in shown_labels:
            visible_items.append((label, checkpoint[key]))
            shown_labels.add(label)
    if not visible_items:
        return

    lines = ["\n已加载模型配置如下(Model configuration loaded):"]
    lines.extend(f"{label}: {value}" for label, value in visible_items)
    print("\n".join(lines))


def ensure_plot_directory(folder):
    """确保可视化输出目录存在。"""
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _attention_image_path(output_dir: Path, sequence_length: int, layer_index: int, head_index: int) -> Path:
    return output_dir / f"Time_{sequence_length}_Layer_{layer_index}_Head_{head_index + 1}.png"


def _first_batch_attention_maps(attention_scores):
    dimensions = tuple(attention_scores.shape)
    if len(dimensions) != 4:
        raise ValueError(f"attention_scores 应为 4 维张量，实际形状为 {dimensions}")

    return attention_scores[0].detach().cpu().numpy()


def _apply_token_axis_ticks(axis, sequence_length: int) -> None:
    token_positions = range(sequence_length)
    axis.set_xticks(token_positions)
    axis.set_yticks(token_positions)
    axis.set_xticklabels(token_positions)
    axis.set_yticklabels(token_positions)


def _save_attention_heatmap(matrix, figure_path: Path, title: str, sequence_length: int, figure_size: int) -> None:
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    axis.imshow(matrix, cmap="viridis")
    axis.set_title(title)
    _apply_token_axis_ticks(axis, sequence_length)
    figure.savefig(figure_path, bbox_inches="tight", dpi=300)
    plt.close(figure)


def plot_attention_scores(attention_scores, layer_index, folder=GlobalConfig.attention_plot_dir):
    """把某一层各注意力头的注意力矩阵保存成图片。"""
    output_dir = ensure_plot_directory(folder)
    head_maps = _first_batch_attention_maps(attention_scores)
    head_count, sequence_length, _ = head_maps.shape

    for head_index, matrix in enumerate(head_maps):
        title = f"Time_{sequence_length}_Layer_{layer_index}_Head_{head_index + 1}"
        _save_attention_heatmap(
            matrix=matrix,
            figure_path=_attention_image_path(output_dir, sequence_length, layer_index, head_index),
            title=title,
            sequence_length=sequence_length,
            figure_size=head_count,
        )


def _token_id_list(token_ids):
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, int):
        return [token_ids]
    return list(token_ids)


def _token_position_rows(tokenizer, token_ids) -> list[list[str]]:
    rows = []
    for position, token_id in enumerate(token_ids):
        token_text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        rows.append([str(position), repr(token_text)])
    return rows


def _save_token_table(rows: list[list[str]], output_path: Path) -> None:
    figure_height = len(rows) * 0.3 + 1
    figure, axis = plt.subplots(figsize=(8, figure_height))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["Position", "Token"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    figure.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(figure)


def _save_text_preview(tokenizer, token_ids, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 2))
    axis.text(
        0.5,
        0.5,
        tokenizer.decode(token_ids, skip_special_tokens=False),
        ha="center",
        va="center",
        fontsize=12,
        wrap=True,
    )
    axis.axis("off")
    figure.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(figure)


def render_token_position_table(tokenizer, token_ids, folder=GlobalConfig.attention_plot_dir):
    """渲染 token 位置表，并与注意力图拼接，方便观察“位置-词元”对应关系。"""
    output_dir = ensure_plot_directory(folder)
    normalized_token_ids = _token_id_list(token_ids)

    _save_token_table(
        rows=_token_position_rows(tokenizer, normalized_token_ids),
        output_path=output_dir / _TOKEN_TABLE_IMAGE,
    )
    _save_text_preview(
        tokenizer=tokenizer,
        token_ids=normalized_token_ids,
        output_path=output_dir / _TEXT_PREVIEW_IMAGE,
    )
    merge_attention_images(output_dir)


def _load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as source:
        return source.convert("RGB")


def _join_images_horizontally(left: Image.Image, right: Image.Image) -> Image.Image:
    canvas_size = (left.width + right.width, max(left.height, right.height))
    canvas = Image.new("RGB", canvas_size, "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def merge_attention_images(folder):
    """把注意力热力图与 token 位置表横向拼接成一张图。"""
    output_dir = Path(folder)
    table_image = _load_rgb_image(output_dir / _TOKEN_TABLE_IMAGE)
    for attention_image_path in sorted(output_dir.glob(_ATTENTION_IMAGE_PATTERN)):
        attention_image = _load_rgb_image(attention_image_path)
        _join_images_horizontally(attention_image, table_image).save(attention_image_path)


def count_model_parameters(model) -> tuple[int, int]:
    """统计模型总参数量与可训练参数量。"""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total_parameters, trainable_parameters


def format_parameter_count(parameter_count: int) -> str:
    """将参数量格式化为更易读的字符串。"""
    if parameter_count >= 1_000_000_000:
        return f"{parameter_count / 1_000_000_000:.2f}B"
    if parameter_count >= 1_000_000:
        return f"{parameter_count / 1_000_000:.2f}M"
    if parameter_count >= 1_000:
        return f"{parameter_count / 1_000:.2f}K"
    return str(parameter_count)


def display_model_parameter_summary(model) -> None:
    """打印模型参数量摘要。"""
    total_parameters, trainable_parameters = count_model_parameters(model)
    frozen_parameters = total_parameters - trainable_parameters
    print(
        "模型参数量",
        f"{format_parameter_count(total_parameters)} ({total_parameters:,})",
    )
    print(
        "可训练参数量",
        f"{format_parameter_count(trainable_parameters)} ({trainable_parameters:,})",
    )
    print(
        "冻结参数量",
        f"{format_parameter_count(frozen_parameters)} ({frozen_parameters:,})",
    )


