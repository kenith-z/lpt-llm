"""项目级配置定义。

这个模块将“训练/推理运行环境配置”“文本生成采样配置”和
“LoRA 微调配置”分开管理，避免把行为常量散落在各个模块里。
"""

from dataclasses import dataclass

import torch


class GlobalConfig:
    """全局运行配置。

    这里继续使用类属性，方便教学项目中的各个模块直接读取。
    与之前不同的是，训练长度、推理长度和 RoPE 缓存长度现在显式拆开，
    避免为了极少出现的长上下文场景而长期承担额外显存开销。
    """

    model_abbr = "LPT"
    model_name_en = "Ling Pre-trained Transformer"
    model_name_zh = "灵预"
    chat_template_version = "lpt-ds-v1"
    training_stage = "chat_sft"

    # 当前数据集较短，训练阶段没有必要默认保留超长上下文。
    train_max_sequence_length = 7680
    # 推理允许比训练更长，但仍应控制在实际可用范围内。
    inference_max_sequence_length = 65536 // 2
    # 训练态与推理态分别维护独立的 RoPE 缓存上限，避免训练时被长推理缓存占用显存。
    train_rope_cache_max_sequence_length = train_max_sequence_length
    inference_rope_cache_max_sequence_length = inference_max_sequence_length

    # 训练参数保留 FP32，前向再通过 autocast 使用 BF16，收敛更稳。
    parameter_dtype = torch.float32
    autocast_dtype = torch.bfloat16
    # attention/RetNet 的长上下文缓存路径对 BF16 数值误差更敏感，默认提到 FP32 保稳定。
    stable_attention_fp32_enabled = True
    stable_retnet_fp32_enabled = True
    # 为 Tensor Core 友好地对齐 padding 长度。
    pad_to_multiple_of = 8
    # 流式数据集使用有限缓冲区近似打乱，避免整包语料常驻内存。
    dataset_shuffle_buffer_size = 2048
    # 训练阶段启用梯度检查点，优先回收中间激活显存。
    gradient_checkpointing_enabled = True

    # 优先使用 GPU；如果没有可用 CUDA，则自动退回 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LoRA 模式下，训练流程只保存适配器参数而不保存整模型权重。
    lora_mode = False
    # 打开后会输出注意力热力图和 token 位置表，便于学习模型行为。
    attention_plot_enabled = False
    attention_plot_dir = "./attn_scores_plots"


class BaseTrainingRecipeConfig:
    """训练 recipe 通用配置。"""

    weight_decay = 0.1
    gradient_accumulation_steps = 1
    # 训练侧默认启用严格样本边界的 sequence packing，减少 padding 浪费。
    sequence_packing_enabled = True
    max_grad_norm = 1.0
    random_seed = 7
    deterministic_algorithms = True
    tensorboard_enabled = True
    log_interval_steps = 10
    eval_interval_steps: int | None = None
    eval_batch_size: int | None = None
    eval_max_batches: int | None = None
    # LongRoPE2 训练侧默认在原始窗口与训练目标窗口之间混合采样。
    # 具体窗口会在训练时根据 ModelConfig.original_max_len 和 GlobalConfig.train_max_sequence_length 解析。
    longrope2_window_sampling_enabled = True
    longrope2_window_lengths: tuple[int, ...] | None = None
    longrope2_window_sampling_weights: tuple[float, ...] | None = None
    longrope2_auto_factor_refresh_enabled = True


class TextPretrainingConfig(BaseTrainingRecipeConfig):
    """文本继续预训练配置。"""

    batch_size = 1
    target_total_epochs = 3
    learning_rate = 3e-4
    warmup_ratio = 0.1
    save_optimizer = True
    save_scheduler = True
    key_checkpoints: tuple[int, ...] = ()


class ChatSFTTrainingConfig(BaseTrainingRecipeConfig):
    """Chat 全参数监督微调配置。"""

    batch_size = 16
    target_total_epochs = 1
    learning_rate = 3e-4
    warmup_ratio = 0.1
    save_optimizer = True
    save_scheduler = True
    key_checkpoints: tuple[int, ...] = ()


class ChatLoRATrainingConfig(BaseTrainingRecipeConfig):
    """Chat LoRA 微调配置。"""

    batch_size = 10
    target_total_epochs = 1
    learning_rate = 6e-4
    warmup_ratio = 0.1
    save_optimizer = True
    save_scheduler = True
    key_checkpoints: tuple[int, ...] = ()


@dataclass
class GenerationConfig:
    """文本生成采样配置。"""

    do_sample: bool = True
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    max_length: int = 200
    repetition_penalty: float = 1.1
    # 只对最近窗口施加重复惩罚，避免历史越长额外开销越大。
    repetition_window_size: int | None = 256


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA 训练配置。"""

    rank: int = 2
    alpha: int = 4
    dropout_p: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
