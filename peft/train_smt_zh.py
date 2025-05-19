import os
import sys
import json
import copy
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any
from datasets import list_datasets, Dataset
from transformers import HfArgumentParser, TrainingArguments, set_seed, Trainer, get_scheduler
from transformers.trainer import *
from transformers.trainer import _is_peft_model

# 导入必要的库和模块
from smt_config import SMTConfig, smt_create_and_prepare_model, create_local_datasets, create_hf_datasets, \
    gradient_collection, get_optimizer_grouped_parameters
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
import collections
import math
import warnings
from packaging import version
from tuner.smt import SMTModel

from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import TrainOutput
from transformers.utils import logging
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import reissue_pt_warnings
from transformers.file_utils import WEIGHTS_NAME
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import os

# 设置环境变量以禁用wandb
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_DISABLED"] = "true"

logger = logging.get_logger(__name__)


@dataclass
class SMTModelArguments:
    """
    SMT模型参数类
    定义与稀疏矩阵调优(Sparse Matrix Tuning)相关的所有参数
    """
    model_name_or_path: str = field(
        metadata={"help": "预训练模型路径或huggingface.co/models上的模型标识符"})
    chat_template_format: Optional[str] = field(default="none", metadata={
        "help": "chatml|zephyr|none. 如果数据集已经用聊天模板格式化，传递`none`"})
    use_peft_smt: Optional[bool] = field(default=False, metadata={"help": "是否启用PEFT SMT进行训练"})

    smt_dropout: float = field(default=0.0)
    smt_offload: Optional[bool] = field(default=False, metadata={"help": "SMT的卸载状态"})
    smt_zero_stage: int = field(default=0, metadata={"help": "SMT的ZeRO优化级别"
                                                             "选择 [0, 1, 2, 3]之一"})

    num_submatrix_mlp: int = field(default=0, metadata={"help": "应用于MLP层的子矩阵数量"})
    num_submatrix_attn: int = field(default=0, metadata={"help": "应用于注意力层的子矩阵数量"})

    selection_strategy: str = field(default="no_restriction",
                                    metadata={"help": "子矩阵分布选择策略"})
    calculation_strategy: str = field(default="mean_abs", metadata={"help": "子矩阵内梯度计算方法"})
    target_modules: List[str] = field(
        default=None,
        metadata={
            "help": "要替换为SMT的模块名称列表或模块名称的正则表达式"
                    "例如 ['q', 'v', 'k'] 或 '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'"
        },
    )
    smt_deepspeed: Optional[str] = field(default="configs/smt_deepspeed_config.yaml",
                                         metadata={"help": "SMT微调的deepspeed配置路径"})

    smt_learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "SMT的学习率"})
    smt_w_decay: Optional[float] = field(default=0.0, metadata={"help": "SMT的权重衰减"})


@dataclass
class FFTModelArguments:
    """
    全量微调(Full Fine-Tuning)模型参数类
    定义与全量微调阶段相关的所有参数
    """
    full_ft_steps: int = field(default=100, metadata={"help": "在SMT之前进行全量微调的迭代次数"})
    use_reentrant: Optional[bool] = field(default=False,
                                          metadata={"help": "梯度检查点参数。请参考相关文档"})
    fft_offload: Optional[bool] = field(default=False, metadata={"help": "FFT的卸载状态"})
    fft_zero_stage: Optional[int] = field(default=2, metadata={"help": "FFT的ZeRO优化级别"
                                                                       "选择 [0, 1, 2, 3]之一"})


@dataclass
class DataArguments:
    """
    数据集参数类
    定义与训练和评估数据集相关的所有参数
    """
    dataset_name_or_path: Optional[str] = field(
        default="/u/li19/data_folder/hector/peft-smt/data/commonsense_170k.json",
        metadata={"help": "要使用的数据集名称或路径"})
    eval_set_size: Optional[int] = field(default=120, metadata={"help": "验证集大小"})
    compute_fp32_loss: Optional[bool] = field(default=False, metadata={
        "help": "是否将<|endoftext|>作为额外的特殊标记添加到分词器"})
    add_eot_token: Optional[bool] = field(default=False, metadata={
        "help": "与低精度数据类型(fp16, bf16等)相关。如果指定，则以fp32计算损失"})
    max_seq_length: Optional[int] = field(default=2048)

    # 源代码相关参数
    append_concat_token: Optional[bool] = field(default=False, metadata={
        "help": "如果为True，在每个样本末尾附加`eos_token_id`"})
    add_special_tokens: Optional[bool] = field(default=False, metadata={
        "help": "如果为True，分词器将特殊标记添加到每个被处理的样本"})
    packing: Optional[bool] = field(default=False, metadata={"help": "是否使用打包方式创建数据集"})


def main():
    """
    主函数：训练流程的入口点
    步骤:
    1. 解析命令行参数
    2. 设置随机种子
    3. 创建模型、分词器
    4. 加载数据集
    5. 先进行全量微调(FFT)阶段训练
    6. 收集梯度信息
    7. 进行稀疏矩阵调优(SMT)阶段训练
    8. 保存最终模型
    """
    # 解析命令行参数到数据类
    parser = HfArgumentParser((SMTModelArguments, FFTModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        smt_model_args, fft_model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        smt_model_args, fft_model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 检查DeepSpeed配置是否有效
    if training_args.deepspeed:
        try:
            with open(training_args.deepspeed, 'r') as f:
                fft_deepspeed_config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"读取SMT DeepSpeed配置文件时出错: {e}")
            sys.exit(1)

    # 检查SMT DeepSpeed配置是否有效
    if smt_model_args.smt_deepspeed:
        try:
            with open(smt_model_args.smt_deepspeed, 'r') as f:
                smt_deepspeed_config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"读取FFT DeepSpeed配置文件时出错: {e}")
            sys.exit(1)

    # 设置随机种子以确保可重现性
    set_seed(training_args.seed)

    smt_model_args.full_ft_steps = fft_model_args.full_ft_steps

    training_args.deepspeed = fft_deepspeed_config
    # 设置FFT阶段的最大训练步数
    training_args.max_steps = fft_model_args.full_ft_steps

    print(f"smt_model_args: {smt_model_args}")
    print(f"fft_model_args: {fft_model_args}")
    print(f"data_args: {data_args}")
    print(f"training_args: {training_args}")

    # 创建模型、PEFT配置和分词器
    model, peft_config, tokenizer = smt_create_and_prepare_model(smt_model_args, fft_model_args, data_args)

    # 梯度检查点设置
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = training_args.gradient_checkpointing
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": fft_model_args.use_reentrant}

    # 创建数据集
    # datasets_list = list_datasets()
    # if data_args.dataset_name_or_path in datasets_list:
    #     # 如果是HuggingFace上的数据集
    #     train_dataset, eval_dataset, data_collator = create_hf_datasets(
    #         tokenizer,
    #         data_args,
    #         training_args,
    #         apply_chat_template=smt_model_args.chat_template_format != "none",
    #     )
    # else:
        # 如果是本地数据集
    train_dataset, eval_dataset, data_collator = create_local_datasets(
        tokenizer,
        data_args,
        training_args,
        apply_chat_template=smt_model_args.chat_template_format != "none",
    )

    # 创建全量微调训练器
    ffttrainer = FFTtrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        num_submatrix=smt_model_args.num_submatrix_mlp,
        num_submatrix_attentoion=smt_model_args.num_submatrix_attn,
    )

    ffttrainer.accelerator.print(f"{ffttrainer.model}")

    # 开始全量微调训练
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    ffttrainer.train(resume_from_checkpoint=checkpoint)
    # 获取MLP和注意力层的梯度
    mlp_grads = ffttrainer.mlp_warmup_grads
    attention_grads = ffttrainer.attention_warmup_grads

    ffttrainer.print_trainable_parameters()

    # 为SMT阶段更新训练参数
    training_args.learning_rate = smt_model_args.smt_learning_rate
    training_args.weight_decay = smt_model_args.smt_w_decay
    training_args.deepspeed = smt_deepspeed_config

    # 重置SMT阶段的最大训练步数
    training_args.max_steps = -1

    print(f"smt_model_args: {smt_model_args}")
    print(f"fft_model_args: {fft_model_args}")
    print(f"data_args: {data_args}")
    print(f"training_args: {training_args}")

    model = ffttrainer.model
    # 创建SMT模型，使用收集的梯度
    smt_model = SMTModel(smt_model_args, model, attention_grads, mlp_grads)

    # 保存梯度
    if smt_model_args.num_submatrix_mlp > 0:
        save_gradients(mlp_grads, "mlp_grad", training_args.output_dir)
    if smt_model_args.num_submatrix_attn > 0:
        save_gradients(attention_grads, "attn_grad", training_args.output_dir)

    # 创建SMT训练器
    smttrainer = SMTTrainer(
        model=smt_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        fft_steps=fft_model_args.full_ft_steps,
    )

    smttrainer.accelerator.print(f"{smttrainer.model}")
    smttrainer.model.print_trainable_parameters()

    # 开始SMT训练
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    smttrainer.train(resume_from_checkpoint=checkpoint)

    # 保存最终模型
    if smttrainer.is_fsdp_enabled:
        smttrainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    smttrainer.save_model()


def save_gradients(gradients, grad_name, output_dir):
    """
    保存梯度信息到文件
    
    该函数将在FFT阶段收集的梯度信息保存到磁盘，以便在SMT阶段使用。
    梯度会被保存为PyTorch张量文件(.pt)，存储在以梯度名称命名的子目录中。
    
    参数:
        gradients (dict): 要保存的梯度字典，通常包含模型层名称和对应的梯度张量
        grad_name (str): 梯度类型名称，通常为'mlp_grad'（MLP层梯度）或'attn_grad'（注意力层梯度）
        output_dir (str): 输出根目录路径
        
    保存结构:
        {output_dir}/{grad_name}/gradients.pt
    """
    # 确保输出根目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 构建梯度文件完整路径
    gradient_file = os.path.join(output_dir, grad_name, "gradients.pt")
    # 创建梯度类型对应的子目录
    folder = os.path.join(output_dir, grad_name)
    os.makedirs(folder, exist_ok=True)
    
    # 使用torch.save保存梯度张量到文件
    torch.save(gradients, gradient_file)


class FFTtrainer(Trainer):
    """
    全量微调(Full Fine-Tuning)训练器
    继承自Transformers的Trainer类，添加了梯度收集功能
    
    在训练过程中收集MLP和注意力层的梯度信息，供后续SMT使用
    """
    def __init__(
            self,
            model=None,
            args: TrainingArguments = None,
            data_collator=None,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=None,
            model_init=None,
            compute_metrics=None,
            callbacks=None,
            optimizers=(None, None),
            num_submatrix=0,
            num_submatrix_attentoion=0,
    ):
        # 初始化梯度存储
        self.mlp_warmup_grads = {}
        self.attention_warmup_grads = {}
        self.num_submatrix = num_submatrix
        self.num_submatrix_attentoion = num_submatrix_attentoion
        print(f"args within training class: {args}")

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

    # 重写train方法，在合适的位置收集梯度信息
    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
            trial: Union["optuna.Trial", Dict[str, Any]] = None,
            ignore_keys_for_eval: Optional[List[str]] = None,
            **kwargs,
    ):
        """
        全量微调(FFT)阶段的训练函数
        
        该函数是整个训练流程的核心，主要功能包括：
        1. 训练前的准备工作（加载检查点、初始化模型、优化器等）
        2. 训练循环的执行（按epoch和step进行迭代训练）
        3. 在梯度裁剪后收集MLP和注意力层的梯度信息，以供后续SMT阶段使用
        4. 训练过程中的评估、日志记录和检查点保存
        5. 训练完成后的清理工作
        
        参数:
            resume_from_checkpoint: 恢复训练的检查点路径或布尔值
            trial: Optuna超参数搜索的trial对象
            ignore_keys_for_eval: 评估时忽略的键列表
            **kwargs: 额外的关键字参数
            
        返回:
            TrainOutput: 包含训练步数、训练损失和其他指标的对象
        """
        # train方法的主要实现，继承自原始Trainer类，但在梯度裁剪后添加了梯度收集逻辑
        # 该方法很长，大部分代码与原始Trainer.train相同，只在关键处添加了自定义代码
        
        # ... [此处省略大量继承自Trainer的代码] ...
        
        # 关键添加的代码位于梯度裁剪之后:
        # self.mlp_warmup_grads, self.attention_warmup_grads = gradient_collection(model,
        #                                                                         self.mlp_warmup_grads,
        #                                                                         self.attention_warmup_grads,
        #                                                                         self.num_submatrix,
        #                                                                         self.num_submatrix_attentoion)
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # 尽早设置内存指标，用于跟踪训练过程中的内存使用情况
        self._memory_tracker.start()

        # 获取训练参数
        args = self.args

        # 设置训练状态标志
        self.is_in_train = True

        # 如果设置了NEFTune噪声参数，为模型添加NEFTune钩子
        # NEFTune是一种通过向嵌入添加噪声来提高语言模型性能的技术
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train参数不可靠的处理方式
        # 即使没有设置do_train，train()方法也可能被调用
        # 以下是一种变通方法，在评估时将模型移到正确的设备上
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        # 处理废弃的model_path参数，现在应该使用resume_from_checkpoint
        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path`已废弃，将在未来版本中移除。请使用`resume_from_checkpoint`",
                FutureWarning,
            )
            
        # 检查是否有未知的关键字参数传入
        if len(kwargs) > 0:
            raise TypeError(f"train()收到了意外的关键字参数: {', '.join(list(kwargs.keys()))}.")
            
        # 超参数搜索设置 - 这可能会改变随机种子，所以需要先运行
        self._hp_search_setup(trial)
        
        # 设置训练批次大小
        self._train_batch_size = self.args.train_batch_size

        # 模型重新初始化逻辑
        model_reloaded = False
        if self.model_init is not None:
            # 使用model_init时，必须在实例化模型前设置随机种子
            # 根据full_determinism参数决定使用哪种随机种子设置方法
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            
            # 调用模型初始化函数
            self.model = self.call_model_init(trial)
            model_reloaded = True
            
            # 重新初始化优化器和学习率调度器
            self.optimizer, self.lr_scheduler = None, None

        # 加载潜在的模型检查点
        # 如果resume_from_checkpoint是True，则自动查找输出目录中的最后一个检查点
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"在输出目录({args.output_dir})中未找到有效的检查点")

        # 如果指定了检查点路径，从检查点加载模型状态
        if resume_from_checkpoint is not None:
            # 只有在不使用SageMaker MP、DeepSpeed或FSDP时才直接加载检查点
            if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
                
            # 在重复find_executable_batch_size的情况下，正确设置`self._train_batch_size`
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

        # 如果模型被重新初始化，将其放在正确的设备上并更新self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        # 查找可执行的批次大小并获取内部训练循环函数
        # 这允许自动查找当前硬件可处理的最大批次大小
        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        
        # 处理模型上传到Hugging Face Hub的情况
        if args.push_to_hub:
            try:
                # 上传模型检查点时禁用进度条，避免污染标准输出
                hf_hub_utils.disable_progress_bars()
                return inner_training_loop(
                    args=args,
                    resume_from_checkpoint=resume_from_checkpoint,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )
            finally:
                # 确保无论如何都会重新启用进度条
                hf_hub_utils.enable_progress_bars()
        else:
            # 常规情况下，直接调用内部训练循环
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        内部训练循环方法 - 实现训练的核心逻辑
        
        该方法是训练过程的实际执行部分，负责:
        1. 初始化训练环境和状态
        2. 创建数据加载器和设置训练控制变量
        3. 执行训练循环，包括梯度计算、梯度收集、参数更新等
        4. 处理评估、日志记录和检查点保存
        5. 返回训练结果
        
        参数:
            batch_size: 训练批次大小
            args: 训练参数
            resume_from_checkpoint: 恢复训练的检查点路径
            trial: 超参数优化的trial对象
            ignore_keys_for_eval: 评估时忽略的键列表
            
        返回:
            TrainOutput对象，包含训练步数、损失和其他指标
        """
        # 释放加速器使用的内存，避免内存泄漏
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        
        # 自动查找批次大小的相关处理
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                # 释放模型包装器的内存
                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # 如果启用了DeepSpeed，需要修改配置
                if self.is_deepspeed_enabled:
                    # 临时取消设置训练批次大小
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
            
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        
        # 创建训练数据加载器
        train_dataloader = self.get_train_dataloader()
        
        # TPU特定处理
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # 设置训练控制变量:
        # - num_train_epochs: 训练的总轮数
        # - num_update_steps_per_epoch: 每个epoch的更新步数
        # - max_steps: 要执行的总训练步数
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        
        # 当数据加载器有确定长度时的处理
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            # 计算每个epoch的更新步数
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            
            # 如果设置了max_steps
            if args.max_steps > 0:
                max_steps = args.max_steps
                # 根据max_steps和每个epoch的步数计算总epoch数
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # 可能在最后一个批次中稍有不准确，但这是能做到的最好情况
                num_train_samples = args.max_steps * total_train_batch_size
                # 计算token数量（如果需要）
                if args.include_tokens_per_second:
                    num_train_tokens = (
                            self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                # 如果没有设置max_steps，基于epoch数计算
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        # 当数据加载器没有确定长度时，依赖max_steps
        elif args.max_steps > 0:
            max_steps = args.max_steps
            # 设置非常大的epoch数，以便在迭代器上根据需要多次遍历
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            # 如果数据加载器没有长度且未设置max_steps，则抛出错误
            raise ValueError(
                "args.max_steps必须设置为正值，如果数据加载器没有长度，当前值为"
                f" {args.max_steps}"
            )

        # 调试选项 - 处理下溢/上溢
        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model)复制模型，创建新变量，这里注册的模块引用在其他GPU上不再有效
                raise ValueError(
                    "当前--debug underflow_overflow在DP下不支持。请使用DDP"
                    " (torchrun或torch.distributed.launch(已弃用))。"
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        # 是否延迟创建优化器
        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # 重置学习率调度器，因为其参数在后续调用中可能不同
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        # DeepSpeed初始化
        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        # 如果不延迟创建优化器，立即创建优化器和调度器
        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # 创建训练状态
        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # 根据比例计算日志记录、评估和保存步骤的绝对值
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # 如果需要，激活梯度检查点
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        # 包装模型以支持分布式训练或其他优化
        model = self._wrap_model(self.model_wrapped)

        # 根据模型包装情况确定是否使用accelerator.prepare
        # 这是为了处理未处理的情况，如FSDP-XLA、SageMaker MP/DP、DataParallel、IPEX等
        use_accelerator_prepare = True if model is self.model else False

        # 如果延迟创建优化器，在模型准备好后创建
        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # 使用`accelerator`准备模型、优化器和调度器
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # 处理传递"DummyScheduler"的情况，例如在DeepSpeed配置中指定时
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # 在这种情况下，我们有DDP + LOMO，应该支持
            self.optimizer = self.accelerator.prepare(self.optimizer)

        # FSDP特定处理
        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # 在此函数的其余部分中，`model`是外部模型，无论它是否被包装
        if model is not self.model:
            self.model_wrapped = model

        # 向后兼容
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # 加载检查点
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # 检查是否存在已保存的优化器或调度器状态，并加载
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # 重要提示：在这一点上:
        # self.model         是Transformers模型
        # self.model_wrapped 是DDP(Transformers模型)、Deepspeed(Transformers模型)、
        # FSDP(Transformers模型)、Dynamo优化模块(Transformers模型)等

        # 开始训练!
        logger.info("***** 开始训练 *****")
        logger.info(f"  样本数量 = {num_examples:,}")
        logger.info(f"  训练轮数 = {num_train_epochs:,}")
        logger.info(f"  每个设备的即时批次大小 = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  使用DataParallel训练，批次大小已调整为: {self._train_batch_size:,}")
        logger.info(f"  总训练批次大小 (包括并行、分布式和累积) = {total_train_batch_size:,}")
        logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
        logger.info(f"  总优化步数 = {max_steps:,}")
        logger.info(f"  可训练参数数量 = {get_model_param_count(model, trainable_only=True):,}")

        # 初始化训练状态变量
        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # 检查是否从检查点继续训练
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            # 从检查点加载训练状态
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            
            # 计算已训练的轮数和当前轮中已训练的步数
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                # 计算在当前epoch中已经训练过的步数，需要跳过这些步骤
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                # 考虑梯度累积因素，实际需要跳过的批次数量为步数乘以梯度累积步数
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                # 如果设置了ignore_data_skip，则不跳过任何步骤
                steps_trained_in_current_epoch = 0

            # 记录从检查点恢复训练的信息
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", self.state.global_step)
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # 更新回调处理器的引用
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        
        # 处理超参数优化相关设置
        if self.hp_name is not None and self._trial is not None:
            # 使用self._trial因为SigOpt/Optuna超参数优化在使用DDP时只调用`_hp_search_setup(trial)`
            # 而不是将trial参数传递给Train
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            # 获取超参数分配，根据后端类型选择不同的获取方式
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
            
        # 即使状态已保存，这些值也应相同，但为了安全起见，
        # 在加载后重新设置这些值，以防训练参数发生变化
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # 训练损失初始化
        # 使用张量来避免通过.item()方法同步TPU
        tr_loss = torch.tensor(0.0).to(args.device)
        # _logging_loss_scalar在每次需要调用tr_loss.item()时更新，存储所有损失的总和
        self._logging_loss_scalar = 0
        # 记录总浮点运算次数
        self._total_flos = self.state.total_flos
        # 清零模型梯度
        model.zero_grad()

        # 触发训练开始回调
        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # 训练主循环 - 按epoch迭代
        for epoch in range(epochs_trained, num_train_epochs):
            # 如果使用分布式采样器，在每个epoch开始时设置epoch
            # 这确保了在分布式训练中每个进程看到不同的数据样本
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # 如果需要，在每个epoch开始时重置past mems状态
            # past_index是用于缓存注意力值的参数，主要用于加速生成式模型的推理
            if self.args.past_index >= 0:
                self._past = None

            # 计算当前epoch中的步数
            steps_in_epoch = len(epoch_iterator) if train_dataset_is_sized else self.args.max_steps
            # 触发epoch开始回调
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            # 批次训练循环 - 处理每个批次
            for step, inputs in enumerate(epoch_iterator):

                # 如果恢复训练，跳过已训练的步骤
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                # 在每个梯度累积周期的开始触发步骤开始回调
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                # 处理分布式训练中的梯度同步
                # 在非梯度累积步骤且使用DDP时，使用no_sync()上下文管理器避免不必要的梯度同步
                if (
                        ((step + 1) % self.args.gradient_accumulation_steps != 0)
                        and self.args.local_rank != -1
                        and _use_ddp_no_sync
                ):
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs)
                else:
                    tr_loss += self.training_step(model, inputs)
                    
                # 更新浮点运算计数
                self._total_flos += self.floating_point_ops(inputs)

                # 在梯度累积完成或是epoch的最后一步时执行梯度更新
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # 处理最后一步但步数小于梯度累积步数的情况
                        steps_in_epoch <= self.args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    # 梯度裁剪 - 防止梯度爆炸问题
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    # 执行优化器步骤，更新模型参数
                    self.optimizer.step()

                    # 更新学习率调度器
                    self.lr_scheduler.step()
                    # 清零模型梯度，准备下一步
                    model.zero_grad()
                    # 更新全局步数和epoch进度
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    # 触发步骤结束回调
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                    # 可能记录日志、保存模型和进行评估
                    self._maybe_log_save_evalute(tr_loss, model, trial, epoch)

                # 检查是否应该停止当前epoch或整个训练
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            # 触发epoch结束回调
            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            # epoch结束时记录日志、保存模型和评估
            self._maybe_log_save_evalute(tr_loss, model, trial, epoch)

            # TPU指标调试检查
            if self.args.tpu_metrics_debug or self.args.debug:
                logger.warning(
                    "您启用了PyTorch/XLA调试指标，但您没有配置TPU。"
                    "如果这是意外情况，请检查您的训练配置。"
                )
                
            # 检查是否应该停止训练
            if self.control.should_training_stop:
                break

        # 训练结束后清理工作
        if self.args.past_index and hasattr(self, "_past"):
            # 在训练结束时清理past状态
            delattr(self, "_past")

        # 训练完成
        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        
        # 如果设置了加载最佳模型并且存在最佳模型检查点
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            # 根据模型类型选择不同的加载方式
            if isinstance(model, PreTrainedModel):
                # 对于PreTrainedModel，使用from_pretrained方法加载
                self.model = model.from_pretrained(self.state.best_model_checkpoint)
                self.model = self.model.to(self.args.device)
            else:
                # 对于其他模型，直接加载状态字典
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

        # 触发训练结束回调
        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)

        # 返回训练结果：全局步数和每步平均损失
        return TrainOutput(self.state.global_step, tr_loss.item() / self.state.global_step)

   


    def print_trainable_parameters(self):
        """
        打印模型中可训练参数的数量
        计算并显示可训练参数占总参数的百分比
        """
        trainable_params = 0
        all_param = 0
        for param in self.model.parameters():
            num_params = param.numel()
            # 如果使用DS Zero 3且权重初始化为空
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

class SMTTrainer(Trainer):
    """
    稀疏矩阵调优(Sparse Matrix Tuning)训练器
    继承自Transformers的Trainer类，专为SMT训练阶段设计
    
    使用从FFT阶段收集的梯度信息来训练SMT模型
    """
    def __init__(
            self,
            model=None,
            args: TrainingArguments = None,
            data_collator=None,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=None,
            model_init=None,
            compute_metrics=None,
            callbacks=None,
            optimizers=(None, None),
            fft_steps=None,
    ):
        # 记录FFT阶段的训练步数，用于正确跳过已训练的步骤
        self.steps_trained_in_current_epoch = fft_steps
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

    # 重写train方法，与FFTtrainer类似，但定制为SMT训练阶段的需求
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        稀疏矩阵调优(SMT)阶段的训练函数
        
        该函数是SMT训练流程的核心，主要功能包括：
        1. 训练前的准备工作（加载检查点、初始化模型、优化器等）
        2. 训练循环的执行（按epoch和step进行迭代训练）
        3. 使用在FFT阶段收集的梯度信息指导SMT模型的训练
        4. 训练过程中的评估、日志记录和检查点保存
        5. 训练完成后的清理工作
        
        与FFTTrainer的区别：
        - 已经包含了预先收集的梯度信息
        - 训练的是包含稀疏矩阵的模型
        - 会跳过已经在FFT阶段训练过的步骤
        
        参数:
            resume_from_checkpoint: 恢复训练的检查点路径或布尔值
            trial: Optuna超参数搜索的trial对象
            ignore_keys_for_eval: 评估时忽略的键列表
            **kwargs: 额外的关键字参数
            
        返回:
            TrainOutput: 包含训练步数、训练损失和其他指标的对象
        """
        # train方法的主要实现，继承自原始Trainer类，但针对SMT做了定制
        
        # ... [此处省略大量继承自Trainer的代码] ...


        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # 尽早设置内存指标，用于跟踪训练过程中的内存使用情况
        self._memory_tracker.start()

        # 获取训练参数
        args = self.args

        # 设置训练状态标志
        self.is_in_train = True

        # 如果设置了NEFTune噪声参数，为模型添加NEFTune钩子
        # NEFTune是一种通过向嵌入添加噪声来提高语言模型性能的技术
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train参数不可靠的处理方式
        # 即使没有设置do_train，train()方法也可能被调用
        # 以下是一种变通方法，在评估时将模型移到正确的设备上
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        # 处理废弃的model_path参数，现在应该使用resume_from_checkpoint
        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path`已废弃，将在未来版本中移除。请使用`resume_from_checkpoint`",
                FutureWarning,
            )
            
        # 检查是否有未知的关键字参数传入
        if len(kwargs) > 0:
            raise TypeError(f"train()收到了意外的关键字参数: {', '.join(list(kwargs.keys()))}.")
            
        # 超参数搜索设置 - 这可能会改变随机种子，所以需要先运行
        self._hp_search_setup(trial)
        
        # 设置训练批次大小
        self._train_batch_size = self.args.train_batch_size

        # 模型重新初始化逻辑
        model_reloaded = False
        if self.model_init is not None:
            # 使用model_init时，必须在实例化模型前设置随机种子
            # 根据full_determinism参数决定使用哪种随机种子设置方法
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            
            # 调用模型初始化函数
            self.model = self.call_model_init(trial)
            model_reloaded = True
            
            # 重新初始化优化器和学习率调度器
            self.optimizer, self.lr_scheduler = None, None

        # 加载潜在的模型检查点
        # 如果resume_from_checkpoint是True，则自动查找输出目录中的最后一个检查点
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"在输出目录({args.output_dir})中未找到有效的检查点")

        # 如果指定了检查点路径，从检查点加载模型状态
        if resume_from_checkpoint is not None:
            # 只有在不使用SageMaker MP、DeepSpeed或FSDP时才直接加载检查点
            if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
                
            # 在重复find_executable_batch_size的情况下，正确设置`self._train_batch_size`
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

        # 如果模型被重新初始化，将其放在正确的设备上并更新self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        # 查找可执行的批次大小并获取内部训练循环函数
        # 这允许自动查找当前硬件可处理的最大批次大小
        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        
        # 处理模型上传到Hugging Face Hub的情况
        if args.push_to_hub:
            try:
                # 上传模型检查点时禁用进度条，避免污染标准输出
                hf_hub_utils.disable_progress_bars()
                return inner_training_loop(
                    args=args,
                    resume_from_checkpoint=resume_from_checkpoint,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )
            finally:
                # 确保无论如何都会重新启用进度条
                hf_hub_utils.enable_progress_bars()
        else:
            # 常规情况下，直接调用内部训练循环
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        内部训练循环方法 - 实现训练的核心逻辑
        
        该方法是训练过程的实际执行部分，负责:
        1. 初始化训练环境和状态
        2. 创建数据加载器和设置训练控制变量
        3. 执行训练循环，包括梯度计算、梯度收集、参数更新等
        4. 处理评估、日志记录和检查点保存
        5. 返回训练结果
        
        参数:
            batch_size: 训练批次大小
            args: 训练参数
            resume_from_checkpoint: 恢复训练的检查点路径
            trial: 超参数优化的trial对象
            ignore_keys_for_eval: 评估时忽略的键列表
            
        返回:
            TrainOutput对象，包含训练步数、损失和其他指标
        """
        # 释放加速器使用的内存，避免内存泄漏
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        
        # 自动查找批次大小的相关处理
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                # 释放模型包装器的内存
                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # 如果启用了DeepSpeed，需要修改配置
                if self.is_deepspeed_enabled:
                    # 临时取消设置训练批次大小
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
            
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        
        # 创建训练数据加载器
        train_dataloader = self.get_train_dataloader()
        
        # TPU特定处理
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # 设置训练控制变量:
        # - num_train_epochs: 训练的总轮数
        # - num_update_steps_per_epoch: 每个epoch的更新步数
        # - max_steps: 要执行的总训练步数
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        
        # 当数据加载器有确定长度时的处理
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            # 计算每个epoch的更新步数
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            
            # 如果设置了max_steps
            if args.max_steps > 0:
                max_steps = args.max_steps
                # 根据max_steps和每个epoch的步数计算总epoch数
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # 可能在最后一个批次中稍有不准确，但这是能做到的最好情况
                num_train_samples = args.max_steps * total_train_batch_size
                # 计算token数量（如果需要）
                if args.include_tokens_per_second:
                    num_train_tokens = (
                            self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                # 如果没有设置max_steps，基于epoch数计算
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        # 当数据加载器没有确定长度时，依赖max_steps
        elif args.max_steps > 0:
            max_steps = args.max_steps
            # 设置非常大的epoch数，以便在迭代器上根据需要多次遍历
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            # 如果数据加载器没有长度且未设置max_steps，则抛出错误
            raise ValueError(
                "args.max_steps必须设置为正值，如果数据加载器没有长度，当前值为"
                f" {args.max_steps}"
            )

        # 调试选项 - 处理下溢/上溢
        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model)复制模型，创建新变量，这里注册的模块引用在其他GPU上不再有效
                raise ValueError(
                    "当前--debug underflow_overflow在DP下不支持。请使用DDP"
                    " (torchrun或torch.distributed.launch(已弃用))。"
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        # 是否延迟创建优化器
        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # 重置学习率调度器，因为其参数在后续调用中可能不同
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        # DeepSpeed初始化
        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        # 如果不延迟创建优化器，立即创建优化器和调度器
        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # 创建训练状态
        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # 根据比例计算日志记录、评估和保存步骤的绝对值
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # 如果需要，激活梯度检查点
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        # 包装模型以支持分布式训练或其他优化
        model = self._wrap_model(self.model_wrapped)

        # 根据模型包装情况确定是否使用accelerator.prepare
        # 这是为了处理未处理的情况，如FSDP-XLA、SageMaker MP/DP、DataParallel、IPEX等
        use_accelerator_prepare = True if model is self.model else False

        # 如果延迟创建优化器，在模型准备好后创建
        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # 使用`accelerator`准备模型、优化器和调度器
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # 处理传递"DummyScheduler"的情况，例如在DeepSpeed配置中指定时
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # 在这种情况下，我们有DDP + LOMO，应该支持
            self.optimizer = self.accelerator.prepare(self.optimizer)

        # FSDP特定处理
        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # 在此函数的其余部分中，`model`是外部模型，无论它是否被包装
        if model is not self.model:
            self.model_wrapped = model

        # 向后兼容
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # 加载检查点
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # 检查是否存在已保存的优化器或调度器状态，并加载
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # 重要提示：在这一点上:
        # self.model         是Transformers模型
        # self.model_wrapped 是DDP(Transformers模型)、Deepspeed(Transformers模型)、
        # FSDP(Transformers模型)、Dynamo优化模块(Transformers模型)等

        # 开始训练!
        logger.info("***** 开始训练 *****")
        logger.info(f"  样本数量 = {num_examples:,}")
        logger.info(f"  训练轮数 = {num_train_epochs:,}")
        logger.info(f"  每个设备的即时批次大小 = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  使用DataParallel训练，批次大小已调整为: {self._train_batch_size:,}")
        logger.info(f"  总训练批次大小 (包括并行、分布式和累积) = {total_train_batch_size:,}")
        logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
        logger.info(f"  总优化步数 = {max_steps:,}")
        logger.info(f"  可训练参数数量 = {get_model_param_count(model, trainable_only=True):,}")

        # 初始化训练状态变量
        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = self.steps_trained_in_current_epoch
        steps_trained_progress_bar = None

        # 检查是否从检查点继续训练
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            # 从检查点加载训练状态
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            
            # 计算已训练的轮数和当前轮中已训练的步数
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                # 计算在当前epoch中已经训练过的步数，需要跳过这些步骤
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                # 考虑梯度累积因素，实际需要跳过的批次数量为步数乘以梯度累积步数
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                # 如果设置了ignore_data_skip，则不跳过任何步骤
                steps_trained_in_current_epoch = 0

            # 记录从检查点恢复训练的信息
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", self.state.global_step)
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # 更新回调处理器的引用
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        
        # 处理超参数优化相关设置
        if self.hp_name is not None and self._trial is not None:
            # 使用self._trial因为SigOpt/Optuna超参数优化在使用DDP时只调用`_hp_search_setup(trial)`
            # 而不是将trial参数传递给Train
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            # 获取超参数分配，根据后端类型选择不同的获取方式
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
            
        # 即使状态已保存，这些值也应相同，但为了安全起见，
        # 在加载后重新设置这些值，以防训练参数发生变化
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # 训练损失初始化
        # 使用张量来避免通过.item()方法同步TPU
        tr_loss = torch.tensor(0.0).to(args.device)
        # _logging_loss_scalar在每次需要调用tr_loss.item()时更新，存储所有损失的总和
        self._logging_loss_scalar = 0
        # 记录总浮点运算次数
        self._total_flos = self.state.total_flos
        # 清零模型梯度
        model.zero_grad()

        # 触发训练开始回调
        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # 训练主循环 - 按epoch迭代
        for epoch in range(epochs_trained, num_train_epochs):
            # 如果使用分布式采样器，在每个epoch开始时设置epoch
            # 这确保了在分布式训练中每个进程看到不同的数据样本
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # 如果需要，在每个epoch开始时重置past mems状态
            # past_index是用于缓存注意力值的参数，主要用于加速生成式模型的推理
            if self.args.past_index >= 0:
                self._past = None

            # 计算当前epoch中的步数
            steps_in_epoch = len(epoch_iterator) if train_dataset_is_sized else self.args.max_steps
            # 触发epoch开始回调
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            # 批次训练循环 - 处理每个批次
            for step, inputs in enumerate(epoch_iterator):

                # 如果恢复训练，跳过已训练的步骤
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                # 在每个梯度累积周期的开始触发步骤开始回调
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                # 处理分布式训练中的梯度同步
                # 在非梯度累积步骤且使用DDP时，使用no_sync()上下文管理器避免不必要的梯度同步
                if (
                        ((step + 1) % self.args.gradient_accumulation_steps != 0)
                        and self.args.local_rank != -1
                        and _use_ddp_no_sync
                ):
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs)
                else:
                    tr_loss += self.training_step(model, inputs)
                    
                # 更新浮点运算计数
                self._total_flos += self.floating_point_ops(inputs)

                # 在梯度累积完成或是epoch的最后一步时执行梯度更新
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # 处理最后一步但步数小于梯度累积步数的情况
                        steps_in_epoch <= self.args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    # 梯度裁剪 - 防止梯度爆炸问题
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    # 执行优化器步骤，更新模型参数
                    self.optimizer.step()

                    # 更新学习率调度器
                    self.lr_scheduler.step()
                    # 清零模型梯度，准备下一步
                    model.zero_grad()
                    # 更新全局步数和epoch进度
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    # 触发步骤结束回调
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                    # 可能记录日志、保存模型和进行评估
                    self._maybe_log_save_evalute(tr_loss, model, trial, epoch)

                # 检查是否应该停止当前epoch或整个训练
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            # 触发epoch结束回调
            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            # epoch结束时记录日志、保存模型和评估
            self._maybe_log_save_evalute(tr_loss, model, trial, epoch)

            # TPU指标调试检查
            if self.args.tpu_metrics_debug or self.args.debug:
                logger.warning(
                    "您启用了PyTorch/XLA调试指标，但您没有配置TPU。"
                    "如果这是意外情况，请检查您的训练配置。"
                )
                
            # 检查是否应该停止训练
            if self.control.should_training_stop:
                break

        # 训练结束后清理工作
        if self.args.past_index and hasattr(self, "_past"):
            # 在训练结束时清理past状态
            delattr(self, "_past")

        # 训练完成
        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        
        # 如果设置了加载最佳模型并且存在最佳模型检查点
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            # 根据模型类型选择不同的加载方式
            if isinstance(model, PreTrainedModel):
                # 对于PreTrainedModel，使用from_pretrained方法加载
                self.model = model.from_pretrained(self.state.best_model_checkpoint)
                self.model = self.model.to(self.args.device)
            else:
                # 对于其他模型，直接加载状态字典
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

        # 触发训练结束回调
        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)

        # 返回训练结果：全局步数和每步平均损失
        return TrainOutput(self.state.global_step, tr_loss.item() / self.state.global_step)

    def print_trainable_parameters(self):
        """
        打印模型中可训练参数的数量和比例
        
        该方法统计并显示以下信息：
        1. 可训练参数的总数量（requires_grad=True的参数）
        2. 模型中所有参数的总数量
        3. 可训练参数占总参数的百分比
        
        这对于监控SMT的参数效率非常重要，可以清晰地看到与全量微调相比减少了多少训练参数
        """
        trainable_params = 0  # 可训练参数计数器
        all_param = 0  # 所有参数计数器
        
        # 遍历模型的所有参数
        for param in self.model.parameters():
            # 获取参数中元素的数量
            num_params = param.numel()
            
            # DeepSpeed Zero-3优化的特殊处理
            # 在Zero-3中，某些参数可能初始化为空，但有ds_numel属性记录实际大小
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # 累加总参数数量
            all_param += num_params
            
            # 如果参数设置为可训练，累加到可训练参数计数器
            if param.requires_grad:
                trainable_params += num_params
                
        # 打印统计结果，包括可训练参数数量、总参数数量和可训练参数比例
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )


if __name__ == "__main__":
    main()