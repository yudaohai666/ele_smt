#!/usr/bin/env python
# coding=utf-8

import os
import sys
import json
import torch
import torch.distributed as dist
from transformers import HfArgumentParser, set_seed

from models import SMTModelArguments, FFTModelArguments, SMTModel
from tuner.smt_base import SMTModel
from data import DataArguments, create_local_datasets
from smt_config import SMTConfig, smt_create_and_prepare_model, create_local_datasets, create_hf_datasets, \
    gradient_collection, get_optimizer_grouped_parameters
from train_smt import FFTtrainer, SMTTrainer, save_gradients,DataArguments
from transformers import TrainingArguments


def load_gradients(path, prefix):
    """Load gradients from saved files"""
    gradients = []
    i = 0
    while True:
        file_path = os.path.join(path, f"{prefix}_{i}.pt")
        if not os.path.exists(file_path):
            break
        gradient = torch.load(file_path, map_location="cpu")
        gradients.append(gradient)
        i += 1
    print(f"Loaded {len(gradients)} {prefix} gradients from {path}")
    return gradients


def main():
    # Add a custom argument for phase selection
    class PhaseArguments:
        def __init__(self):
            self.fft_warmup = True  # True to run FFT phase, False to run SMT phase
            self.resume_smt = False  # True to resume SMT after FFT in same run

    # Parse arguments into dataclasses
    parser = HfArgumentParser((SMTModelArguments, FFTModelArguments, DataArguments, TrainingArguments, PhaseArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        smt_model_args, fft_model_args, data_args, training_args, phase_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        smt_model_args, fft_model_args, data_args, training_args, phase_args = parser.parse_args_into_dataclasses()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Store original values
    orig_learning_rate = training_args.learning_rate
    orig_weight_decay = training_args.weight_decay
    orig_deepspeed = training_args.deepspeed
    orig_max_steps = training_args.max_steps

    smt_model_args.full_ft_steps = fft_model_args.full_ft_steps

    # Check if DeepSpeed configs are valid
    if training_args.deepspeed:
        try:
            with open(training_args.deepspeed, 'r') as f:
                fft_deepspeed_config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading FFT DeepSpeed config file: {e}")
            sys.exit(1)

    if smt_model_args.smt_deepspeed:
        try:
            with open(smt_model_args.smt_deepspeed, 'r') as f:
                smt_deepspeed_config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading SMT DeepSpeed config file: {e}")
            sys.exit(1)

    # FFT PHASE
    if phase_args.fft_warmup:
        print("=" * 50)
        print("RUNNING FFT PHASE")
        print("=" * 50)

        # Configure for FFT phase
        training_args.deepspeed = fft_deepspeed_config
        training_args.max_steps = fft_model_args.full_ft_steps

        print(f"smt_model_args: {smt_model_args}")
        print(f"fft_model_args: {fft_model_args}")
        print(f"data_args: {data_args}")
        print(f"training_args: {training_args}")

        # Model
        from models import smt_create_and_prepare_model
        model, peft_config, tokenizer = smt_create_and_prepare_model(smt_model_args, fft_model_args, data_args)

        # Gradient checkpointing
        model.config.use_cache = not training_args.gradient_checkpointing
        training_args.gradient_checkpointing = training_args.gradient_checkpointing
        if training_args.gradient_checkpointing:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": fft_model_args.use_reentrant}

        train_dataset, eval_dataset, data_collator = create_local_datasets(
            tokenizer,
            data_args,
            training_args,
            apply_chat_template=smt_model_args.chat_template_format != "none",
        )

        # Trainer
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

        # start full fine-tuning
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        ffttrainer.train(resume_from_checkpoint=checkpoint)

        print("#" * 20)

        # obtain mlp and attention gradients
        mlp_grads = ffttrainer.mlp_warmup_grads
        attention_grads = ffttrainer.attention_warmup_grads

        ffttrainer.print_trainable_parameters()

        # Save gradients
        os.makedirs(training_args.output_dir, exist_ok=True)
        
        if smt_model_args.num_submatrix_mlp > 0:
            save_gradients(mlp_grads, "mlp_grad", training_args.output_dir)
        if smt_model_args.num_submatrix_attn > 0:
            save_gradients(attention_grads, "attn_grad", training_args.output_dir)

        # Save config and arguments for future SMT phase
        config_path = os.path.join(training_args.output_dir, "fft_config.json")
        config = {
            "smt_model_args": smt_model_args.__dict__,
            "fft_model_args": fft_model_args.__dict__,
            "data_args": data_args.__dict__,
            "training_args": {k: v for k, v in training_args.__dict__.items() if not k.startswith('_')}
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"FFT training completed. Gradients saved to {training_args.output_dir}")
        
        # Save FFT model 
        if ffttrainer.is_fsdp_enabled:
            ffttrainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        ffttrainer.save_model()
        
        # Clean up FFT resources
        del ffttrainer
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # If not resuming SMT in same run, exit here
        if not phase_args.resume_smt:
            print("FFT phase completed. Use --fft_warmup=False to run SMT phase.")
            return

        # Clean up distributed process group for FFT phase
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        
        # Need to completely exit and restart for SMT phase due to DeepSpeed limitations
        print("Cannot resume SMT in same run due to DeepSpeed limitations.")
        print("Please run again with --fft_warmup=False to perform SMT phase.")
        return

    # SMT PHASE
    else:
        print("=" * 50)
        print("RUNNING SMT PHASE")
        print("=" * 50)

        # Update training arguments for SMT phase
        training_args.learning_rate = smt_model_args.smt_learning_rate
        training_args.weight_decay = smt_model_args.smt_w_decay
        training_args.deepspeed = smt_deepspeed_config
        training_args.max_steps = smt_model_args.smt_steps if smt_model_args.smt_steps > 0 else -1

        print(f"smt_model_args: {smt_model_args}")
        print(f"fft_model_args: {fft_model_args}")
        print(f"data_args: {data_args}")
        print(f"training_args: {training_args}")

        # Load gradients from previous FFT phase
        mlp_grads = load_gradients(training_args.output_dir, "mlp_grad") if smt_model_args.num_submatrix_mlp > 0 else []
        attention_grads = load_gradients(training_args.output_dir, "attn_grad") if smt_model_args.num_submatrix_attn > 0 else []

        if not mlp_grads and not attention_grads:
            print("No gradients found. Please run with --fft_warmup=True first.")
            sys.exit(1)

        # Initialize distributed environment if needed
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend='nccl')
            print(f"Initialized process group: rank={dist.get_rank()}, world_size={dist.get_world_size()}")

        # Model
        from models import smt_create_and_prepare_model
        base_model, _, tokenizer = smt_create_and_prepare_model(smt_model_args, fft_model_args, data_args)
        model = SMTModel(smt_model_args, base_model, attention_grads, mlp_grads)

        train_dataset, eval_dataset, data_collator = create_local_datasets(
            tokenizer,
            data_args,
            training_args,
            apply_chat_template=smt_model_args.chat_template_format != "none",
        )

        # Initialize SMTTrainer
        smttrainer = SMTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            fft_steps=fft_model_args.full_ft_steps,
        )

        smttrainer.accelerator.print(f"{smttrainer.model}")
        smttrainer.model.print_trainable_parameters()

        # Train
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        smttrainer.train(resume_from_checkpoint=checkpoint)

        # Save final model
        if smttrainer.is_fsdp_enabled:
            smttrainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        smttrainer.save_model()
        print(f"SMT training completed. Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()