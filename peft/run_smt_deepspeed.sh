export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=True
export CUDA_LAUNCH_BLOCKING=1
ulimit -n 65536
ulimit -u 65536

WANDB__SERVICE_WAIT=300
deepspeed --include=localhost:0,1,2,3 --master_port 60000 train_smt.py \
--seed 100 \
--model_name_or_path "/data/ydh/models/Llama-2-7b-hf" \
--chat_template_format None \
--use_peft_smt True \
--smt_dropout 0.0 \
--smt_offload False \
--num_submatrix_mlp 0 \
--num_submatrix_attn 890 \
--selection_strategy "no_restriction" \
--calculation_strategy "mean_abs" \
--smt_learning_rate 1e-4 \
--smt_w_decay 0.0 \
--target_modules ['q_proj', 'k_proj', 'v_proj'] \
--dataset_name_or_path "/home/ydh/workspace/Sparse_Matrix_Tuning/data/commonsense_170k.json" \
--eval_set_size 120 \
--add_special_tokens False \
--append_concat_token False \
--packing False \
--max_seq_length 2048 \
--num_train_epochs 3 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--eval_steps 30 \
--eval_strategy "epoch" \
--save_strategy "epoch" \
--hub_private_repo False \
--hub_strategy "every_save" \
--bf16 True \
--learning_rate 9.65e-6 \
--lr_scheduler_type "linear" \
--weight_decay 0.0 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "deepspeed-smt" \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 1 \
--gradient_checkpointing True \
--use_reentrant False \
--full_ft_steps 10 \
--fft_offload True \
--fft_zero_stage 2 \
--load_best_model_at_end True \
--metric_for_best_model ""eval_loss"" \
--smt_deepspeed "/home/ydh/workspace/Sparse_Matrix_Tuning/peft/configs/smt_deepspeed_config.json" \
--deepspeed "/home/ydh/workspace/Sparse_Matrix_Tuning/peft/configs/fft_deepspeed_config.json"






