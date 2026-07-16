#!/bin/bash

GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6008



MODEL="../model_zoo/statevlm_seq_5000"
DATA="../datasets/OSAR/annotations/state_train_dia.json"
EVAL_DATA="../datasets/OSAR/annotations/state_val_dia.json"

IMAGE_PATH="../datasets/OSAR/images/train_images/"
LLM_TYPE="qwen2"
MODEL_MAX_Length=2048 

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun $DISTRIBUTED_ARGS ../statevlm/train/train.py \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --image_path $IMAGE_PATH \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 false \
    --bf16_full_eval false \
    --fp16 true \
    --fp16_full_eval true \
    --do_train \
    --do_eval \
    --tune_resampler true\
    --tune_vision true \
    --tune_llm_model false \
    --tune_llm_lm_head true \
    --use_lora true \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 9 \
    --max_steps 10000 \
    --eval_steps 2500 \
    --output_dir ../outputs/statevlm_seq_5000_lora_adapter \
    --logging_dir ../runs/statevlm_seq_5000_lora_adapter \
    --logging_strategy "steps" \
    --per_device_train_batch_size 48 \
    --per_device_eval_batch_size 48 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 2500 \
    --save_total_limit 10 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed ds_config_zero2.json \
    --report_to "tensorboard" # wandb
