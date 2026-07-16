#!/bin/bash
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=8322

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MODEL="../../model_zoo/statevlm_seq_5000"
path_to_adapter="../../model_zoo/statevlm_seq_5000_lora_adapter"


TEST_DATA="../../datasets/OSAR/annotations/rec_state_test_dia.json"
IMAGE_PATH="../../datasets/OSAR/images/test_images/"

LLM_TYPE="qwen2" 
MODEL_MAX_Length=2048 

CUDA_VISIBLE_DEVICES=1 torchrun $DISTRIBUTED_ARGS ../../statevlm/eval/model_refcoco_lora.py \
  --test_data_path $TEST_DATA \
  --image_path $IMAGE_PATH \
  --model_embedding false \
  --model_name_or_path $MODEL \
  --path_to_adapter $path_to_adapter \
