#!/bin/bash

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6066

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MODEL="../../model_zoo/statevlm_emb_20000"
TEST_DATA="../../datasets/REC/refcoco3_dia_annotations/REC_refcoco_unc_testB_dia_eva.json"
IMAGE_PATH="../../datasets/REC/"
LLM_TYPE="qwen2" 
MODEL_MAX_Length=2048 

CUDA_VISIBLE_DEVICES=0 torchrun $DISTRIBUTED_ARGS ../../statevlm/eval/model_refcoco.py \
  --test_data_path $TEST_DATA \
  --image_path $IMAGE_PATH \
  --model_embedding False \
  --model_name_or_path $MODEL \
