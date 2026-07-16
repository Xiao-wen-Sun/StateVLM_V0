import glob
import json
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Union, Literal, Tuple
from types import MethodType
import torch
import transformers
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers import AutoTokenizer
from transformers.integrations import deepspeed
from statevlm.data.dataset import make_supervised_data_module, data_collator
from statevlm.model.statevlm_base import StateVLM
from statevlm.utils.builder import build_transform
from trainer import StateVLMTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    model_embedding: Optional[bool] = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    test_data_path: str = field(
        default=None, metadata={"help": "Path to the test data."}
    )
    image_path: str = field(
        default=None, metadata={"help": "Path to the image data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tune_resampler: Optional[bool] = field(default=True)
    tune_vision: Optional[bool] = field(default=True)
    tune_llm_model: Optional[bool] = field(default=True)
    tune_llm_location_decoder: Optional[bool] = field(default=True)
    tune_llm_lm_head: Optional[bool] = field(default=True)
    llm_type: str = field(default="qwen2")
    use_lora: Optional[bool] = field(default=False)
    max_slice_nums: Optional[int] = field(default=9)


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer.save_model(output_dir,)


# def build_transform():
#     IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5) # timm.data.IMAGENET_INCEPTION_MEAN
#     IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD
#     return transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
#                 ),
#             ]
#         )

def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
        
    return {'Total': all_param, 'Trainable': trainable_params}


local_rank = 0

def init_model(model_args, data_args, training_args, lora_args):
    if getattr(training_args, "deepspeed", None) : 
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = None
    
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )
    # '/data/sun/StateVLM/MiniCPM-V-2_6'
    print(f"the pre-trained model is from {model_args.model_name_or_path}")
    model = StateVLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        device_map=device_map,
    )

    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.model_name_or_path, trust_remote_code=True, use_fast=False,
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    
    if not training_args.tune_vision:
        model.vpm.requires_grad_(False)
        
    if not training_args.tune_resampler:
        model.resampler.requires_grad_(False)
        
    if not training_args.tune_llm_model:
        model.llm.model.requires_grad_(False)
    if not training_args.tune_llm_location_decoder:
        model.llm.loc_decoder.requires_grad_(False)
    if not training_args.tune_llm_lm_head:
        model.llm.lm_head.requires_grad_(False)
        
    if training_args.use_lora:
        if training_args.use_lora and training_args.tune_llm_model:
            raise ValueError("The model cannot simultaneously adjust LLM parameters and apply LoRA.")
            
        rank0_print("Currently using LoRA for fine-tuning the MiniCPM-V model for StateVLM.")
        for name, param in model.llm.named_parameters():
            param.requires_grad = False
        modules_to_save = ['embed_tokens','resampler']
        if training_args.tune_vision:
            modules_to_save.append('vpm')
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            layers_to_transform=lora_args.lora_layers_to_transform,
            modules_to_save=modules_to_save,
        )
        if not hasattr(model, 'get_input_embeddings'):
            def get_input_embeddings(self):
                return self.llm.get_input_embeddings()
            model.get_input_embeddings = MethodType(get_input_embeddings, model)
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    rank0_print(get_parameter_number(model))
    llm_type = training_args.llm_type    
    rank0_print(f'llm_type={llm_type}')

    
    # Load data
    if hasattr(model.config, "slice_config"):
        model.config.slice_config.max_slice_nums = training_args.max_slice_nums
        slice_config = model.config.slice_config.to_dict()
    else:
        model.config.max_slice_nums = training_args.max_slice_nums
        slice_config = model.config.to_dict()

    if hasattr(model.config, "batch_vision_input"):
        batch_vision = model.config.batch_vision_input
    else:
        batch_vision = False

    transform_func = build_transform()
    data_module = make_supervised_data_module(
                                                tokenizer=tokenizer,
                                                data_args=data_args,
                                                transform=transform_func,
                                                data_collator=data_collator,
                                                slice_config=slice_config,
                                                llm_type=llm_type,
                                                patch_size=model.config.patch_size,
                                                query_nums=model.config.query_num,
                                                batch_vision=batch_vision,
                                                max_length=training_args.model_max_length,
                                            )
    
    return model, data_module, tokenizer

def train():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    
    ( model_args, data_args, training_args, lora_args) = parser.parse_args_into_dataclasses()
    
    # init_model(model_args, data_args, training_args, lora_args)
    
    model, data_module, tokenizer = init_model(model_args, data_args, training_args, lora_args)
    
    # rank0_print("=" * 80)
    # rank0_print("Final Model Architecture:")
    # rank0_print(model)
    # rank0_print("\nFinal Model Config:")
    # rank0_print(model.config)
    # rank0_print("=" * 80)
    
    training_args.gradient_checkpointing_kwargs={"use_reentrant":False}
    trainer = StateVLMTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    
    # if training_args.do_train:
    train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)  
    
    trainer.save_state()
    print(f"Saving model to {training_args.output_dir}")
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        bias=lora_args.lora_bias)
    print("The training process is end.")



if __name__ == "__main__":
    train()

