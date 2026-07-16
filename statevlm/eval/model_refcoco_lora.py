# Copyright Xiaowen Sun 2024.12
import argparse
from matplotlib import text
import tqdm
import logging
import re
import copy
import ast
from typing import Optional
import torch
import torch.utils.data as torch_data
from dataclasses import dataclass, field
from transformers import AutoModel, AutoTokenizer, AutoProcessor, HfArgumentParser             
from statevlm.model.image_processing_statevlm import StateVLMImageProcessor
from statevlm.model.statevlm_base import StateVLM
from statevlm.data.dataset import data_collator_eval, make_supervised_evaluation_data_module
from statevlm.utils.box_ops import generalized_box_iou, box_iou
from statevlm.utils.builder import build_transform
import json
from datetime import datetime
from peft import PeftModel
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(10)

logger = logging.getLogger(__name__)

VOCAB_IMAGE_W = 1000
VOCAB_IMAGE_H = 1000
DEFAULT_REGION_FEA_TOKEN = "<region_fea>"


def resize_bbox(box, image_w=None, image_h=None):
    ratio_w = image_w * 1.0 / VOCAB_IMAGE_W
    ratio_h = image_h * 1.0 / VOCAB_IMAGE_H

    new_box = [int(box[0] * ratio_w), int(box[1] * ratio_h), \
               int(box[2] * ratio_w), int(box[3] * ratio_h)]
    return new_box

def find_bbox_template(text, img_w, img_h):
    pattern = r'\[(\d+), (\d+), (\d+), (\d+)\]'
    matches = re.findall(pattern, text)
    new_bboxes = []
    old_bboxes = []
    for match in matches:
        x1, y1, x2, y2 = map(int, match)
        new_box = resize_bbox([x1, y1, x2, y2], img_w, img_h)
        new_bboxes.append(new_box)
        old_bboxes.append([x1, y1, x2, y2])
    
    set_old_bboxes = sorted(set(map(tuple, old_bboxes)), key=list(map(tuple, old_bboxes)).index)
    list_old_bboxes = list(map(list, set_old_bboxes))

    set_bboxes = sorted(set(map(tuple, new_bboxes)), key=list(map(tuple, new_bboxes)).index)
    list_bboxes = list(map(list, set_bboxes))

    for i in range(len(list_bboxes)):
        x1, y1, x2, y2 = list_old_bboxes[i]
        text = text.replace('[{}, {}, {}, {}]'.format(x1, y1, x2, y2), '')
    
    if text.endswith(" ."):
        text = text[:-2]
    split_text = text.split(" . ")
    entities = [item.strip() for item in split_text if item.strip() != '']

    return entities, list_bboxes

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    path_to_adapter: Optional[str] = field(default="")
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

def init_model(model_name_or_path, path_to_adapter):
    print(f"model_name_or_path: {model_name_or_path}")
    print(f"path_to_adapter: {path_to_adapter}")
    if "MiniCPM-V" in model_name_or_path:
        model = AutoModel.from_pretrained(
                                model_name_or_path, 
                                trust_remote_code=True,
                                attn_implementation='sdpa', 
                                torch_dtype=torch.bfloat16
                            )
        # processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
         
    else:
        model =  StateVLM.from_pretrained(
                                model_name_or_path,
                                trust_remote_code=True
                    )
          
        lora_model = PeftModel.from_pretrained(
                        model,
                        path_to_adapter,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16
                    ).eval().cuda()
       
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    return lora_model, tokenizer

def is_valid_structure_safe(data):
    if isinstance(data, list) and len(data) == 1:
        inner = data[0]
        if isinstance(inner, str) and inner.startswith('[') and inner.endswith(']'):
            try:
                numbers = ast.literal_eval(inner)
                return (
                    isinstance(numbers, list) and 
                    len(numbers) == 4 and 
                    all(isinstance(num, (int, float)) for num in numbers)
                )
            except:
                return False
    return False


def recursive_to_cuda(item):
                if isinstance(item, torch.Tensor):
                    return item.cuda()
                elif isinstance(item, list):
                    return [recursive_to_cuda(sub_item) for sub_item in item]
                else:
                    return item
    
def eval_model_refexp(args):
    # set up parameters
    sampling=True
    min_new_tokens=0
    use_image_id=None
    max_slice_nums = 9
    model_max_length=2048
    llm_type='qwen2' # qwen2
    
    global local_rank
    parser = HfArgumentParser((ModelArguments, DataArguments))
    (model_args, data_args) = parser.parse_args_into_dataclasses()
    print(f"data_args[0]: {data_args}")
    print(f"model_args: {model_args}")
    # loading model
    model, tokenizer = init_model(model_name_or_path=model_args.model_name_or_path, path_to_adapter=model_args.path_to_adapter)
    # loading Data
    if hasattr(model.config, "slice_config"):
        model.config.slice_config.max_slice_nums = max_slice_nums
        slice_config = model.config.slice_config.to_dict()
    else:
        model.config.max_slice_nums = max_slice_nums
        slice_config = model.config.to_dict()

    if hasattr(model.config, "batch_vision_input"):
        batch_vision = model.config.batch_vision_input
    else:
        batch_vision = False

    transform_func = build_transform()
    data_module = make_supervised_evaluation_data_module(
                                        tokenizer=tokenizer,
                                        data_args=data_args,
                                        transform=transform_func,
                                        data_collator_eval=data_collator_eval,
                                        slice_config=slice_config,
                                        llm_type=llm_type,
                                        patch_size=model.config.patch_size,
                                        query_nums=model.config.query_num,
                                        batch_vision=batch_vision,
                                        max_length=model_max_length,
                                    )
    dataloader = torch_data.DataLoader(data_module['test_dataset'], shuffle=True, batch_size=1, collate_fn=data_module['data_collator_eval'])

    if sampling:
        generation_config = {
            "top_p": 0.8,
            "top_k": 100,
            "temperature": 0.7,
            "do_sample": True,
            "repetition_penalty": 1.05
        }
    else:
        generation_config = {
            "num_beams": 3,
            "repetition_penalty": 1.2,
        }
            
    if min_new_tokens > 0:
        generation_config['min_new_tokens'] = min_new_tokens

    # generation_config.update(
    #     (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
    # )
   # Create a unique output file for each run.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = os.path.join("outputs", f"results_{run_timestamp}.json")
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    
    # do inference
    with torch.inference_mode():
        correct = 0
        failed = 0
        exception = 0
        for batch in tqdm.tqdm(dataloader, f'Inference'):
            # Reset per-sample values so previous iterations do not leak into current logging.
            iou = None
            predict_location = None
            coordinates = None
            image_id = batch["image_id"]
            image_size = batch["image_size"]
            inputs = copy.deepcopy(batch)  
            inputs.pop('position_ids')
            inputs.pop('text_labels')
            inputs.pop('loc_labels')
            inputs.pop('image_id')
            inputs.pop('image_size')
            inputs = {
                key: recursive_to_cuda(value) if isinstance(value, (torch.Tensor, list)) else value
                for key, value in inputs.items()
            }
            target_location = batch["loc_labels"]
            # if model_args.model_embedding:    
            #     locations, texts = model.generate_emb(
            #             **inputs,
            #             tokenizer=tokenizer,
            #             max_new_tokens=2048,
            #             vision_hidden_states=None,
            #             stream=False,
            #             **generation_config,
            #     )
                
            #     print(f"texts: {texts}")
            #     print(f"locations: {locations}")
            #     # match = re.search(r"<box>([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)</box>", texts[0])
            #     match = re.search(r'<box>(.*?)</box>', texts[0])
                
            #     if match:
            #         try:
            #             # coordinates = [float(match.group(i)) for i in range(1, 5)]
            #             coordinates = [float(x.strip()) for x in match.group(1).split(',')]                      
            #             print(f"coordinates: {coordinates}")
            #             print(f"target_location: {target_location}")
            #             if len(coordinates) > 0:
            #                 predict_location = torch.tensor([coordinates], dtype=torch.float16).to(torch.float16) 
            #                 if predict_location.shape[-1] == 4:
            #                     try:
            #                         iou, union = box_iou(predict_location, target_location)
            #                         print(f"iou: {iou}")
            #                         if iou > 0.5:
            #                             correct += 1
            #                         else:
            #                             failed += 1
            #                     except AssertionError as e:
            #                         exception += 1
            #                         print("The coordinates value are not correct.")       
            #                 else:
            #                     exception += 1
            #                     print("No correct format coordinates found.")                            
            #             else:
            #                 exception += 1
            #                 print("No coordinates found.")      
            #         except ValueError as e:
            #             print(f"Error converting to float: {e}")
            #             exception += 1 
            #     else:
            #         exception += 1
            #         print("No match found.")  
            #     print(f"Correct: {correct}, Failed: {failed}, Exception: {exception}, Accuracy: {correct / (correct + failed + exception)}")

            # else:
            texts = model.generate(
                    **inputs,
                    tokenizer=tokenizer,
                    max_new_tokens=2048,
                    vision_hidden_states=None,
                    decode_text=True,
                    stream=False,
                    **generation_config,
            )
            print("/n")
            print(f"texts: {texts}")
            #match = re.search(r"<box>([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)</box>", texts[0])
            match = re.search(r'<box>(.*?)</box>', texts[0])
            
            if match:
                try:
                    # coordinates = [float(match.group(i)) for i in range(1, 5)]
                    coordinates = [float(x.strip()) for x in match.group(1).split(',')]
                    print(f"coordinates: {coordinates}")
                    print(f"target_location: {target_location}")
                    if len(coordinates) > 0:
                        predict_location = torch.tensor([coordinates], dtype=torch.float16).to(torch.float16) 
                        if predict_location.shape[-1] == 4:
                            try:
                                iou, union = box_iou(predict_location, target_location)
                                print(f"iou: {iou}")
                                if iou > 0.5:
                                    correct += 1
                                else:
                                    failed += 1                                    
                            except AssertionError as e:
                                exception += 1
                                print("The coordinates value are not correct.")       
                        else:
                            exception += 1
                            print("No correct format coordinates found.")                            
                    else:
                        exception += 1
                        print("No coordinates found.")      
                except ValueError as e:
                    print(f"Error converting to float: {e}")
                    exception += 1 
            else:
                exception += 1
                print("No match found.")  
            print(f"Correct: {correct}, Failed: {failed}, Exception: {exception}, Accuracy: {correct / (correct + failed + exception)}")
                
            new_entry = {
                "image_id": image_id,
                "image_size": image_size,
                "text_prediction": texts,
                "coordinates": coordinates,
                "ground_truth": target_location.tolist(),
                "predict_location": predict_location.tolist() if predict_location is not None else None,
                "iou": iou.item() if iou is not None else None,
                "correct": correct,
                "failed": failed,
                "exception": exception
            }
            # Check if the file exists
            if os.path.exists(json_file):
                # Load existing data
                with open(json_file, "r") as file:
                    try:
                        data = json.load(file)
                    except json.JSONDecodeError:
                        data = []
            else:
                # If file doesn't exist, start with an empty list
                data = []
            # Append the new entry
            data.append(new_entry)
            # Save back to the JSON file
            with open(json_file, "w") as file:
                json.dump(data, file, indent=4)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process arguments for model evaluation.")

    parser.add_argument("--test_data_path", default="/data/sun/statevlm_annotation/REC_state/rec_state_annotations/rec_state_test_dia.json", type=str, help="Path to the test data JSON file.")
    parser.add_argument("--image_path", default="/data/sun/statevlm_annotation/images/test_images/",type=str, help="Path to the image directory.")
    parser.add_argument("--model_embedding", default=False, help="Embedding Paradigm or not.")
    
    parser.add_argument("--model_name_or_path", default="/data/sun/StateVLM/statevlm_emb", help="model path.")
    parser.add_argument("--path_to_adapter", default="/data/sun/StateVLM/outputs/output_lora_emb_v1/checkpoint-2500", help="model path.")
    
    
    # Parse the arguments
    args = parser.parse_args()
    
    print(f"args: {args}")

    eval_model_refexp(args)