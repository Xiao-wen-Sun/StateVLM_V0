import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from statevlm.model.statevlm_base import StateVLM
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(0)
# export PYTHONPATH=/data/sun/StateVLM
# python statevlm/eval/inference_demo.py
model_name_or_path = "/data/sun/StateVLM/statevlm_emb/statevlm_model_emb_full_5000"
# model_name_or_path="/data/sun/StateVLM/statevlm_emb/statevlm_emb_7500_v4"
# model_name_or_path = "./statevlm_emb/statevlm_emb_2500_v2"
# model_name_or_path = "./statevlm_emb/statevlm_emb_8000"
# model_name_or_path = "./statevlm_emb/statevlm_emb_12000_v1"
# model_name_or_path = './statevlm_emb/statevlm_emb_12000'
# model_name_or_path = '/data/sun/StateVLM/MiniCPM-V-2_6'
# model_name_or_path = '/data/sun/StateVLM/outputs/statevlm_model_full_v1/checkpoint-10000'

# image_path1 = './datasets/train2014/COCO_train2014_000000577583.jpg'
# image_path2 = '/data/sun/datasets/combination_data/scene1.jpg'
#image_path2 = '/data/sun/20250523-TableDataset5Templates/template1/1000images-6step/images/template1-0to999_diffusion_lightning_pass-1_img__00006_.png'
#image_path2 = "/data/sun/20250523-TableDataset5Templates/template2/Template2/_final_images/image__00005_.png"
image_path2 = "/data/sun/20250523-TableDataset5Templates/template1/Template1/_final_images/img__00020_.png"

image = Image.open(image_path2).convert('RGB')
image_size = image.size
width, height = image_size[0], image_size[1]
# First round chat 
# question = "Where is the location of the plate in the image?"
question = "Where is the location of the clean bowl in the image?"
msgs = [{'role': 'user', 'content': [image, question]}]

# question = "Hello"
# msgs = [{'role': 'user', 'content': [question]}]

print(f"msgs: {msgs}")

# system_prompt = '''a chat between a human and an AI that understands visuals.'''

# system_prompt = '''a chat between a human and an AI that understands visuals. 
# In images, [x, y] denotes points: top-left is [0, 0], bottom-right is [1, 1].
# Increasing x moves right; y moves down. Bounding box: [x1, y1, x2, y2] where (x1, y1) is top-left, (x2, y2) is bottom-right. Follow instructions.
# '''
system_prompt = ''
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
if "MiniCPM-V" in model_name_or_path:
    model = AutoModel.from_pretrained(
                                model_name_or_path, 
                                trust_remote_code=True,
                                attn_implementation='sdpa', 
                                torch_dtype=torch.bfloat16
                            )
    model = model.eval().cuda()
    answer = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )
    print(f"answer: {answer}")

else:
    model = StateVLM.from_pretrained(
                                    model_name_or_path, 
                                    trust_remote_code=True,
                                    attn_implementation='sdpa', 
                                    torch_dtype=torch.bfloat16
                                ) # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()    
    
    # print(f"model: {model}")
    
    if "statevlm_seq" in model_name_or_path:
        answer = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
        )
        print(f"answer: {answer}")
    else:
        print("inference for statevlm with bounding box generation")
        location, answer = model.chat_emb(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
        )
        
        print(f"location: {location}, answer: {answer}")
        bbox = [location[0][0] * width, location[0][1] * height, location[0][2] * width, location[0][3] * height]
        print(f"bbox: {str(bbox)}")