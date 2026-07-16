from peft import PeftModel
from transformers import AutoTokenizer, AutoModel
from statevlm.model.statevlm_base import StateVLM
import torch
from PIL import Image
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_type="./statevlm_emb"   # or openbmb/MiniCPM-Llama3-V-2_5 , openbmb/MiniCPM-V-2
#path_to_adapter="/data/sun/StateVLM/output/statevlm_model_lora/checkpoint-10"

path_to_adapter="./outputs/output_lora_emb_v1/checkpoint-2500"
model =  StateVLM.from_pretrained(
        model_type,
        trust_remote_code=True
        )
#print(f"Auto model1: {model}")

print("--------------------------------------------------------")
print("--------------------------------------------------------")
print("--------------------------------------------------------")

lora_model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).eval().cuda()

#print(f"Peft model: {lora_model}")

image_path2 = "/data/sun/statevlm_annotation/images/test_images/image_00094.png"
image = Image.open(image_path2).convert('RGB')
image_size = image.size
width, height = image_size[0], image_size[1]
# First round chat 
# question = "Where is the location of the plate in the image?"
question = "pass me the dirty plate"
msgs = [{'role': 'user', 'content': [image, question]}]

# question = "Hello"
# msgs = [{'role': 'user', 'content': [question]}]

print(f"msgs: {msgs}")

system_prompt = '''a chat between a human and an AI that understands visuals.'''

# system_prompt = '''a chat between a human and an AI that understands visuals. 
# In images, [x, y] denotes points: top-left is [0, 0], bottom-right is [1, 1].
# Increasing x moves right; y moves down. Bounding box: [x1, y1, x2, y2] where (x1, y1) is top-left, (x2, y2) is bottom-right. Follow instructions.
# '''
# system_prompt = ''
tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)


answer = lora_model.chat(
image=None,
msgs=msgs,
tokenizer=tokenizer,
system_prompt=system_prompt,
)
print(f"answer: {answer}")


       