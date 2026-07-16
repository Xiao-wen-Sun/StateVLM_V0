import json
import os
def format_instance(lines, formatted_data):
    for idx, line in enumerate(lines):
        data = json.loads(line)
        experssion = data["expression"].lower()
        img_width = data["width"]
        img_height = data["height"]
        # nor_x1 = round(data["bbox"][0]/img_width, 5)
        # nor_y1 = round(data["bbox"][1]/img_height, 5)
        # nor_x2 = round(data["bbox"][2]/img_width, 5)
        # nor_y2 = round(data["bbox"][3]/img_height, 5)
        nor_x1 = data["bbox"][0]/img_width
        nor_y1 = data["bbox"][1]/img_height
        nor_x2 = data["bbox"][2]/img_width
        nor_y2 = data["bbox"][3]/img_height
        
        bbox_str = str(nor_x1) +", "+ str(nor_y1) +", "+ str(nor_x2) +", "+ str(nor_y2)
        bbox_num = [nor_x1, nor_y1, nor_x2, nor_y2]
        if experssion.startswith("a"):
            formatted_entry = {
                "id": str(idx),
                "image": f"train2014/{data['img_path']}",
                "conversations": [
                    # {
                    #     "role": "system",
                    #     "content": "A chat between a human and an AI that understands visuals. Follow instructions."
                    # },
                    {
                        "role": "user",
                        "content": "<image>\nShow me " + experssion + " in the image?",
                    },
                    # {
                    #     "role": "assistant",
                    #     "content": "Here is the location of the " + experssion + " in the image.",
                    #     "bbox": bbox_num
                    # }
                    {
                        "role": "assistant",
                        "content": "response: Here is " + experssion + " in the image. <box>" + bbox_str + "</box>",
                        "bbox": bbox_num
                    }
                ]
            }
        elif experssion.startswith("the "):
            formatted_entry = {
                "id": str(idx),
                "image": f"train2014/{data['img_path']}",
                "conversations": [
                    # {
                    #     "role": "system",
                    #     "content": "A chat between a human and an AI that understands visuals. Follow instructions."
                    # },
                    {
                        "role": "user",
                        "content": "<image>\nWhere is " + experssion + " in the image?",
                    },
                    # {
                    #     "role": "assistant",
                    #     "content": "Here is the location of the " + experssion + " in the image.",
                    #     "bbox": bbox_num
                    # }
                    {
                        "role": "assistant",
                        "content": "response: Here is " + experssion + " in the image. <box>" + bbox_str + "</box>",
                        "bbox": bbox_num
                    }
                ]
            }
        else:         
            formatted_entry = {
                "id": str(idx),
                "image": f"train2014/{data['img_path']}",
                "conversations": [
                    # {
                    #     "role": "system",
                    #     "content": "A chat between a human and an AI that understands visuals. Follow instructions."
                    # },
                    {
                        "role": "user",
                        "content": "<image>\nWhere is the location of the " + experssion + " in the image?",
                    },
                    # {
                    #     "role": "assistant",
                    #     "content": "Here is the location of the " + experssion + " in the image.",
                    #     "bbox": bbox_num
                    # }
                    {
                        "role": "assistant",
                        "content": "response: Here is the location of the " + experssion + " in the image. <box>" + bbox_str + "</box>",
                        "bbox": bbox_num
                    }
                ]
            }
        formatted_data.append(formatted_entry)
    return formatted_data

def format_refcoco_to_dia(REC_refcoco, REC_refcoco_dia):
    formatted_data = []
    with open(REC_refcoco, "r") as f:
        lines = f.readlines()
    formatted_data = format_instance(lines, formatted_data)
    with open(REC_refcoco_dia, "w") as f:
        json.dump(formatted_data, f, indent=4)
    print(f"Formatted data saved to {REC_refcoco_dia}")

def format_val_to_one(val_names, REC_ref3_val_dia, current_path):
    formatted_data = []
    all_lines = []
    for val_name in val_names:
        print(f"Reading {val_name}.jsonl")
        with open(f"{current_path}/datasets/refcoco3_annotations/{val_name}.jsonl", "r") as f:
            lines = f.readlines()
            all_lines.extend(lines)
    print(f"Read {len(all_lines)} lines")
    formatted_data = format_instance(all_lines, formatted_data)
    with open(REC_ref3_val_dia, "w") as f:
        json.dump(formatted_data, f, indent=4)
    print(f"Formatted data saved to {REC_ref3_val_dia}")
    
def format_single_data(test_dataset_names, save_root, current_path):
    for test_name in test_dataset_names:
        formatted_data = []
        print(f"Reading {test_name}.jsonl")
        with open(f"{current_path}/datasets/refcoco3_annotations/{test_name}.jsonl", "r") as f:
            lines = f.readlines()
        formatted_data = format_instance(lines, formatted_data)
        
        save_path = save_root + test_name +"_dia.json"
        with open(save_path, "w") as f:
            json.dump(formatted_data, f, indent=4)
        print(f"Formatted data saved to {save_path}")
 
if __name__ == "__main__":
    current_path = os.getcwd()
    print("Current Path:", current_path)

    REC_ref3_train = current_path + "/datasets/refcoco3_annotations/REC_ref3_train.jsonl"  
    REC_ref3_train_dia = current_path +"/datasets/training/REC_ref3_train_dia.json"  

    val_dataset_names = ["REC_refcoco_unc_val", "REC_refcoco+_unc_val", "REC_refcocog_umd_val"]
    REC_ref3_val_dia = current_path + "/datasets/training/REC_ref3_val_dia.json"  
    
    test_dataset_names = ["REC_refcocog_umd_test", "REC_refcoco+_unc_testA", "REC_refcoco+_unc_testB", "REC_refcoco_unc_testA", "REC_refcoco_unc_testB"]
    save_root = current_path + "/datasets/training/" 
    
    format_refcoco_to_dia(REC_ref3_train, REC_ref3_train_dia)
    format_val_to_one(val_dataset_names, REC_ref3_val_dia, current_path)
    
    # format_single_data(val_dataset_names, save_root, current_path)
    # format_single_data(test_dataset_names, save_root, current_path)
    