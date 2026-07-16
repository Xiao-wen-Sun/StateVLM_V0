import cv2
import os
import json

# REC_refcoco_testa = '/data/sun/StateVLM/datasets/refcoco3_annotations/REC_ref3_train.jsonl'
REC_refcoco_testa = '/data/sun/StateVLM/outputs/results1.json'
image_dir = '/data/sun/StateVLM/datasets/'
# output_dir = '/data/sun/StateVLM/datasets/coco_train/'
output_dir = '/data/sun/StateVLM/datasets/coco_train2/'
# with open(REC_refcoco_testa, "r") as f:
#         lines = f.readlines()
# Check if the file exists
if os.path.exists(REC_refcoco_testa):
    # Load existing data
    with open(REC_refcoco_testa, "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            data = []
else:
    # If file doesn't exist, start with an empty list
    data = []
for i, line in enumerate(data):
    image_id = line["image_id"][0]
    width = line["image_size"][0][0]
    height = line["image_size"][0][1]
    ground_truth = line["ground_truth"]
    predict_location = line["predict_location"]
    image_path = os.path.join(image_dir, image_id)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Couldn't read image {image_name}")
        continue
    print(f"ground_truth: {ground_truth[0][0]}")
    print(f"predict_location: {predict_location[0]}")
    x_min1 = int(ground_truth[0][0]*width)
    y_min1 = int(ground_truth[0][1]*height)
    x_max1 = int(ground_truth[0][2]*width)
    y_max1 = int(ground_truth[0][3]*height)
    
    x_min2 = int(predict_location[0][0]*width)
    y_min2 = int(predict_location[0][1]*height)
    x_max2 = int(predict_location[0][2]*width)
    y_max2 = int(predict_location[0][3]*height)
    
    print(f"x_min1: {x_min1}, y_min1: {y_min1}, x_max1: {x_max1}, y_max1: {y_max1}")
    print(f"x_min2: {x_min2}, y_min2: {y_min2}, x_max2: {x_max2}, y_max2: {y_max2}")

    cv2.rectangle(image, (x_min1, y_min1), (x_max1, y_max1), (0, 255, 0), 2)  # Green box
    
    cv2.rectangle(image, (x_min2, y_min2), (x_max2, y_max2), (0, 0, 255), 2)  # Red box
   
   
   
    image_name = str(i) + image_id.split('/')[-1]
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")
    if i > 200:
        break
