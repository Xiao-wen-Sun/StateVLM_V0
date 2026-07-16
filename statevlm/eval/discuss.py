from collections import Counter
import json
import os
import math
import re

# file_path = "/data/sun/StateVLM/script_state/emb/outputs/results_20260327_194958_rec_emb.json"
# file_path = "/data/sun/StateVLM/script_state/seq/outputs/results_20260327_115657_rec_seq.json"

# file_path = "/data/sun/StateVLM/script_state/emb/outputs/results_20260327_194856_aff_emb.json"

file_path = "/data/sun/StateVLM/script_state/seq/outputs/results_20260327_115720_aff_seq.json"


# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

# Load the JSON file
with open(file_path, 'r', encoding='utf-8') as file:
    raw_text = file.read()


data = json.loads(raw_text)
print(f"Loaded data from {file_path}")

if not isinstance(data, list):
    raise ValueError("The JSON file must contain a list of objects.")

# Extract IoU values from the list
iou_values = []
invalid_iou_count = 0
for item in data:
    if not isinstance(item, dict) or "iou" not in item:
        continue

    iou = item["iou"]
    if isinstance(iou, (int, float)) and math.isfinite(iou):
        iou_values.append(float(iou))
    else:
        invalid_iou_count += 1
        

# Define IoU ranges
ranges = [(i / 10, (i + 1) / 10) for i in range(10)]

# print(ranges)

# Calculate the distribution
distribution = Counter()
for iou in iou_values:
    for start, end in ranges:
        if start <= iou < end:
            distribution[f"{start:.1f}-{end:.1f}"] += 1
            break
        if iou == 1.0 and end == 1.0:
            distribution[f"{start:.1f}-{end:.1f}"] += 1
            break

# Print the distribution
for range_label, count in distribution.items():
    print(f"{range_label}: {count}")

print(f"Valid IoU count: {len(iou_values)}")
print(f"Invalid/ignored IoU count: {invalid_iou_count}")