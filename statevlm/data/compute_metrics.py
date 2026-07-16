# import sys
# import logging
# from typing import Dict, Any, Sequence

# from transformers import EvalPrediction

# from statevlm.utils.common import decode_generate_ids

# from functools import partial
# from typing import Callable, Dict, Tuple, Any, Optional

# from torch.utils.data import Dataset
# from transformers import EvalPrediction, TrainingArguments

# from .root import DATASETS, METRICS, TRANSFORMS, FUNCTIONS
# from .single_image_convsation import WRAPPER_DATASET
# from .single_image_interactive import SingleImageInteractive
# from ..conversation import get_conv_template
# from .utils import init_ceph_client_if_needed

# import re
# import sys
# import logging
# import typing
# from typing import List, Dict, Any, Tuple, Union

# from ..utils.transform import norm_box_xyxy, norm_point_xyxy

# from ..root import (
#     FUNCTIONS,
#     BaseTargetProcessFunc,
#     BOXES_PLACEHOLDER,
#     BOXES_PROCESSOR,
#     POINTS_PLACEHOLDER,
# )

# from ...utils import smart_tokenizer_and_embedding_resize

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     handlers=[logging.StreamHandler(sys.stdout), ],
# )

# Box = List[Union[float, int]]
# Boxes = List[Box]
# BoxesSeq = List[Boxes]

# DatasetDict = Dict[str, Dataset]
# ComputeMetrics = Callable[[EvalPrediction], Dict]

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     handlers=[logging.StreamHandler(sys.stdout), ],
# )

# class BaseComputeMetrics:
#     def __init__(self, preprocessor: Dict[str, Any]):
#         self.preprocessor = preprocessor
#         self.tokenizer = self.preprocessor['text']

#     def __call__(self, eval_preds: EvalPrediction) -> Dict[str, Any]:
#         preds, targets = eval_preds
#         logger.warning(f"preds shape: {preds.shape}. targets shape: {targets.shape}")
#         preds = decode_generate_ids(self.tokenizer, preds)
#         targets = decode_generate_ids(self.tokenizer, targets)
#         assert len(preds) == len(targets)
#         return self.calculate_metric(preds, targets)

#     def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
#         correct = 0
#         failed = 0
#         target_failed = 0
#         for pred, target in zip(preds, targets):
#             extract_pred = self.extract_ans(pred)
#             extract_target = self.extract_ans(target)
#             if extract_target is None:
#                 target_failed += 1
#                 logger.warning(f"failed to extract ans from target. maybe the response string is truncated: {target}.")
#                 continue
#             if extract_pred is None:
#                 failed += 1
#             if extract_pred == extract_target:
#                 correct += 1
#         return {
#             'accuracy': 1.0 * correct / len(targets),
#             'target_failed': target_failed,
#             'failed': failed,
#         }

#     def extract_ans(self, string: str):
#         raise NotImplementedError

# class BoxFormatter:
#     def __init__(self, bboxes_token=BOXES_PLACEHOLDER, points_token=POINTS_PLACEHOLDER):
#         self.bboxes_token = bboxes_token
#         self.points_token = points_token
#         # normally the bboxes_token_pat is the same as bboxes_token if u not use some weird token
#         self.bboxes_token_pat = re.compile(bboxes_token)
#         self.points_token_pat = re.compile(points_token)

#     def __call__(self, sentence: str, bboxes_seq: BoxesSeq) -> str:
#         all_box = self.bboxes_token_pat.findall(sentence)
#         assert len(all_box) == len(bboxes_seq), f"not match. sentence: {sentence}. boxes:{bboxes_seq}"
#         if len(all_box) == 0:
#             return sentence
#         bboxes_strs = [self.format_box(bboxes) for bboxes in bboxes_seq]
#         converted = sentence.replace(self.bboxes_token, '{}').format(*bboxes_strs)
#         return converted

#     def call_on_point(self, sentence: str, points_seq: BoxesSeq) -> str:
#         all_box = self.points_token_pat.findall(sentence)
#         assert len(all_box) == len(points_seq), f"not match. sentence: {sentence}. boxes:{points_seq}"
#         if len(all_box) == 0:
#             return sentence
#         bboxes_strs = [self.format_point(bboxes) for bboxes in points_seq]
#         converted = sentence.replace(self.points_token, '{}').format(*bboxes_strs)
#         return converted

#     def format_point(self, points) -> str:
#         raise NotImplementedError

#     def format_box(self, bboxes: Boxes) -> str:
#         raise NotImplementedError

#     def extract(self, string: str) -> List[Boxes]:
#         raise NotImplementedError

#     def extract_point(self, string: str) -> List[Boxes]:
#         raise NotImplementedError

# @METRICS.register_module()
# class RECComputeMetrics(BaseComputeMetrics):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.box_formatter: BoxFormatter = self.preprocessor['target']['boxes']

#     def __call__(self, eval_preds) -> Dict[str, Any]:
#         preds, targets = eval_preds
#         logger.warning(f"preds shape: {preds.shape}. targets shape: {targets.shape}")
#         assert len(preds) == len(targets)
#         return self.calculate_metric(preds, targets)
#     def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
#         failed = 0
#         target_failed = 0

#         # pred_boxes, target_boxes = [], []
#         # for pred, target in zip(preds, targets):
#         #     extract_pred = self.extract_ans(pred)
#         #     extract_target = self.extract_ans(target)
#         #     if extract_target is None:
#         #         target_failed += 1
#         #         logger.warning(f"failed to extract ans for target: {target}")
#         #         continue
#         #     if extract_pred is None:
#         #         failed += 1
#         #         logger.warning(f"failed to extract ans for pred: {pred}")
#         #         extract_pred = [0, 0, 0, 0]
#         #     target_boxes.append(extract_target)
#         #     pred_boxes.append(extract_pred)
#         pred_boxes = preds
#         target_boxes = targets
#         with torch.no_grad():
#             target_boxes = torch.tensor(target_boxes)
#             pred_boxes = torch.tensor(pred_boxes)
#             # normalized box value is too small, so that the area is 0.
#             ious = box_iou(pred_boxes * 1000, target_boxes * 1000)
#             ious = torch.einsum('i i -> i', ious)  # take diag elem
#             # NOTE: please note iou only calculate for success target
#             iou = ious.mean().item()
#             correct = (ious > 0.5).sum().item()

#         # HACK: currently we expand image to square. so this iou is the real iou.
#         warn_message = "this iou is calculate on normalized box. just for non-rigorous training progress checking." \
#                        "the value is consistent with real iou only if image.width == image.height."
#         warnings.warn(warn_message)

#         return {
#             'accuracy': 1.0 * correct / len(targets),
#             'target_failed': target_failed,
#             'failed': failed,
#             'iou': iou,
#             'warning': warn_message,
#         }

#     def extract_ans(self, string: str):
#         try:
#             list_of_boxes = self.box_formatter.extract(string)
#             if len(list_of_boxes) != 1 or len(list_of_boxes[0]) != 1:
#                 return None
#             box = list_of_boxes[0][0]
#             if len(box) != 4:
#                 return None
#             return box
#         except Exception as e:
#             logger.warning(f"extract_ans for {string} but get exception: {e}")
#             return None
