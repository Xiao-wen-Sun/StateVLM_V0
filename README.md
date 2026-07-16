# StateVLM
## StateVLM: A State-Aware Vision-Language Model for Robotic Affordance Reasoning

Vision-language models (VLMs) have demonstrated strong performance across robotic perception and instruction-following tasks. However, they still struggle with precise spatial reasoning, particularly in predicting object locations and fine-grained object states.
We propose StateVLM, a vision-language model designed to learn fine-grained object representations, including object localization and grasp-relevant region prediction. 
We introduce a joint training objective that integrates an auxiliary regression loss (ARL) with the standard causal language modeling (CLM) objective to improve numerical reasoning and spatial understanding. 
To evaluate whether models can move beyond category-level grounding toward state-aware spatial understanding, we introduce an open-source benchmark, Object State Affordance Reasoning (OSAR), comprising 1,172 scenes with 7,746 individual objects and their corresponding bounding boxes.
Empirical experiments on RefCOCO, RefCOCO+, and RefCOCOg demonstrate that integrating ARL improves model performance compared with CLM only. Experiments on the OSAR benchmark further demonstrate that StateVLM with ARL achieves an average improvement of 5.2\% over models trained with CLM only. These results show that ARL is particularly beneficial for the complex affordance reasoning tasks that require object state understanding.
The joint training objective in StateVLM demonstrates that integrating an auxiliary regression objective into VLM training improves numerical reasoning, and the OSAR benchmark provides a new testbed for understanding object-state affordances in robotics.



## Contents <!-- omit in toc -->

- [1. Installation](#Installation)
- [2. Dataset Preparation](#Dataset-Preparation)
- [3. Comparative Training Methods](#Comparative-Training-Methods)
- [4. Model Zoo](#Model-Zoo)
- [5. Evaluation](#Evaluation)


## 1. Installation

1. Clone this repository and navigate to the source folder

```bash
git clone https://github.com/Xiao-wen-Sun/StateVLM.git
cd StateVLM
```

2. Create conda environment

```Shell
conda create -n StateVLM python=3.10 -y
conda activate StateVLM
```

3. Install dependencies

```Shell
pip install -r requirements.txt
```

4. Configure PYTHONPATH

```Shell
export PYTHONPATH=/PATH/StateVLM_V0
```

## Dataset Preparation
### Experiment 1: Referring Expression Comprehension (RefCOCO, RefCOCO+, and RefCOCOg)
#### 1.1 Image from COCO
https://cocodataset.org/#download

In this work, we only need train2014

Download [2014 train images](http://images.cocodataset.org/zips/train2014.zip)      

http://images.cocodataset.org/zips/train2014.zip


#### 1.2 Annotations
    
##### 1.2.1 Download adapted refcoco3_dialog annotations directly (recommended)


Download [Adapted annotations](https://drive.google.com/file/d/10DBwdFa8cuM1qn1FlwxvpT8ckIU3wJrj/view?usp=sharing)

https://drive.google.com/file/d/10DBwdFa8cuM1qn1FlwxvpT8ckIU3wJrj/view?usp=sharing

##### 1.2.2 Convert refcoco3 to refcoco3_dialog for training

Download [Original annotations](
https://drive.google.com/file/d/1ZiNQ1YZlmVcvRGQqIjEhzfR3d8zDz88M/view?usp=sharing)

https://drive.google.com/file/d/1ZiNQ1YZlmVcvRGQqIjEhzfR3d8zDz88M/view?usp=sharing

###### Convert refcoco3 to refcoco3_dialog for training

```Shell
python statevlm/utils/prepare_refcoco_for_training.py
```
###### Convert refcoco3 test to refcoco3_dialog test for evaluation

```Shell
python statevlm/utils/prepare_refcoco_for_evaluation.py
```

### Experiment 2: Object State Affordance Reasoning (OSAR)

Download [OSAR Benchmark](
https://www2.informatik.uni-hamburg.de/wtm/OSAR/OSAR_Benchmark.zip)

https://www2.informatik.uni-hamburg.de/wtm/OSAR/OSAR_Benchmark.zip


```text
StateVLM/
└── datasets/
    ├── REC/
    │   ├── train2014/
    │   │   ├── COCO_train2014_000000000009.jpg
    │   │   └── ...
    │   ├── refcoco3_annotations/ (Optional)
    │   │   ├── REC_ref3_train.jsonl
    │   │   └── ...
    │   ├── refcoco3_dia_annotations/
    │       ├── REC_ref3_train_dia.json
    │       │   ...
    │       ├── REC_refcoco_unc_testA_dia_eva.json
    │       └── ...
    └── OSAR_Benchmark/
        ├── images/
        │   ├── train_images/
        │   └── test_images/
        └── annotations/
            ├── state_train_dia.json
            └── ...
```

## Comparative Training Methods

https://huggingface.co/openbmb/MiniCPM-V-2_6/tree/main


## Train the model with joint training objective: Causal Language Modeling (CLM) + Auxiliary Regression Loss (ARL)
### Full fine-tune StateVLM on adapted REC benchmark
set "model_embedding = True" in the configuration_statevlm.py 

(
/path/StateVLM_V0/statevlm/model/configuration_statevlm.py)
```
cd script_REC
bash train_statevlm_emb.sh
```
### LoRA fine-tune StateVLM on OSAR
```
cd script_OSAR
bash train_statevlm_lora_emb.sh
```

## Train the model with Causal Language Modeling (CLM)

set "model_embedding = False" in the configuration_statevlm.py 

(
/path/StateVLM_V0/statevlm/model/configuration_statevlm.py)
```
cd script_REC
bash train_statevlm_seq.sh
```
### LoRA fine-tune StateVLM on OSAR
```
cd script_OSAR
bash train_statevlm_lora_seq.sh
```



## Model Zoo

Causal Language Modeling (CLM) is the default autoregressive objective used by MiniCPM-V.

Auxiliary Regression Loss (ARL) is our proposed regression objective.

Object State Affordance Reasoning (OSAR) is our proposed benchmark for referring expression comprehension related to object states.

| Model | Fine-Tuning Method | Training objective | Memory |  Datesets | Download |
|:-----------|:--:|:-----------:|:-----------:|:-------------------|:---------------:|
| MiniCPM-V 2.6|Baseline|--| 17 GB  | --   |  [<img src="./assets/modelscope_logo.png" width="20px"></img>]() |
| statevlm_seq_5000 | Full Fine-Tuning|$StateVLM _{CLM}$ | 17 GB  | RefCOCO3 | [<img src="./assets/modelscope_logo.png" width="20px"></img>]() |
| statevlm_emb_20000 |Full Fine-Tuning| $StateVLM _{CLM+ARL}$ | 17 GB  |  RefCOCO3| [<img src="./assets/modelscope_logo.png" width="20px"></img>]() |
| statevlm_seq_5000_lora_adapter | LoRA Fine-Tuning |$StateVLM _{CLM, lora}$ | 17 GB  | OSAR | [<img src="./assets/modelscope_logo.png" width="20px"></img>]() |
| statevlm_emb_20000_lora_adapter |LoRA Fine-Tuning | $StateVLM _{CLM+ARL, lora}$  | 17 GB  | OSAR | [<img src="./assets/modelscope_logo.png" width="20px"></img>]() |



## Evaluation

You can download pretrained models from the model zoo to evaluate them directly.
```text
StateVLM/
└── model_zoo/
    ├── MiniCPM-V-2_6
    ├── statevlm_seq_5000  
    ├── statevlm_seq_5000_lora_adapter 
    ├── statevlm_emb_20000
    ├── statevlm_emb_20000_lora_adapter
        
```



Performance Evaluation of the Baseline MiniCPM-V on Adapted RefCOCO3
(Folder script_REC/baseline)

Performance Evaluation of the $StateVLM _{CLM}$ on Adapted RefCOCO3
(Folder script_REC/emb)

Performance Evaluation of the $StateVLM _{CLM+ARL}$ on Adapted RefCOCO3
(Folder script_REC/seq)

Performance Evaluation of the $StateVLM _{CLM, lora}$ and $StateVLM _{CLM+ARL, lora}$ on OSAR
(Folder script_OSAR/baseline)

Performance Evaluation of the $StateVLM _{CLM+ARL, lora}$ on OSAR
(Folder script_OSAR/emb)

Performance Evaluation of the $StateVLM _{CLM, lora}$ on OSAR
(Folder script_OSAR/seq)



