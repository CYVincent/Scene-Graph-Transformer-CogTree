# Scene Graph Transformer with CogTree Loss in Pytorch
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.4.0-%237732a8)

## Contents

1. [Overview](#Overview)
2. [Install the Requirements](INSTALL.md)
3. [Prepare the Dataset](DATASET.md)
4. [Training on Scene Graph Generation](#Examples of the Training SG-Transformer Command)
5. [Evaluation on Scene Graph Generation](#Evaluation)

## Overview

This code is based on [Scene Graph Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and [Neural-Backed Decision Trees](https://github.com/alvinwan/neural-backed-decision-trees). We propose a new SG-Transformer model and a novel CogTree loss in SGG task. We merge the Pytorch implementation into Scene Graph Benchmark. Please see more details in our paper [CogTree: Cognition Tree Loss for Unbiased Scene Graph Generation](https://arxiv.org/abs/2009.07526).

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Pretrained Models

You can download the [pretrained Faster R-CNN](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779870&authkey=AH5CPVb9g5E67iQ) we used in the paper, which is the same as [Scene Graph Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). Please check [Scene Graph Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) for more details.

After you download the [Faster R-CNN model](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779870&authkey=AH5CPVb9g5E67iQ), please extract all the files to a directory and set the same path in your training command.

## Perform training on Scene Graph Generation

There are **three standard protocols**: (1) Predicate Classification (PredCls): taking ground truth bounding boxes and labels as inputs, (2) Scene Graph Classification (SGCls) : using ground truth bounding boxes without labels, (3) Scene Graph Detection (SGDet): detecting SGs from scratch. We use two switches ```MODEL.ROI_RELATION_HEAD.USE_GT_BOX``` and ```MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL``` to select the protocols. 

For **Predicate Classification (PredCls)**, we need to set:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
```
For **Scene Graph Classification (SGCls)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```
For **Scene Graph Detection (SGDet)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```

### Predefined Models
We abstract various SGG models to be different ```relation-head predictors``` in the file ```roi_heads/relation_head/roi_relation_predictors.py```, which are independent of the Faster R-CNN backbone and relation-head feature extractor. To select our predefined models, you can use ```MODEL.ROI_RELATION_HEAD.PREDICTOR```.

For [Neural-MOTIFS](https://arxiv.org/abs/1711.06640) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor
```
For [Iterative-Message-Passing(IMP)](https://arxiv.org/abs/1701.02426) Model (Note that SOLVER.BASE_LR should be changed to 0.001 in SGCls, or the model won't converge):
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR IMPPredictor
```
For [VCTree](https://arxiv.org/abs/1812.01880) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor
```
For [Unbiased-Causal-TDE](https://arxiv.org/abs/2002.11949) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor
```
For [SG-Transformer](https://arxiv.org/abs/2009.07526) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor
```

The default settings are under ```configs/e2e_relation_X_101_32_8_FPN_1x.yaml``` and ```maskrcnn_benchmark/config/defaults.py```. The priority is ```command > yaml > defaults.py```.

### CogTree loss
The CogTree loss consist of two parts: the class-balanced cross-entropy loss (CB) and the tree-based class-balanced hierarchical classification loss (TCB).

For CB, we need to set:
```bash
MODEL.ROI_RELATION_HEAD.LOSS.USE_CLASS_BALANCED_LOSS True
```
For TCB, we need to set:
```bash
MODEL.ROI_RELATION_HEAD.LOSS.USE_NBDT_LOSS True
```

### Examples of the Training SG-Transformer Command
Training Example 1 : (SGCls)
```bash
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 25000 SOLVER.VAL_PERIOD 5000 SOLVER.CHECKPOINT_PERIOD 5000 SOLVER.STEPS 25000, GLOVE_DIR home/yuanchai/glove_dir MODEL.PRETRAINED_DETECTOR_CKPT home/yuanchai/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR home/yuanchai/checkpoints/SG-Transformer-SGCls
```
where ```GLOVE_DIR``` is the directory used to save glove initializations, ```MODEL.PRETRAINED_DETECTOR_CKPT``` is the pretrained Faster R-CNN model you want to load, ```OUTPUT_DIR``` is the output directory used to save checkpoints and the log.

Training Example 2 : (SGCls, CogTree)
```bash
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 25000 SOLVER.VAL_PERIOD 5000 SOLVER.CHECKPOINT_PERIOD 5000 SOLVER.STEPS 25000, GLOVE_DIR home/yuanchai/glove_dir MODEL.PRETRAINED_DETECTOR_CKPT home/yuanchai/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR home/yuanchai/checkpoints/SG-Transformer-SGCls-CogTree MODEL.ROI_RELATION_HEAD.LOSS.USE_CLASS_BALANCED_LOSS True MODEL.ROI_RELATION_HEAD.LOSS.USE_NBDT_LOSS True
```


## Evaluation

### Examples of the Test SG-Transformer Command
Test Example 1 : (SGCls)
```bash
CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/yuanchai/glove_dir MODEL.PRETRAINED_DETECTOR_CKPT home/yuanchai/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR home/yuanchai/checkpoints/SG-Transformer-SGCls-CogTree
```
