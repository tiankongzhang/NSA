# Unsupervised Domain Adaptive Detection with Network Stability Analysis

[[`Paper`](https://arxiv.org/abs/2308.08182)] [[`Project`](https://github.com/tiankongzhang/NSA)] 

This project hosts the code for the implementation of Unsupervised Domain Adaptive Detection with Network Stability Analysis (ICCV 2023).

## Introduction

Domain adaptive detection aims to improve the generality of a detector, learned from the labeled source domain, on
the unlabeled target domain. In this work, drawing inspiration from the concept of stability from the control theory
that a robust system requires to remain consistent both externally and internally regardless of disturbances, we propose a novel framework that achieves unsupervised domain adaptive detection through stability analysis. In specific, we treat discrepancies between images and regions from different domains as disturbances, and introduce a novel simple but effective Network Stability Analysis (NSA) framework that considers various disturbances for domain adaptation. Particularly, we explore three types of perturbations including heavy and light image-level disturbances and instancelevel disturbance. For each type, NSA performs external consistency analysis on the outputs from raw and perturbed images and/or internal consistency analysis on their features, using teacher-student models. By integrating NSA into Faster R-CNN, we immediately achieve state-of-the-art results. In particular, we set a new record of 52.7% mAP on Cityscapes-to-FoggyCityscapes, showing the potential of NSA for domain adaptive detection. It is worth noticing, our NSA is designed for general purpose, and thus applicable to one-stage detection model (e.g., FCOS) besides the adopted one, as shown by experiments.

![NSA design](assets/framework_fig.png?raw=true)

## Installation

The implementation of our anchor-based detector is heavily based on Faster-RCNN ([\#f0a9731](https://isrc.iscas.ac.cn/gitlab/research/domain-adaption)).

Install NSA:

```
pip install imgaug==0.4.0
```

## Model Checkpoints

Three model versions of the model are available with different backbone sizes. These models can be instantiated by running

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adaptation Type</th>
      <th>Detector</th>
      <th>mAP</th>
      <th>model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cityscapes->FoggyCityscapes</th>
      <td>Faster-RCNN</td>
      <td>0.527</td>
      <td>model(https://drive.google.com/drive/folders/1TuZMUqbA3Or-BtJPo29lDJ5cs_4bkkG6)</td>
    </tr>
    <tr>
      <th>Cityscapes->RainCityscapes</th>
      <td>Faster-RCNN</td>
      <td>0.587</td>
      <td>model(https://drive.google.com/drive/folders/1TuZMUqbA3Or-BtJPo29lDJ5cs_4bkkG6)</td>
    </tr>
    <tr>
      <th>Sim10k->Cityscapes</th>
      <td>Faster-RCNN</td>
      <td>0.563</td>
      <td>model(https://drive.google.com/drive/folders/1TuZMUqbA3Or-BtJPo29lDJ5cs_4bkkG6)</td>
    </tr>
    <tr>
      <th>KITTI->Cityscapes</th>
      <td>Faster-RCNN</td>
      <td>0.556</td>
      <td>model(https://drive.google.com/drive/folders/1TuZMUqbA3Or-BtJPo29lDJ5cs_4bkkG6)</td>
    </tr>
    <tr>
      <th>Cityscapes->BDD100k</th>
      <td>Faster-RCNN</td>
      <td>0.355</td>
      <td>model(https://drive.google.com/drive/folders/1TuZMUqbA3Or-BtJPo29lDJ5cs_4bkkG6)</td>
    </tr>
  </tbody>
</table>


## Evaluation

The trained model can be evaluated by the following command.

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2800 dis_test-nsa.py --config-file configs/NSA/city/adv_vgg16_cityscapes_2_foggy_nsa_s1.yaml --resume /your_path/ --test-only
```



## Citations

Please consider citing our paper in your publications if the project helps your research.

```
@InProceedings{Zhou_2023_ICCV_NSA,
    author    = {Zhou, Wenzhang and Heng, Fan and Luo, Tiejian and Zhang, Libo},
    title     = {Unsupervised Domain Adaptive Detection with Network Stability Analysis},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2023}
}
```
