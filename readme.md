# Source-Free Domain Adaptation for LiDAR Point Cloud Semantic Segmentation via Spatially Aware Prototype Learning&nbsp;&nbsp;&nbsp;

The code of our work "Source-Free Domain Adaptation for LiDAR Point Cloud Semantic Segmentation via Spatially Aware Prototype Learning".


# Introduction 

Source-Free Domain Adaptation (SFDA) offers a practical solution for deploying LiDAR point cloud segmentation models in new environments where source data cannot be accessed due to privacy constraints and storage limitations. Existing SFDA methods, however, are primarily designed for images and fail to account for the spatial–statistical structure of LiDAR data. In particular, they overlook global spatial heterogeneity and local spatial auto-correlation, resulting in unreliable pseudo-labels and unstable adaptation. To overcome these limitations, we propose Spatially Aware Prototype Learning (SAPL), a framework that embeds global and local spatial priors into a teacher–student pipeline for robust LiDAR adaptation. SAPL transforms conventional static, globally defined prototype learning into a dynamic, spatially aware process. It first employs an Adaptive Expert Inference (AEI) module to construct a scene-specific mixture-of-experts system based on the Bayesian Information Criterion, effectively mitigating global heterogeneity. A Local Spatial Consensus (LSC) mechanism, equipped with a novel Voxel Gini Impurity (VGI) metric, then quantifies neighborhood consistency and protects prototype calculation from spatial noise. Finally, a Geometric Calibration Module (GCM) fuses predictions from the parametric teacher and non-parametric prototypes to produce reliable, geometry-aligned pseudo-labels resistant to source-domain bias. Extensive experiments on multiple large-scale benchmarks demonstrate that SAPL establishes new state-of-the-art results, underscoring the importance of spatial–statistical modeling for SFDA in LiDAR point cloud segmentation.

---

##  Dependencies

This code was implemented and tested with python 3.10, PyTorch 1.13.1 and CUDA 11.7.
The MinkUnet backbone is implemented with version 1.4.0 of [Torchsparse](https://github.com/mit-han-lab/torchsparse.)([Exact commit](https://github.com/mit-han-lab/torchsparse/commit/69c1034ddb285798619380537802ea0ff03aeba6)).
The complete environment and dependencies can be found in the [environment.yml](https://github.com/feng946gg/SAPL/edit/master/environment.yml).

##  Datasets 
The datasets should be placed in data/ 
### SemanticPOSS
Download SemanticPOSS dataset from [here](http://poss.pku.edu.cn/semanticposs.html), then prepare data folders as follows:
```
./
├── 
├── ...
└── data/SemanticPOSS
    └──sequences/
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        └── 01/
```

### SemanticKITTI
To download SemanticKITTI follow the instructions [here](http://www.semantic-kitti.org). Then, prepare the paths as follows:
```
./
├── 
├── ...
└── data/SemanticKITTI/
      └──dataset
          ├── sequences
                ├── 00/           
                │   ├── velodyne/	
                |   |	   ├── 000000.bin
                |   |	   ├── 000001.bin
                |   |	   └── ...
                │   ├── labels/ 
                |   |      ├── 000000.label
                |   |      ├── 000001.label
                |   |      └── ...
                |   ├── calib.txt
                |   ├── poses.txt
                |   └── times.txt
                └── 08/
```
### PandaSet
To download PandaSet follow the instructions [here](https://www.kaggle.com/datasets/usharengaraju/pandaset-dataset/data). Then, prepare the paths as follows:
```
./
├── 
├── ...
└── data/pandaset/
      ├──001/
          ├── lidar/
          │    ├── 00.pkl
          │    ├── 01.pkl
          │    ├── ...
          ├── annotations/
          │    ├── semseg
          │    │    ├── 00.pkl
          │    │    ├── 01.pkl
          │    │    ├── ...
      ├──002/
      ├── ...
                
```

### Waymo Open
Follow the instructions [here](https://waymo.com/open/) to download the data and paths will be already like that:
```
./
├── 
├── ...
└── data/waymo_open_v_2_0_0/
      └──training/
          └── lidar/
          └── lidar_segmentation/
          └── lidar_calibration/
          └── lidar_pose/
      └──validation/
          └── lidar/
          └── lidar_segmentation/
          └── lidar_calibration/
          └── lidar_pose/
```
Note: If you want to use the Waymo Open dataset, you need to run the [preprocess_waymo.py](https://github.com/feng946gg/SAPL/edit/master/preprocess_waymo.py)
## Source-models
We use the same source model as [TTYD](https://github.com/valeoai/TTYD), which can be downloaded from the following link. Thank them for their excellent work!

| Source datatset | Link                                                         |
| --------------- | ------------------------------------------------------------ |
| NuScenes        | [Link](https://drive.google.com/drive/folders/1NpjvWzo7agNtLFu6HODRhIElTP3a04n7?usp=drive_link) |
| SyntheticLiDAR  | [Link](https://drive.google.com/drive/folders/1NrFpTUYmlmBBHqjyAolvp9FAoBSdvdoa?usp=drive_link) |

## Trainging

``python SAPL.py --name="sapl_synth_poss"  --bn_layer="scaling_per_channel" --resume_path=source_models/synth_semantic_TorchSparseMinkUNet --setting='Synth2POSS' --learning_rate=0.0025  --tensorboard_folder='SAPL'``

We explain it for SynLiDAR to SemanticPOSS. For other combinations, please change the --setting command (NS2SK, Synth2SK, Synth2POSS, NS2POSS, NS2PD, NS2WY). 

---

## SAPL models 

| Setting | Link |
| ------- | ---- |
| NS2SK   | TBD  |
| SL2SK   | TBD  |
| SL2SP   | TBD  |
| NS2SP   | TBD  |
| NS2PD   | TBD  |
| NS2WO   | TBD  |


# Acknowledgments

We thanks the open source projects, [TTYD](https://github.com/valeoai/TTYD) and [Torchsparse](https://github.com/mit-han-lab/torchsparse.). 

---
