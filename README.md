# EquiSym

# Data Preparation
1.Download[ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip), and place it in the following directory:  `datasets/modelnet40_normal_resampled/`.
2.Download[ShapeNet](https://condor-datasets.s3.us-east-2.amazonaws.com/dataset/ShapeNetAtlasNetH5_1024.zip), and place it in the following directory: `datasets/ShapeNetAtlasNetH5_1024/`.
3.Download[ScanObjectNN](ScanObjectNN), and place it in the following directory: `datasets/ScanObjectNN`.

# Usage
## Single-Category Training/Testing
```bash
python train_pose.py --model $optional$ --data_choice shapenet
python test_pose.py --model $optional$ --data_choice shapenet
```
## Multi-Category Training/Testing
```bash
python train_pose.py --model $optional$ --data_choice modelnet
python test_pose.py --model $optional$ --data_choice modelnet
```
## Real-World Data Training/Testing
```bash
python train_pose.py --model $optional$ --data_choice scanobject
python test_pose.py --model $optional$ --data_choice modelnet
```
Here, `$optional$` can be one of the following: `vn_pointnet`, `vn_dgcnn`, `vn_transformer`, `vn_pointnet_am`, `vn_dgcnn_am`, `vn_transformer_am`. The suffix `_am` indicates that our method was used.


# Acknowledgement
This repository is built upon the following works:  [PointNet/PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), [DGCNN](https://github.com/WangYueFt/dgcnn), [VNN](https://github.com/FlyingGiraffe/vnn), [OAVNN](https://github.com/sidhikabalachandar/oavnn) and [VN-Transformer](https://github.com/lucidrains/VN-transformer). We gratefully acknowledge the authors for their foundational contributions.

