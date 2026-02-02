# Open 3D UAV Detection Codebase (2D UAV, 3D UAV detection, 6D UAV pose estimation)

6D pose estimation require object shape and class prior, 3D UAV detection does not require such prior.

## Detection Framework




## Datasets

UG2 challenge 

https://drive.google.com/drive/folders/1wk-c5xVX6701WNI_In1ba3_D4LSjRYv5

MAV6D

## Model Zoo and Benchmark

## 2D

## Getting Started
### Dependency
Our released implementation is tested on.


### Prepare dataset

MAV6D dataset
```
data
├── MAV6D
│   ├── phantom4
│   │   │── JPEGImages
│   │   │   │──01(scene)
│   │   │   │   │──0101 (seq)
│   │   │   │   │   │──1.jpg
│   │   │   │   │   │──...
│   │   │   │──02
│   │   │   │──...
│   │   │── labels
│   │   │── split
│   │   │   │── train.txt
│   │   │   │── val.txt
│   │   │   │── test.txt
├── uavdet3d
├── tools
```



### Setup

```
cd Open3DUAVDet
python setup.py develop
```

### Training

### Testing

## License

This code is released under the [Apache 2.0 license](LICENSE).

## High-level API



## Citation

```
@inproceedings{VirConv,
    title={Virtual Sparse Convolution for Multimodal 3D Object Detection},
    author={Wu, Hai and Wen,Chenglu and Shi, Shaoshuai and Wang, Cheng},
    booktitle={CVPR},
    year={2023}
}
```




