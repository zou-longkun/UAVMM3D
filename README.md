# UAV-MM3D & LGFusionNet

This repository provides the dataset organization and baseline implementation for **UAV-MM3D**, a large-scale synthetic multimodal benchmark for low-altitude UAV 3D perception, together with the proposed **LGFusionNet** (LiDAR-Guided Fusion Network).

The dataset and code support multiple UAV perception tasks, including **3D detection**, **6-DoF pose estimation**, **target tracking**, and **trajectory prediction**.

---

## 1. Dataset Overview

The dataset root directory is **`data_collect/`**, which contains four CARLA simulated scenes with different urban layouts and structural complexity:

```
data_collect/
├── Town01_Opt
├── Town03HD_Opt
├── Town07HD_Opt
└── Town10HD_Opt
```

---

## 2. Dataset Download

The **UAV-MM3D** dataset is released via Baidu Netdisk.

* **Dataset Name**: UAV-MM3D
* **Download Link**: [https://pan.baidu.com/s/1TzAlk8yeTbmHqnpS5JGAKg](https://pan.baidu.com/s/1TzAlk8yeTbmHqnpS5JGAKg)
* **Extraction Code**: `uavm`

After downloading, please extract the dataset and organize it as follows:

```
data_collect/
├── Town01_Opt
├── Town03HD_Opt
├── Town07HD_Opt
└── Town10HD_Opt
```

---

## 3. Directory Structure

Each scene directory follows a unified hierarchical structure to ensure scene–weather–platform–target-count consistency.

```
Town01_Opt/
└── carla_data/
    └── 00001/                 # Scenario ID (up to 7 UAV targets)
        └── clear_dy/           # Weather condition (8 types)
            └── DJI-avata2/     # UAV platform
                ├── images_rgb/
                ├── images_ir/
                ├── images_dvs/
                ├── lidar_1/
                ├── radar_1/
                ├── boxes_rgb/
                ├── boxes_ir/
                ├── boxes_dvs/
                ├── distance_info.txt
                ├── im_info.pkl
                ├── drone_info.pkl
                └── lidar_radar_info.pkl
```

### Naming Conventions

* **Scenario ID (`00001`)**: each scenario contains **1–7 UAV targets**
* **Weather conditions**: 8 physically based settings
  *(clear / rain / fog / snow × day / night)*
* **UAV platforms**: multiple real-world UAV models (e.g., DJI Avata 2, Matrice series)

---

## 4. Modalities and Annotations

Each sequence contains synchronized multi-modal data.

### Sensor Modalities

* **RGB images** (`images_rgb`)
* **Infrared images** (`images_ir`)
* **Event-based data (DVS)** (`images_dvs`)
* **LiDAR point clouds** (`lidar_1`)
* **Millimeter-wave radar** (`radar_1`)

### Annotations

* 2D bounding boxes for each visual modality
* 3D bounding boxes and full **6-DoF poses**
* Instance-level UAV identities

### Sensor Metadata

* `im_info.pkl`: camera intrinsics and extrinsics
* `drone_info.pkl`: UAV physical parameters and pose annotations
* `lidar_radar_info.pkl`: LiDAR–Radar calibration and projection information
* `distance_info.txt`: UAV-to-sensor distance statistics

All modalities are temporally synchronized and geometrically aligned under a unified OpenCV camera coordinate system.

---

## 5. LGFusionNet Code Structure

The implementation of **LGFusionNet** is provided under the `uavdet3d/` directory.

```
uavdet3d/
├── datasets/          # Dataset loaders and preprocessing
├── model/             # LGFusionNet and related modules
├── utils/             # Geometry, projection, and evaluation utilities
├── config.py
├── version.py
└── __init__.py
```

Additional directories:

```
configs/
├── laam6d_det_lidar_fusion_config.py

tools/
output_models/
```

### Key Features

* Dual-branch RGB / IR feature encoding
* LiDAR-guided cross-modal spatial alignment
* Feature-level multimodal fusion
* Support for 3D detection and 6-DoF pose estimation

---

## 6. Installation

Install the project in development mode before training or evaluation:

```
python setup.py develop
```

---

## 7. Usage

* Dataset preparation should strictly follow the directory structure described above.
* Configuration files are provided in `configs/`.
* Training and evaluation scripts are located in `tools/`.

---

## 8. License

This project is released under the **Apache 2.0 License**.

---

## 9. Citation

If you use this dataset or code, please cite:
```bibtex
@article{zou2025uav,
  title={UAV-MM3D: A Large-Scale Synthetic Benchmark for 3D Perception of Unmanned Aerial Vehicles with Multi-Modal Data},
  author={Zou, Longkun and Wang, Jiale and Liang, Rongqin and Wu, Hai and Chen, Ke and Wang, Yaowei},
  journal={arXiv preprint arXiv:2511.22404},
  year={2025}
}
```
