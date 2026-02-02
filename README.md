# UAV-MM3D & LGFusionNet

This repository provides the data organization and baseline implementation for **UAV-MM3D**, a large-scale synthetic multimodal benchmark for low-altitude UAV 3D perception, together with the proposed **LGFusionNet** (LiDAR-Guided Fusion Network).

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

Each scene directory follows a unified hierarchical structure to ensure **scene–weather–platform–target-count** consistency.

---

## 2. Directory Structure

Taking one scene as an example, the directory structure is organized as follows:

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

* **`00001`**: scenario index (each scenario contains **1–7 UAV targets**)
* **Weather (`clear_dy`, etc.)**: 8 physically based conditions
  *(clear / rain / fog / snow × day / night)*
* **UAV platform**: e.g., `DJI-avata2`, `Matrice300`, etc.

---

## 3. Modalities and Annotations

Each sequence contains **synchronized multi-modal data**:

### Sensor Modalities

* **RGB images** (`images_rgb`)
* **Infrared images** (`images_ir`)
* **Event-based data (DVS)** (`images_dvs`)
* **LiDAR point clouds** (`lidar_1`)
* **Millimeter-wave radar** (`radar_1`)

### Annotations

* **2D bounding boxes** per modality (`boxes_rgb`, `boxes_ir`, `boxes_dvs`)
* **3D bounding boxes and 6-DoF poses**
* **Instance-level UAV identities**

### Sensor Metadata

* `im_info.pkl`: camera intrinsics & extrinsics
* `drone_info.pkl`: UAV physical parameters and pose annotations
* `lidar_radar_info.pkl`: LiDAR–Radar calibration and projection info
* `distance_info.txt`: UAV-to-sensor distance statistics

All modalities are **temporally synchronized** and **geometrically aligned** under a unified OpenCV camera coordinate system.

---

## 4. LGFusionNet Code Structure

The LGFusionNet implementation is provided under the `uavdet3d/` directory.

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

* Dual-branch **RGB / IR** feature encoding
* **LiDAR-guided cross-modal alignment**
* Feature-level multimodal fusion
* Support for **6-DoF pose estimation** and **3D detection**

---

## 5. Installation

Before training or evaluation, install the project in development mode:

```bash
python setup.py develop
```

Make sure all dependencies listed in the original `README.md` are installed.

---

## 6. Usage

* Dataset preparation follows the directory structure described above.
* Configuration files are provided in `configs/`.
* Training and testing scripts are located in `tools/`.

Please refer to the original `README.md` for detailed training and evaluation commands.

---

## 7. License

This project is released under the **Apache 2.0 License**.

---

## 8. Citation

If you use this dataset or code, please cite:

```bibtex
@article{Zou_UAVMM3D,
  title={UAV-MM3D: A Large-Scale Synthetic Benchmark for 3D Perception of Unmanned Aerial Vehicles with Multi-Modal Data},
  author={Zou, Longkun and Wang, Jiale and Liang, Rongqin and Wu, Hai and Chen, Ke and Wang, Yaowei},
  journal={arXiv preprint arXiv:2511.22404},
  year={2025}
}
```

