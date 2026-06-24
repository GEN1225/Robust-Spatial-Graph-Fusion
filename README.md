# EAAI: Segmented Architecture for DEM Spatial Generalization
# 分段架构DEM空间泛化研究

基于分段架构的DEM（数字高程模型）空间泛化研究项目。
Research on segmented architecture for DEM (Digital Elevation Model) spatial generalization.

> **Note / 注意**: This repository contains datasets, preprocessing scripts, and experiment frameworks.
> Core model implementations (FusionModel, SoftGatedSegmentedModel) are proprietary and available
> upon reasonable request for academic collaboration.
>
> 本仓库包含数据集、预处理脚本和实验框架。核心模型实现为专有代码，可通过学术合作获取。

## Project Structure / 项目结构

```
eaai/
├── models/                    # Model interfaces / 模型接口
│   ├── data_utils.py          # Data loading & graph construction / 数据加载与图构建
│   ├── gnn_models.py          # GNN model interfaces / GNN模型接口
│   └── trainer.py             # Training interfaces / 训练接口
│
├── experiments/               # Experiment scripts / 实验脚本
│   ├── exp1_segmentation_ablation.py    # Segmentation ablation / 分段消融
│   ├── exp2_tkg_ekg_ablation.py        # TKG/EKG ablation / 图源消融
│   ├── exp3_boundary_robustness.py     # Boundary robustness / 边界鲁棒性
│   ├── exp4_spatial_generalization.py  # Spatial generalization / 空间泛化
│   ├── exp5_graph_structure_analysis.py # Graph structure analysis / 图结构分析
│   └── strict_statistical_analysis.py  # Statistical analysis / 统计分析
│
├── fastgtn_repo/              # FastGTN implementation / FastGTN实现
│
├── data/                      # Dataset files / 数据文件
│   ├── caseA_forest_biomass/  # Forest biomass (4000 samples, 12 dims)
│   ├── caseB_california_housing/  # California housing (20640 samples, 8 dims)
│   ├── caseC_county_poverty/  # County poverty (1749 samples, 9 dims)
│   └── caseD_eurosat/         # EuroSAT remote sensing (27000 samples, 512 dims)
│
└── data_preprocessing/        # Dataset construction scripts / 数据构建脚本
```

## Datasets / 数据集

| Dataset | Samples | Features | Task | Source |
|---------|---------|----------|------|--------|
| Case A: Forest Biomass | 4,000 | 12 | Regression | Remote sensing (NDVI + AGB + DEM) |
| Case B: California Housing | 20,640 | 8 | Regression | 1990 California Census |
| Case C: County Poverty | 1,749 | 9 | Regression | ACS American Community Survey |
| Case D: EuroSAT | 27,000 | 512 | Classification | Sentinel-2 satellite imagery |

### Data References / 数据引用

**Case A**: Zhu, X., et al. (2024). "Forest biomass estimation using multi-source remote sensing data and graph neural networks." *Remote Sensing of Environment*, 305.

**Case B**: Pace, R. K., & Barry, R. (1997). "Sparse spatial autoregressions." *Statistics & Probability Letters*, 33(3), 291-297.

**Case C**: U.S. Census Bureau. (2022). *American Community Survey 5-Year Estimates*. Table S1701.

**Case D**: Helber, P., et al. (2019). "EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification." *IEEE JSTARS*, 12(7), 1993-2001.

## Usage / 使用方法

```bash
# Run all experiments / 运行所有实验
python run_all_experiments.py

# Run single experiment / 运行单个实验
python experiments/exp1_segmentation_ablation.py
```

```python
# Load Case A data / 加载Case A数据
import numpy as np
features = np.load('data/caseA_forest_biomass/aligned_features_multiyear.npy')
labels = np.load('data/caseA_forest_biomass/aligned_labels_multiyear.npy')

# Load Case B data / 加载Case B数据
from data.caseB_california_housing.load_california_housing import load_california_housing
features, labels, names = load_california_housing()
```

## Dependencies / 依赖

- PyTorch >= 1.8.0
- PyTorch Geometric
- NumPy, SciPy, scikit-learn, pandas

## License / 许可

Academic use only. / 仅用于学术研究。
