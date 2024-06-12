# ELK: Enhanced Learning through cross-modal Knowledge transfer for lesion detection in limited-sample contrast-enhanced mammography datasets

We propose a cross-modal knowledge transfer pipeline to adapt a model pretrained on digital breast tomosynthesis (DBT) and digital mammography data into a target CEM population with limited data-volume. Our approach leverages diffusion models to synthesize high-resolution, realistic mass-like lesions, enriching underrepresented datasets and enhancing model performance.


## Repository structure
```
.
├── README.md
├── data
│   ├── CDD-CESM
│   │   ├── images
│   │   ├── masks
│   │   ├── masks_closeup
│   │   └── metadata
│   ├── models
│   │   ├── config_trained_R_101_30k.yaml
│   │   └── model_final_R_101_omidb_30k_dbt9k_f12_gray.pth
│   └── SET-Mex
│       ├── binary_masks
│       ├── images
│       └── metadata
├── data_analysis
├── detection
├── envs
├── generation
├── utils.py
