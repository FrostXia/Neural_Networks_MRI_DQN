# AI_DL: OASIS1 CNN + DQN Active Learning

This project is designed for CNN classification, DQN-based active learning, and optional Differential Evolution (DE) hyperparameter search on OASIS1 MRI data. It supports 2.5D multi-slice input and configurable CNN backbones.

## Jupyter Notebooks

| Notebook | Purpose |
| --- | --- |
| `run_al_dqn.ipynb` | Builds the OASIS1 manifest, runs CNN + DQN active learning training and evaluation, and can generate heatmap results. |
| `run_de.ipynb` | Runs grouped Differential Evolution hyperparameter search for OASIS1 experiments, supporting `kfold`, `seeds`, or `split` evaluation modes, and uses the best checkpoint for subsequent training and evaluation. |

## Main Scripts

- `build_manifest.py`: Builds training, validation, test, and active learning manifests from OASIS1 image paths and the clinical table.
- `oasis_common.py`: Provides data processing, dataset splitting, and evaluation utilities.
- `train_cnn_dqn_de.py`: Main training entry point for CNN, DQN, and DE experiments.

## Required Libraries

Only the necessary top-level runtime dependencies identified from the project source and the provided Conda environment are listed below:

```text
torch==2.11.0
torchvision==0.26.0
monai==1.5.2
nibabel==5.4.2
numpy==2.2.6
pandas==2.3.3
scikit-learn==1.7.2
matplotlib==3.10.9
Pillow==12.2.0
tqdm==4.67.3
openpyxl==3.1.5
ipykernel==7.2.0
```

Install the environment with:

```bash
conda create -n oasis_sfcn python=3.10 -y
conda activate oasis_sfcn
pip install -r requirements.txt
python -m ipykernel install --user --name oasis_sfcn --display-name "Python (oasis_sfcn)"
```

## Data and Usage

1. Keep the raw OASIS1 dataset locally and set `OASIS1_ROOT` in `run_al_dqn.ipynb` to the correct path.
2. Place the clinical table at `data_tables/oasis_cross-sectional.xlsx`, or update the `OASIS1_CLINICAL` path in the notebook.
3. Start Jupyter from the project root directory and run the required notebook.

Datasets, caches, and training outputs are not included in this lightweight repository package. Before publishing any data files, confirm the applicable licence and privacy requirements.

## Tested Environment

Tested on the following system:

- OS: Ubuntu 22.04 LTS
- CPU: Intel Ultra 9
- GPU: NVIDIA RTX 5080
- Python: 3.10.20

## Repository Notes

Generated caches and training artifacts, such as `cache*/`, `outputs*/`, `.npz` files, and checkpoints, are excluded by `.gitignore` to keep the repository lightweight and reproducible.
