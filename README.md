# SEG-TransNet Minimal Reproducible Release

This repository contains the cleaned, minimal reproducible code package prepared for the manuscript currently under submission.

This repository is directly related to the manuscript currently submitted to The Visual Computer.

If you use this code, please cite the related manuscript together with any upstream work it builds on.

## Summary

This release keeps only the code path required to reproduce the paper's core crack-image super-resolution workflow built on the SEG-TransNet overall SR network. In the retained implementation, a SEG (Sequential Edge Gating) module is inserted into the backbone trunk. 

## Environment

- Python 3.8 or 3.10
- PyTorch with CUDA support recommended
- Single-GPU execution is the default assumption in the cleaned configs

Install dependencies with `pip install -r requirements.txt` or `conda env create -f environment.yml`.

## Repository structure

`main_train_psnr.py`, `test.py`, `options/`, `data/`, `models/`, `utils/`, `datasets/`, `weights/`, `experiments/`, plus the release documentation files.

## Data preparation

Expected dataset layout:

`datasets/Crack_train_x4/{GT,LR_bicubic}`
`datasets/Crack_val_x4/{GT,LR_bicubic}`
`datasets/Crack_test_x4/{GT,LR_bicubic}`

The retained code assumes paired `GT` and `LR_bicubic` images with matching filenames.

## Training

`python main_train_psnr.py --opt options/train_seg.json`

Baseline:

`python main_train_psnr.py --opt options/train_original.json`

## Testing and evaluation

`python test.py --opt options/test_seg_paper.json --model_path weights/your_trained_model.pth --save_suffix seg_eval`

Baseline:

`python test.py --opt options/test_original.json --model_path weights/original_x4.pth --save_suffix original_eval`

## Reproducing the main results

1. Prepare the dataset folders under `datasets/`.
2. Place the official ×4 initialization checkpoint at `weights/x4.pth`, or train from scratch by editing the config.
3. Run SEG-TransNet training with `options/train_seg.json`.
4. Run evaluation with `options/test_seg_paper.json`.
5. Compare against the retained baseline config `options/train_original.json` and `options/test_original.json`.

