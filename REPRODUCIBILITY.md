# Reproducibility Notes

## Hardware and software assumptions

- 1 NVIDIA GPU is the default assumption in the cleaned configs.
- Python 3.8 or 3.10
- PyTorch, torchvision, timm, albumentations, OpenCV, numpy, matplotlib, piqa

## Dataset organization

The retained configs assume the following directory layout:

`datasets/Crack_train_x4/{GT,LR_bicubic}`
`datasets/Crack_val_x4/{GT,LR_bicubic}`
`datasets/Crack_test_x4/{GT,LR_bicubic}`

Images must be paired by filename between `GT` and `LR_bicubic`.

## Configs retained for the paper flow

- `options/train_seg.json`
- `options/test_seg_paper.json`
- `options/train_original.json`
- `options/test_original.json`

## Training procedure

Main method: `python main_train_psnr.py --opt options/train_seg.json`

Baseline: `python main_train_psnr.py --opt options/train_original.json`

## Testing procedure

Main method: `python test.py --opt options/test_seg_paper.json --model_path weights/your_trained_model.pth --save_suffix seg_eval`

Baseline: `python test.py --opt options/test_original.json --model_path weights/original_x4.pth --save_suffix original_eval`

## Metrics

The retained evaluation path computes PSNR, SSIM, and Edge-PSNR. By default the script evaluates on the Y channel.

## Mapping between experiments and retained assets

- Main manuscript method:
  - Train: `options/train_seg.json`
  - Test: `options/test_seg_paper.json`
  - Code path: `main_train_psnr.py -> data/select_dataset.py -> data/dataset_sr.py -> models/select_model.py -> models/model_plain.py -> models/select_network.py -> models/network_segtransnet.py`
- baseline:
  - Train: `options/train_original.json`
  - Test: `options/test_original.json`

## Incomplete or manual parts

- No checkpoints are included.
- The original workspace did not contain a paper-ready default SEG test config; `options/test_seg_paper.json` was created for this release.
