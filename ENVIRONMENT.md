# Environment

The release is intended to run with Python 3.10 and CUDA-enabled PyTorch when a GPU is available.

## Recommended setup

1. Create the conda environment:

   `conda env create -f environment.yml`

2. Or install the Python packages directly:

   `pip install -r requirements.txt`

3. Use a single GPU by default unless you intentionally change the distributed settings.

## Included dependency files

- `environment.yml`
- `requirements.txt`

## Core packages

The release relies on PyTorch, torchvision, timm, albumentations, OpenCV, numpy, matplotlib, piqa, and requests.
