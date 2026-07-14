# Fixed Public Split Manifests

This directory contains the filename-level split used for the public benchmark release.

- `train.txt`: 800 GT filenames from `datasets/Crack_train_x4/GT`
- `val.txt`: 100 GT filenames from `datasets/Crack_val_x4/GT`
- `test.txt`: 100 GT filenames from `datasets/Crack_test_x4/GT`

Each line contains one image filename. The corresponding LR images are expected to have the same base name in the matching `LR_bicubic` directory.

The training code still reads the directory structure in `datasets/`, but these manifests document the fixed split used for the public benchmark.
