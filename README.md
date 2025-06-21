# PixieDet

PixieDet is a lightweight object detection project implemented in PyTorch. It provides scripts to train a YuNet based model on face detection datasets such as WIDERFace and COCO keypoints.

## Features

- Anchor‑free detector using SimOTA assignment
- Training scripts for WIDERFace and COCO
- Optional Stochastic Weight Averaging (SWA) training
- Utilities for visualization and evaluation

## Installation

Clone the repository and install dependencies:

```bash
pip install torch torchvision numpy pillow
```

Additional packages may be required for dataset evaluation.

## Training

For the WIDERFace dataset:

```bash
python train.py --data_dir /path/to/widerface
```

For COCO keypoints:

```bash
python train_coco.py --data_dir /path/to/coco
```

Both scripts accept standard options such as `--num_epochs`, `--batch_size` and checkpoint paths. Use `--help` to list all arguments.

## Evaluation

The `widerface_evaluate` directory contains scripts to compute validation metrics for WIDERFace. Build the extension first:

```bash
cd widerface_evaluate
python setup.py build_ext --inplace
```

Then run the evaluation:

```bash
python evaluation.py -p <predictions_dir> -g <ground_truth_dir>
```

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

