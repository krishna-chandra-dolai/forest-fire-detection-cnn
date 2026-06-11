# Project Structure

This repository is organized to keep the main CNN/ResNet-50 forest fire image classification project clear and review-ready.

## `src/`

Python source code for training and prediction.

- `config.py`: shared image size, batch size, epoch count, and default model path.
- `train.py`: ResNet-50 transfer-learning training pipeline.
- `predict.py`: single-image prediction script.

## `notebooks/`

Jupyter Notebook work for the main CNN project.

- `forest_fire_detection_cnn_resnet50.ipynb`: notebook version of the CNN/ResNet-50 workflow.

## `data/`

Dataset instructions only. The actual CNN image dataset is not committed.

Expected folders after downloading the dataset:

```text
data/train/fire
data/train/no_fire
data/test/fire
data/test/no_fire
```

## `models/`

Model output instructions only. Trained model files are not committed.

Default model output after training:

```text
models/forest_fire_resnet50.keras
```

## `reports/`

Main project report PDF for the CNN/ResNet-50 forest fire detection project.

## `presentation/`

Final project presentation PDF.

## `docs/`

Project documentation, local file inventory, detailed project notes, and screenshots.

## `legacy/`

Old or unrelated CSV-based forest-fire burned-area prediction materials. These are kept for reference but are separated from the main CNN image-classification project.
