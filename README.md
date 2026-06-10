# Forest Fire Detection Using CNN

Image-based forest fire detection using a Convolutional Neural Network with the ResNet-50 architecture and transfer learning.

The project classifies an input image into:

- Fire
- No Fire

## Project Summary

Forest fires can spread quickly and cause severe environmental and human damage. This project uses deep learning to detect forest fire from images so that early warning systems can respond faster.

The model described in the project report uses:

- ResNet-50
- Transfer learning with ImageNet weights
- TensorFlow and Keras
- 224 x 224 image input
- Binary fire/no-fire classification
- Data augmentation
- Adam optimizer

Reported performance:

| Metric | Value |
|---|---:|
| Test accuracy | 95.80% |
| Validation accuracy | 96.00% |
| Validation loss | 0.0458 |
| Test loss | 0.1589 |

## Repository Structure

```text
forest-fire-detection-cnn/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- data/
|   `-- README.md
|-- docs/
|   `-- project-details.md
|-- models/
|   `-- README.md
|-- notebooks/
|   `-- README.md
|-- presentation/
|   `-- final-presentation.pdf
|-- related/
|   |-- original-notes.txt
|   `-- predict-the-forest-fires-archive.7z
|-- reports/
|   |-- forest-fire-detection-using-cnn.pdf
|   `-- legacy-ml-project-report.docx
`-- src/
    |-- __init__.py
    |-- config.py
    |-- train.py
    `-- predict.py
```

## Dataset

The original report describes a Kaggle image dataset with two classes:

- `fire`
- `no_fire`

All images are resized to `224 x 224` pixels.

Expected local dataset structure:

```text
data/
|-- train/
|   |-- fire/
|   `-- no_fire/
`-- test/
    |-- fire/
    `-- no_fire/
```

The dataset itself is not included in this repository. Add the image folders locally before training.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py --train-dir data/train --val-dir data/test --epochs 20 --model-out models/forest_fire_resnet50.keras
```

## Predict

```bash
python src/predict.py --model models/forest_fire_resnet50.keras --image path/to/image.jpg
```

## Methodology

1. Collect labelled fire and no-fire images.
2. Resize images to 224 x 224 pixels.
3. Apply preprocessing and augmentation.
4. Load ResNet-50 with ImageNet weights.
5. Replace the final classifier with binary classification layers.
6. Train using Adam optimizer and binary cross-entropy loss.
7. Evaluate using accuracy, precision, recall, F1-score, loss, and confusion matrix.
8. Use the trained model for image prediction or future real-time camera integration.

## Future Scope

- Add larger and more diverse datasets.
- Integrate satellite, drone, or CCTV image feeds.
- Add weather data such as temperature, humidity, wind, and rainfall.
- Try EfficientNet, DenseNet, or Vision Transformers.
- Deploy on edge devices for real-time alerts.
- Add explainability with Grad-CAM.

## Author

Krishna Chandra Dolai

Master of Computer Applications, Andhra University College of Engineering (A), Andhra University.
