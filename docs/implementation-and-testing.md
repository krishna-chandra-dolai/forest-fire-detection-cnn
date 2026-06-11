# Implementation And Testing

This document converts the implementation and testing details from the project report into repository-level code references.

## Report Pages Used

The implementation and testing details were reviewed from the report section covering:

- Importing required libraries
- Data preprocessing and augmentation
- ResNet-50 model design
- Epoch-wise performance overview
- Confusion matrix and classification report
- Test cases TC_01 through TC_12

## Implementation Code

The report's implementation is represented in:

- `src/train.py`
- `src/predict.py`
- `src/evaluate.py`
- `src/test_cases.py`

The training pipeline follows the report:

- ResNet-50 transfer learning
- 224 x 224 image input
- `ImageDataGenerator` augmentation
- Rotation, width shift, height shift, shear, zoom, and horizontal flip
- Global average pooling
- Dense layer with ReLU activation
- Dropout
- Sigmoid output layer
- Adam optimizer
- Early stopping
- 20 training epochs

## Training

```bash
python -m src.train --train-dir data/train --val-dir data/test --epochs 20 --model-out models/forest_fire_resnet50.keras
```

## Evaluation

```bash
python -m src.evaluate --model models/forest_fire_resnet50.keras --test-dir data/test --output-dir models/evaluation
```

This generates:

- `models/evaluation/classification_report.txt`
- `models/evaluation/confusion_matrix.png`

## Single Image Prediction

```bash
python -m src.predict --model models/forest_fire_resnet50.keras --image path/to/image.jpg
```

## Report-Style Test Cases

```bash
python -m src.test_cases --model models/forest_fire_resnet50.keras --fire-image path/to/fire.jpg --no-fire-image path/to/no_fire.jpg --outlier-image path/to/outlier.jpg
```

The script covers the practical tests described in the report:

- TC_01 model loading
- TC_02 fire image classification
- TC_03 non-fire image classification
- TC_05 rotation robustness
- TC_06 blur robustness
- TC_07 lighting condition robustness
- TC_08 outlier image handling

The remaining tests require full datasets or live streams:

- TC_04 generalization on the complete test dataset
- TC_10 large-batch performance
- TC_11 scalability on continuous streams
- TC_12 post-deployment real-world image testing

## Supporting Screenshots

The relevant implementation and output figures from the report are stored in:

- `docs/screenshots/cnn-report/cnn-imports-code.png`
- `docs/screenshots/cnn-report/cnn-data-augmentation-code.png`
- `docs/screenshots/cnn-report/cnn-model-training-code.png`
- `docs/screenshots/cnn-report/cnn-confusion-matrix-and-report.png`
