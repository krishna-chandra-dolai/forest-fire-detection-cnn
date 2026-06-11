# Models

Trained model files are not committed to this repository.

After training, the default model output path is:

```text
models/forest_fire_resnet50.keras
```

Prediction uses this path by default:

```bash
python -m src.predict --model models/forest_fire_resnet50.keras --image path/to/image.jpg
```

Evaluation outputs are generated locally and are not committed:

```text
models/evaluation/classification_report.txt
models/evaluation/confusion_matrix.png
```
