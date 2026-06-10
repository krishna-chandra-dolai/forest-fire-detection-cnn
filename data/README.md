# Dataset Folder

Place the CNN fire/no-fire image dataset here before training.

Expected structure:

```text
data/
|-- train/
|   |-- fire/
|   `-- no_fire/
`-- test/
    |-- fire/
    `-- no_fire/
```

The original report mentions a Kaggle dataset with 999 images in the training set and 999 images in the test or validation set, resized to 224 x 224 pixels.

The actual CNN image dataset was not present in the local folder that was provided. The only dataset file found locally was the legacy CSV dataset from `Predict-the-Forest-Fires.7z`:

- `data/legacy_forestfires.csv`
