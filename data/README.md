# Dataset

The CNN image dataset is not included in this repository.

Download the dataset manually and place it in the expected folder structure before training. The project report mentions a Kaggle image dataset with 999 training images and 999 test images.

Dataset source: [Add Kaggle dataset link here]

Expected folder structure:

```text
data/
|-- train/
|   |-- fire/
|   `-- no_fire/
`-- test/
    |-- fire/
    `-- no_fire/
```

Class folder names must be exactly:

- `fire`
- `no_fire`

Training will fail with a clear error if any of these folders are missing.
