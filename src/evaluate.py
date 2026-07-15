import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import BATCH_SIZE, DEFAULT_MODEL_PATH, IMAGE_SIZE
from src.labeling import labels_from_class_indices


def validate_test_dataset(test_dir: str) -> None:
    required_dirs = [
        Path(test_dir) / "fire",
        Path(test_dir) / "no_fire",
    ]

    if any(not path.is_dir() for path in required_dirs):
        raise FileNotFoundError(
            "Test dataset directory not found or incomplete.\n\n"
            "Expected structure:\n"
            "data/test/fire\n"
            "data/test/no_fire\n\n"
            "Download the dataset and place it in the expected folders before evaluation."
        )


def build_test_generator(test_dir: str, batch_size: int):
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    return datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )


def save_confusion_matrix(y_true, y_pred, labels, output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained forest fire detection model.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", default="models/evaluation")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    validate_test_dataset(args.test_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model)
    test_generator = build_test_generator(args.test_dir, args.batch_size)
    labels = labels_from_class_indices(test_generator.class_indices)

    loss, accuracy = model.evaluate(test_generator)
    probabilities = model.predict(test_generator).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    report = classification_report(test_generator.classes, predictions, target_names=labels)

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(report)

    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(
        test_generator.classes,
        predictions,
        labels,
        output_dir / "confusion_matrix.png",
    )


if __name__ == "__main__":
    main()
