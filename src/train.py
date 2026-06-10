import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import BATCH_SIZE, DEFAULT_MODEL_PATH, EPOCHS, IMAGE_SIZE


def build_generators(train_dir: str, val_dir: str, batch_size: int):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    return train_generator, val_generator


def build_model(learning_rate: float):
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_confusion_matrix(y_true, y_pred, labels, output_path: Path):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train a ResNet-50 forest fire detector.")
    parser.add_argument("--train-dir", default="data/train", help="Training dataset folder.")
    parser.add_argument("--val-dir", default="data/test", help="Validation or test dataset folder.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    train_generator, val_generator = build_generators(args.train_dir, args.val_dir, args.batch_size)
    model = build_model(args.learning_rate)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(model_out, monitor="val_loss", save_best_only=True),
    ]

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    loss, accuracy = model.evaluate(val_generator)
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")

    probabilities = model.predict(val_generator).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    labels = list(val_generator.class_indices.keys())

    print(classification_report(val_generator.classes, predictions, target_names=labels))
    plot_confusion_matrix(
        val_generator.classes,
        predictions,
        labels,
        model_out.parent / "confusion_matrix.png",
    )


if __name__ == "__main__":
    main()

