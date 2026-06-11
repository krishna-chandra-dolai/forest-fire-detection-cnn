import argparse
from pathlib import Path

import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from src.config import DEFAULT_MODEL_PATH, IMAGE_SIZE


def load_image(image_path: str):
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return preprocess_input(array)


def main():
    parser = argparse.ArgumentParser(description="Predict fire or no-fire from an image.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to trained model.")
    parser.add_argument("--image", required=True, help="Path to image for prediction.")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    model = load_model(args.model)
    prepared = load_image(args.image)
    probability = float(model.predict(prepared)[0][0])
    label = "Fire" if probability >= args.threshold else "No Fire"

    print(f"Prediction: {label}")
    print(f"Fire probability: {probability:.4f}")


if __name__ == "__main__":
    main()
