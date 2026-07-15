import argparse
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from src.config import DEFAULT_MODEL_PATH, IMAGE_SIZE
from src.labeling import fire_probability_from_binary_output, load_class_indices


def load_image(image_path: str):
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array / 255.0


def predict_image(model, image_path: str, class_indices: dict[str, int], threshold: float):
    prepared = load_image(image_path)
    raw_probability = float(model.predict(prepared)[0][0])
    fire_probability = fire_probability_from_binary_output(raw_probability, class_indices)
    label = "Fire" if fire_probability >= threshold else "No Fire"
    return label, fire_probability


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
    class_indices = load_class_indices(args.model)
    label, probability = predict_image(model, args.image, class_indices, args.threshold)

    print(f"Prediction: {label}")
    print(f"Fire probability: {probability:.4f}")


if __name__ == "__main__":
    main()
