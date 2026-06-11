import argparse
import tempfile
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter
from tensorflow.keras.models import load_model

from src.config import DEFAULT_MODEL_PATH
from src.predict import load_class_indices, predict_image


def print_result(test_id: str, description: str, passed: bool, detail: str) -> None:
    status = "PASSED" if passed else "FAILED"
    print(f"{test_id}: {status} - {description}")
    print(f"  {detail}")


def run_prediction_case(model, class_indices, test_id: str, description: str, image_path: str, expected: str, threshold: float) -> None:
    if not image_path:
        print_result(test_id, description, False, "Skipped because no image path was provided.")
        return
    if not Path(image_path).exists():
        print_result(test_id, description, False, f"Image not found: {image_path}")
        return

    label, probability = predict_image(model, image_path, class_indices, threshold)
    passed = label.lower() == expected.lower()
    print_result(test_id, description, passed, f"Expected: {expected}; predicted: {label}; fire probability: {probability:.4f}")


def save_transformed_image(image_path: str, output_dir: Path, mode: str) -> Path:
    source = Image.open(image_path).convert("RGB")
    if mode == "rotated":
        transformed = source.rotate(90, expand=True)
    elif mode == "blurred":
        transformed = source.filter(ImageFilter.GaussianBlur(radius=2))
    elif mode == "bright":
        transformed = ImageEnhance.Brightness(source).enhance(1.5)
    elif mode == "dim":
        transformed = ImageEnhance.Brightness(source).enhance(0.55)
    else:
        raise ValueError(f"Unknown transform mode: {mode}")

    output_path = output_dir / f"{mode}_{Path(image_path).name}"
    transformed.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run report-style forest fire detection test cases.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--fire-image", help="Known fire image for positive tests.")
    parser.add_argument("--no-fire-image", help="Known no-fire image for negative tests.")
    parser.add_argument("--outlier-image", help="Non-forest/non-fire image expected as No Fire.")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = load_model(args.model)
    class_indices = load_class_indices(args.model)
    print_result("TC_01", "Model Load Test", True, f"Loaded model from {args.model}")

    run_prediction_case(model, class_indices, "TC_02", "Fire Image Classification", args.fire_image, "Fire", args.threshold)
    run_prediction_case(model, class_indices, "TC_03", "Non-Fire Image Classification", args.no_fire_image, "No Fire", args.threshold)

    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        if args.fire_image and Path(args.fire_image).exists():
            rotated = save_transformed_image(args.fire_image, temp_dir, "rotated")
            blurred = save_transformed_image(args.fire_image, temp_dir, "blurred")
            bright = save_transformed_image(args.fire_image, temp_dir, "bright")
            dim = save_transformed_image(args.fire_image, temp_dir, "dim")
            run_prediction_case(model, class_indices, "TC_05", "Rotation Test", str(rotated), "Fire", args.threshold)
            run_prediction_case(model, class_indices, "TC_06", "Blurred Image Test", str(blurred), "Fire", args.threshold)
            run_prediction_case(model, class_indices, "TC_07A", "Bright Lighting Test", str(bright), "Fire", args.threshold)
            run_prediction_case(model, class_indices, "TC_07B", "Dim Lighting Test", str(dim), "Fire", args.threshold)
        else:
            print_result("TC_05", "Rotation Test", False, "Skipped because no fire image was provided.")
            print_result("TC_06", "Blurred Image Test", False, "Skipped because no fire image was provided.")
            print_result("TC_07", "Lighting Condition Test", False, "Skipped because no fire image was provided.")

    run_prediction_case(model, class_indices, "TC_08", "Outlier Image Test", args.outlier_image, "No Fire", args.threshold)

    print("TC_04, TC_10, TC_11, and TC_12 should be validated with the full test set or live image stream.")
    print("Use src.evaluate for full test-set accuracy, classification report, and confusion matrix output.")


if __name__ == "__main__":
    main()
