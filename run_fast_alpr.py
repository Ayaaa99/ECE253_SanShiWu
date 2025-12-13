import os
from fast_alpr import ALPR


def main() -> None:
    images_dir = "images"

    image_name = input("Enter the image file name under images/ (e.g. test1.jpg): ").strip()
    image_path = os.path.join(images_dir, image_name)

    if not os.path.exists(image_path):
        print(f"Error: file '{image_path}' does not exist. Please make sure the file is under images/.")
        return

    correct_label = input(
        "If you have the ground truth plate label, enter it here (or press Enter to skip): "
    ).strip()
    if correct_label == "":
        correct_label = None

    print("Loading FastALPR model...")
    alpr = ALPR(
        detector_model="yolo-v9-t-256-license-plate-end2end",
        ocr_model="cct-xs-v1-global-model",
    )

    print(f"Processing image: {image_path} ...")
    results = alpr.predict(image_path)

    if not results:
        print("No license plates were detected.")
        return

    print("-" * 30)
    print(f"Detected {len(results)} plates:")

    for idx, res in enumerate(results, start=1):
        ocr_result = getattr(res, "ocr", None)
        if ocr_result is None:
            print(f"[{idx}] No OCR result available")
            continue

        plate_text = ocr_result.text or ""
        confidence = ocr_result.confidence or 0.0

        prediction = "".join(ch for ch in plate_text if ch.isalnum())

        print(f"[{idx}] Predicted plate: {prediction} (raw: '{plate_text}', confidence: {confidence:.2f})")

        if correct_label is not None:
            string_acc = 1 if prediction == correct_label else 0

            matches = sum(
                p == l for p, l in zip(prediction, correct_label)
            )
            max_len = max(len(prediction), len(correct_label))
            char_acc = matches / max_len if max_len > 0 else 0.0

            print("  --- Comparison with ground truth ---")
            print(f"  Ground truth plate: {correct_label}")
            print(f"  String accuracy: {string_acc}")
            print(f"  Character accuracy: {char_acc:.2%}")

    print("-" * 30)


if __name__ == "__main__":
    main()
