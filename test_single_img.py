import os
from fast_plate_ocr import LicensePlateRecognizer

# Replace this with the actual image_path testing
image_path = 'images/' + input("Enter the image file name: ")
correct_label = input("Enter the correct license plate label: ")

if not os.path.exists(image_path):
    print(f"Error: The file '{image_path}' was not found in this folder.")
else:
    print("Loading model...")
    m = LicensePlateRecognizer('cct-xs-v1-global-model')

    print(f"Processing {image_path}...")
    prediction = "".join(char for char in m.run(image_path)[0] if char.isalnum())
    string_acc = 1 if correct_label == prediction else 0

    matches = sum(p == l for p, l in zip(prediction, correct_label))
    max_len = max(len(prediction), len(correct_label))
    
    char_acc = matches / max_len if max_len > 0 else 0
    
    print("----------------")
    print(f"Predicted License Plate: {prediction}")
    print(f"Actual License Plate: {correct_label}")
    print(f"String Accuracy: {string_acc}")
    print(f"Character Accuracy: {char_acc:.2%}")
    print("----------------")