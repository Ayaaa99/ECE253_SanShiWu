import os
from fast_plate_ocr import LicensePlateRecognizer

image_folder = 'images/corrected_angle_dataset'
valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
output_file = 'corrected_angle_dataset_results.txt'

image_files = [
    f for f in os.listdir(image_folder)
    if os.path.splitext(f)[1].lower() in valid_extensions
]

if not image_files:
    print(f"No images found in '{image_folder}'")
else:
    print(f"Found {len(image_files)} images")
    print("Loading model...")
    m = LicensePlateRecognizer('cct-xs-v1-global-model')
    
    total_string_acc = 0
    total_char_acc = 0
    
    with open(output_file, 'w') as f:
        for filename in image_files:
            image_path = os.path.join(image_folder, filename)
            correct_label = os.path.splitext(filename)[0].upper()
            
            prediction = "".join(char for char in m.run(image_path)[0] if char.isalnum())
            
            string_acc = 1 if correct_label == prediction else 0
            matches = sum(p == l for p, l in zip(prediction, correct_label))
            max_len = max(len(prediction), len(correct_label))
            char_acc = matches / max_len if max_len > 0 else 0
            
            total_string_acc += string_acc
            total_char_acc += char_acc
            
            # Print and write to file
            status = "✓" if string_acc == 1 else "✗"
            line = f"{status} {filename}: {prediction} (expected: {correct_label}) - Char Acc: {char_acc:.2%}"
            print(line)
            f.write(line + "\n")
        
        # Summary
        num_images = len(image_files)
        summary = f"""
==================================================
SUMMARY
==================================================
Total Images: {num_images}
Exact Matches: {int(total_string_acc)} / {num_images}
Overall String Accuracy: {total_string_acc / num_images:.2%}
Overall Character Accuracy: {total_char_acc / num_images:.2%}
"""
        print(summary)
        f.write(summary)
    
    print(f"\nResults saved to '{output_file}'")