# ECE253

## Requirements
- Python  
- Tensorflow >= 1.5.0  
- numpy  
- PIL
- fast-plate-ocr

## Low Light

### Retinex-Net Testing Usage
```shell
python main.py \
    --use_gpu=1 \                           # use gpu or not
    --gpu_idx=0 \                           
    --gpu_mem=0.5 \                         # GPU memory usage ratio
    --phase=test \                          
    --test_dir=/path/to/your/test/dir/ \    # directory of test images
    --save_dir=/path/to/save/results/ \     # directory to save results
    --decom=0                               # save only enhanced results or together with decomposition results
``` 

### CLAHE Usage
```shell
python lowlight_enhance.py \
    --test_dir /path/to/your/test/dir/ \
    --save_dir /path/to/save/results/
``` 
## Motion Blur

### Wiener Filter Usage
Set `INPUT_DIR` and `OUTPUT_DIR` inside the script to correct paths and run the code below.
```shell
python restore_wiener.py
```

### CLS Filter Usage
Set `INPUT_DIR` and `OUTPUT_DIR` inside the script to correct paths and run the code below.
```shell
python restore_cls.py
```

### Alternative Smoothing & Polishing Usage
Set paths inside the scripts and run the code below for alternative results.
```shell
python smooth.py
python polish_images.py
```


## Angle Correction

### Hough Transform Usage
Set INPUT_FOLDER and OUTPUT_FOLDER to correct paths and run the code below.
```shell
python correct_angle.py
``` 
### Model Fine Tune Usage
Run the Image_rotation_finetune.ipynb file following instructions inside the file


## OCR Recognition Accuracy Testing
### Test Single Image
Run the code below and enter the image path and correct label text as prompted.
```shell
python test_single_img.py
``` 
### Test Group Images
Set image_folder and output_file to correct paths and run the code below.
```shell
python test_group_imgs.py
``` 
