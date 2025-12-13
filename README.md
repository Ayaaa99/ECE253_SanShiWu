# ECE253

## Retinex-Net

### Requirements
- Python  
- Tensorflow >= 1.5.0  
- numpy  
- PIL  

### Testing Usage
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

## CLAHE

### Usage
```shell
python lowlight_enhance.py \
    --test_dir /path/to/your/test/dir/ \
    --save_dir /path/to/save/results/
``` 