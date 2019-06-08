The tool integrates WAMI moving vehicle detector and DeepConcolic.

To start the tool
==============
python main.py 

with a few optional paramters: 

[--attack True/False] with default False
[--wasabi-image-folder path/to/wasabi/image/folder] with default "~/Dropbox/wasabi-detection-python-new/WAPAFB_images_train/training/"
[--input_image_idx n] with default "10"
[--ROI_centre (x,y)] with default "4500, 5000"
[-- output-image-folder path/to/output/folder] with default "../savefig/"


To run the tool. There needs to be come pre-condigurations.
============================================================
1. DeepConcolic is installed
2. change image folder (in main.py): 
    imagefolder = path/to/wasabi/image/folder
3. change which folder to write the result image into (in main.py): 
    writeimagefolder0 = path/to/folder/for/results
    
A few settings (in main.py): 
===========================================================
1. toggle attack or normal by setting
    attack = True / False 
    
