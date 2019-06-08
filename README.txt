The tool integrates WAMI moving vehicle detector and DeepConcolic.

To start the tool
==============
python main.py [--attack True/False] [--wasabi-image-folder path/to/wasabi/image/folder] [--input_image_idx n] [--ROI_centre (x,y)] [-- output-image-folder path/to/output/folder] 


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
    
