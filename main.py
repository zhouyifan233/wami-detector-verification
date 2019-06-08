import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append('MovingObjectDetector')
sys.path.append('TrainNetwork')
sys.path.append('SimpleTracker')

import os
from BaseFunctions2 import createImageDirectory
from run_detection_main import run_detection_main

def main():

    attack = False 

    input_image_idx = 10
    ROI_centre = [4500, 5000]

    model_folder = "Models/"
    imagefolder = "/Users/xiaowei/Dropbox/wasabi-detection-python-new/WAPAFB_images_train/training/"
    writeimagefolder0 = "/Users/xiaowei/Dropbox/temporary_buffer/code/savefig/"

    exampleString = "%s_%s_%s/"%(input_image_idx,ROI_centre[0],ROI_centre[1])
    createImageDirectory(writeimagefolder0+exampleString)
    if attack: 
        writeimagefolder = writeimagefolder0+exampleString+"attacked/"
    else: 
        writeimagefolder = writeimagefolder0+exampleString+"original/"
    createImageDirectory(writeimagefolder)
    
    run_detection_main(attack,model_folder,imagefolder,input_image_idx,ROI_centre,writeimagefolder) 


if __name__=="__main__":
  main()


