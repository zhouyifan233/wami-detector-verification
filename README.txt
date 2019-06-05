The tool integrates WAMI moving vehicle detector and DeepConcolic.

To start the tool
==============
cd MovingObjectDetector
python run_detection_batch


To run the tool. There needs to be come pre-condigurations.
============================================================
1. DeepConcolic is installed
2. You have to change the system path hard-coded in some files, including
   MovingObjectDetector/DetectionRefinement.py: 
      line 9, sys.path.insert(0, '/home/syc/Dropbox/workspace/liverpool/DeepConcolic/src')

   MovingObjectDetector/run_detection_batch.py: 
      line 23, imagefolder = imagefolder = "/home/syc/workspace/gits/WAMI-detection-matlab/WPAFB-images/png/WAPAFB_images_train/training/" #"C:/WPAFB-images/training/"
      line 24, writeimagefolder = "./savefig/"
      line 25, model_folder = "/home/syc/Dropbox/workspace/liverpool/wasabi-detection-python-shared/Models/"
      line 29, matlabfile = hdf5storage.loadmat('/home/syc/Dropbox/workspace/liverpool/wasabi-detection-python-shared/Models/Data/TransformationMatrices_train.mat')


To change the frame under sttack
=================================
Changing the 'i' at line 80 in run_detection_batch.py
Attack happens inside DetectionRefinement.py  "if not self.refinementID is None"
