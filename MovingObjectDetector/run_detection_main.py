import numpy as np
import cv2
import matplotlib.pyplot as plt
from BackgroundModel import BackgroundModel
from DetectionRefinement import DetectionRefinement
import sys
import BaseFunctions as basefunctions
import timeit
from KalmanFilter import KalmanFilter
from copy import copy
from BaseFunctions2 import TimePropagate, TimePropagate_, draw_error_ellipse2d
import hdf5storage
from copy import deepcopy as deepcopy
import os
import math


class location:
    def __init__(self, x, y, delta=None, points=None):
        self.x = x
        self.y = y
        self.delta = delta
        self.points = points


def distance(loc1, loc2):
    x_diff = loc1.x - loc2.x
    y_diff = loc1.y - loc2.y
    return math.sqrt(x_diff * x_diff + y_diff * y_diff)


## to measure the difference between two tracks
## each track is a vector of locations
def diff_mean(track1, track2):
    n = len(track1)
    res = 0
    for i in range(0, n):
        res += distance(track1[i], track2[i])
    return res * 1.0 / n


def diff_max(track1, track2):
    n = len(track1)
    res = 0
    for i in range(0, n):
        tmp = distance(track1[i], track2[i])
        if res < tmp:
            res = tmp
    return res
    
    
####


def run_detection_main(attack,model_folder,imagefolder,input_image_idx,ROI_centre,writeimagefolder,ROI_window,num_of_template): 

    ## to run the WAMI tracker
    ## d_out  : output directory
    ## frames : a vector of frames to attack
    ref_track = None

    image_idx_offset = 0
    # if not os.path.exists(d_out):
    #  os.makedirs(d_out)
    model_binary, aveImg_binary, model_regression, aveImg_regression = basefunctions.ReadModels(model_folder)

    # load transformation matrices
    matlabfile = hdf5storage.loadmat(model_folder+'Data/TransformationMatrices_train.mat')
    TransformationMatrices = matlabfile.get("TransMatrix")

    # Load background
    images = []
    for i in range(num_of_template):
        frame_idx = input_image_idx + image_idx_offset + i - num_of_template
        ReadImage = cv2.imread(imagefolder + "frame%06d.png" % frame_idx, cv2.IMREAD_GRAYSCALE)
        ReadImage = ReadImage[ROI_centre[1] - ROI_window:ROI_centre[1] + ROI_window + 1,
                    ROI_centre[0] - ROI_window:ROI_centre[0] + ROI_window + 1]
        ROI_centre = TimePropagate_(ROI_centre, TransformationMatrices[frame_idx - 1][0])
        ROI_centre = [int(i) for i in ROI_centre]
        images.append(ReadImage)
    bgt = BackgroundModel(num_of_template=num_of_template, templates=images)

    # initialise Kalman filter
    kf = KalmanFilter(np.array([[722], [1487], [0], [0]]), np.diag([900, 900, 400, 400]), 5, 6)
    # kf1 = KalmanFilter(np.array([[2989], [1961], [0], [0]]), np.diag([900, 900, 400, 400]), 5, 6)
    detections_all = []
    detected_track = []
    for i in range(20):
        refinementID = None
        starttime = timeit.default_timer()
        # Read input image
        frame_idx = input_image_idx + image_idx_offset + i
        ReadImage = cv2.imread(imagefolder + "frame%06d.png" % frame_idx, cv2.IMREAD_GRAYSCALE)
        input_image = ReadImage[ROI_centre[1] - ROI_window:ROI_centre[1] + ROI_window + 1,
                      ROI_centre[0] - ROI_window:ROI_centre[0] + ROI_window + 1]
        ROI_centre = TimePropagate_(ROI_centre, TransformationMatrices[frame_idx - 1][0])
        ROI_centre = [int(i) for i in ROI_centre]

        Hs = bgt.doCalculateHomography(input_image)
        bgt.doMotionCompensation(Hs, input_image.shape)
        BackgroundSubtractionCentres, BackgroundSubtractionProperties = bgt.doBackgroundSubtraction(input_image, thres=8)

        dr = DetectionRefinement(input_image, bgt.getCompensatedImages(), BackgroundSubtractionCentres,
                                 BackgroundSubtractionProperties, model_binary, aveImg_binary, model_regression,
                                 aveImg_regression,attack)
        # dr.refinementID=refinementID
        refinedDetections, refinedProperties = dr.doMovingVehicleRefinement()
        regressedDetections = dr.doMovingVehiclePositionRegression()
        regressedDetections = np.asarray(regressedDetections)

        # Kalman filter update
        kf1 = deepcopy(kf)
        # kf1_flag=False
        if i > 0:
            kf.TimePropagate(Hs[num_of_template - 1])
            kf.predict()
            """"""
            # the id in the regressed detections
            regressionID = kf.NearestNeighbourAssociator(regressedDetections)
            # the id in the refinement detections (input to the CNN)
            old_kfz = kf.z
            # print("regressionID:   ")
            # print(regressionID)
            # print("regressedDetectionID:   ")
            # print(dr.regressedDetectionID)
            if isinstance(regressionID, np.int64):
                regression2refinedID = dr.regressedDetectionID[regressionID]
                refinementID = dr.refinedDetectionsID[regression2refinedID]
                print("Background subtraction id:" + str(refinementID))
                print("Background subtraction id type:" + str(type(refinementID)))
            else:
                print("Data Association failed (No detection is assigned to this track)...")
            if isinstance(refinementID, np.int64) and (i > 5):
                #######
                #######  here to play 'attack': to call the dr again with refinementID
                # frame_idx = input_image_idx+image_idx_offset+i
                # ReadImage = cv2.imread(imagefolder + "frame%06d.png" % frame_idx, cv2.IMREAD_GRAYSCALE)
                # input_image = ReadImage[ROI_centre[1]-ROI_window:ROI_centre[1]+ROI_window+1, ROI_centre[0]-ROI_window:ROI_centre[0]+ROI_window+1]
                # ROI_centre = TimePropagate_(ROI_centre, TransformationMatrices[frame_idx-1][0])
                # ROI_centre = [int(i) for i in ROI_centre]

                # Hs = bgt.doCalculateHomography(input_image)
                # bgt.doMotionCompensation(Hs, input_image.shape)
                # BackgroundSubtractionCentres, BackgroundSubtractionProperties = bgt.doBackgroundSubtraction(input_image, thres=10)

                # dr = DetectionRefinement(input_image, bgt.getCompensatedImages(), BackgroundSubtractionCentres, BackgroundSubtractionProperties, model_binary, aveImg_binary, model_regression, aveImg_regression)
                #########
                dr.refinementID = refinementID
                # dr.refinementID=None
                refinedDetections, refinedProperties = dr.doMovingVehicleRefinement()
                regressedDetections = dr.doMovingVehiclePositionRegression()
                regressedDetections = np.asarray(regressedDetections)
                dr.refinementID = None

                kf1.TimePropagate(Hs[num_of_template - 1])
                kf1.predict()
                # the id in the regressed detections
                print('==== old regressionID', regressionID)
                regressionID = kf1.NearestNeighbourAssociator(regressedDetections)
                new_kfz = kf1.z
                print('*********************')
                print(old_kfz)
                print('####')
                print(new_kfz)
                print('*********************')
                print('==== new regressionID', regressionID)
                # the id in the refinement detections (input to the CNN)
                print('#### old refinementID', refinementID)
                if regressionID is None:
                    print('#### new refinementID does not exist, because there is no associated detection')
                else:
                    regression2refinedID = dr.regressedDetectionID[regressionID]
                    refinementID = dr.refinedDetectionsID[regression2refinedID]
                    print('#### new refinementID', refinementID)
                # kf1_flag=True
                kf = deepcopy(kf1)
                #######
                #######

            # if kf1_flag: kf=deepcopy(kf1)
            """"""
            kf.update()
            trackx = kf.mu_t[0, 0]
            tracky = kf.mu_t[1, 0]
            # if kf1_flag:
            #  kf1.update()
            #  trackx = kf1.mu_t[0,0]
            #  tracky = kf1.mu_t[1,0]
            # propagate all detections
            detections_all = TimePropagate(detections_all, Hs[num_of_template - 1])
            detections_all.append(np.array([trackx, tracky]).reshape(2, 1))
            print('Estimated State: ' + str(kf.mu_t.transpose()))
        else:
            trackx = kf.mu_t[0, 0]
            tracky = kf.mu_t[1, 0]
            # if kf1_flag:
            #  trackx = kf1.mu_t[0,0]
            #  tracky = kf1.mu_t[1,0]
            detections_all.append(np.array([trackx, tracky]).reshape(2, 1))

        # update background
        bgt.updateTemplate(input_image)

        # plt.figure()
        minx = np.int32(trackx - 300)
        miny = np.int32(tracky - 300)
        maxx = np.int32(trackx + 301)
        maxy = np.int32(tracky + 301)
        roi_image = np.repeat(np.expand_dims(input_image[miny:maxy, minx:maxx], -1), 3, axis=2)
        cv2.circle(roi_image, (301, 301), 10, (255, 0, 0), 1)
        validRegressedDetections = np.int32(copy(regressedDetections))
        validRegressedDetections[:, 0] = validRegressedDetections[:, 0] - minx
        validRegressedDetections[:, 1] = validRegressedDetections[:, 1] - miny
        for thisDetection in validRegressedDetections:
            if thisDetection[0] > 0 and thisDetection[0] < 600 and thisDetection[1] > 0 and thisDetection[1] < 600:
                cv2.circle(roi_image, (thisDetection[0], thisDetection[1]), 3, (100, 100, 0), -1)

        num_of_detections_all = len(detections_all)

        point1s = [None]
        point2s = [None]

        print('###')
        for idx in range(1, num_of_detections_all):
            point1x = np.int32(detections_all[idx - 1][0, 0]) - minx
            point1y = np.int32(detections_all[idx - 1][1, 0]) - miny
            point2x = np.int32(detections_all[idx][0, 0]) - minx
            point2y = np.int32(detections_all[idx][1, 0]) - miny
            point1s.append((point1x, point1y))
            point2s.append((point2x, point2y))
            if i > 0 and not (ref_track is None):
                print(idx, ref_track[i].points[0][idx], ref_track[i].points[1][idx])
                cv2.line(roi_image, ref_track[i].points[0][idx], ref_track[i].points[1][idx], (0, 255, 0), 1)
            if ref_track is None:
                cv2.line(roi_image, (point1x, point1y), (point2x, point2y), (0, 255, 0), 1)
            else:
                cv2.line(roi_image, (point1x, point1y), (point2x, point2y), (0, 0, 255), 1)
        print('###')

        detected_track.append(location(kf.mu_t[0] - minx, kf.mu_t[1] - miny, kf.sigma_t, [point1s, point2s]))

        # draw_error_ellipse2d(roi_image, (kf1.mu_t[0]-minx, kf1.mu_t[1]-miny), kf1.sigma_t)
        # cv2.circle(input_image, (np.int32(trackx), np.int32(tracky)), 15, (255, 0, 0), 3)
        # cv2.imwrite(writeimagefolder + "%05d.png"%i, roi_image)
        #print("writing into %s"%(writeimagefolder + "%05d.png" % i))
        cv2.imwrite(writeimagefolder + "%05d.png" % i, roi_image)
        """
        plt.figure()
        plt.imshow(np.repeat(np.expand_dims(input_image, -1), 3, axis=2))
        #plt.plot(BackgroundSubtractionCentres[:,0], BackgroundSubtractionCentres[:,1], 'g.')
        #plt.plot(refinedDetections[:,0], refinedDetections[:,1], 'y.')
        plt.plot(np.int32(regressedDetections[:,0]), np.int32(regressedDetections[:,1]), 'r.', markersize=3)
        plt.plot(np.int32(trackx), np.int32(tracky), 'yo', markersize=5)
        plt.show()
        """
        endtime = timeit.default_timer()
        print("Processing Time (Total): " + str(endtime - starttime) + " s... ")


