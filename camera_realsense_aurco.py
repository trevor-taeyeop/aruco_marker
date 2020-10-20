## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################


import os 
import sys
import pdb
pdb.set_trace()
sys.path = ['/home/rainbow/workspace/aruco_marker', '/home/rainbow/anaconda3/envs/aruco/lib/python36.zip', '/home/rainbow/anaconda3/envs/aruco/lib/python3.6', '/home/rainbow/anaconda3/envs/aruco/lib/python3.6/lib-dynload', '/home/rainbow/anaconda3/envs/aruco/lib/python3.6/site-packages']

import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
# pipeline.start(config)
pipe_profile = pipeline.start(config)
font = cv2.FONT_HERSHEY_SIMPLEX

try:
    for i in range(100000):
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

        depth_sensor = pipe_profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)  # Change
        # add aruco marker
        ############################################################################

        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)  # Use 5x5 dictionary to find markers
        parameters = aruco.DetectorParameters_create()  # Marker detection parameters

        # lists of ids and the corners beloning to each id
        cam_matrix = np.array([[color_intrin.fx,0,color_intrin.ppx],[0,color_intrin.fy,color_intrin.ppy],[0,0,1]])
        dist = np.array([color_intrin.coeffs])
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters, cameraMatrix=cam_matrix,distCoeff=dist)

        if np.all(ids != None):
            # estimate pose of each marker and return the values
            # rvet and tvec-different from camera coefficients
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, cam_matrix, dist)
            for i in range(0, ids.size):
                # draw axis for the aruco markers
                aruco.drawAxis(color_image, cam_matrix, dist, rvec[i], tvec[i], 0.05)
            # draw a square around the markers
            aruco.drawDetectedMarkers(color_image, corners)

            # code to show ids of the marker found
            strg = ''
            for i in range(0, ids.size):
                strg = str(ids[i][0])
                center = corners[i][0].min(0)
                cv2.putText(color_image, strg, (center[0], center[1]), font, 0.5, (255, 0,0 ), 1, cv2.LINE_AA)
        else:
            # code to show 'No Ids' when no markers are found
            cv2.putText(color_image, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        ###############################################################################

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image), cv2.COLORMAP_BONE)
        gray_colormap = cv2.applyColorMap(cv2.convertScaleAbs(gray), cv2.COLORMAP_BONE)

        # Stack both images horizontally
        # images = np.hstack((depth_colormap))
        # images = np.hstack((gray_colormap,color_image))
        images = np.hstack((gray_colormap,color_image))
        images = cv2.resize(images,(1500,1500))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('RealSense',500,100)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)
finally:
    # Stop streaming
    pipeline.stop()