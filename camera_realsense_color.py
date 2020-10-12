## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import pdb
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from datetime import datetime
import copy


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
W = 640
H = 480

# video capture for test video backup
# cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
filename = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
out = cv2.VideoWriter(filename+'.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (W,H))
saveVideo_on = False

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
        color_img_back = copy.deepcopy(color_image)
        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)  # Change
        img_hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

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
        # gray_colormap = cv2.applyColorMap(cv2.convertScaleAbs(gray), cv2.COLORMAP_BONE)

        ###############################################################################
        ###### red marker detection

        low_red = np.array([161, 155, 84])
        high_red = np.array([179, 255, 255])
        img_mask = cv2.inRange(img_hsv, low_red, high_red)
        img_mask = cv2.medianBlur(img_mask,7)
        img_result = cv2.bitwise_and(color_image, color_image, mask = img_mask)
        # cv2.imshow('img_result', img_result)
       
        ret, gray_th = cv2.threshold(gray,50,255,cv2.THRESH_BINARY_INV)
        ###############################################################################
        ###### houghLineP      # https://076923.github.io/posts/Python-opencv-28/

        canny = cv2.Canny(img_mask, 5000, 1500, apertureSize = 5, L2gradient = True)
        lines = cv2.HoughLinesP(canny, 1, np.pi / 45, 50, minLineLength = 10, maxLineGap = 50)
        x_cum = 0
        y_xum = 0
        if lines is not None:
            for i in lines:
                # print("a: {}, b: {}, c: {}, d: {}".format(i[0][0], i[0][1], i[0][2], i[0][3]))
                cv2.line(color_image, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 2)
                x_cum += np.mean([i[0][0], i[0][0]])
            x_cum = int(round(x_cum / len(lines)))
            cv2.line(color_image, (x_cum, 0), (x_cum, H), (0, 0, 255), 2)
        
        left = int(W*0.4)
        right = int(W*0.6)
        cv2.line(color_image, (left, 0), (left, H), (0, 255, 0), 2)
        cv2.line(color_image, (right, 0), (right, H), (0, 255, 0), 2)

        if x_cum in range(left, right):
            cv2.putText(color_image, "Grap!", (280, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elif x_cum < left:
            cv2.putText(color_image, "right ->", (170, 64), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        elif x_cum > right:
            cv2.putText(color_image, "<- left", (350, 64), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(color_image, "", (350, 64), font, 1, (0, 0, 255), 2, cv2.LINE_AA)


        cv2.imshow('img_mask', img_mask)
        cv2.imshow('canny', canny)
        ###############################################################################
        
        # Stack both images horizontally
        # images = np.hstack((depth_colormap))
        # images = np.hstack((gray_colormap,color_image))
        # images = cv2.resize(images,(1500,1500))
        # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.moveWindow('RealSense',500,100)
        cv2.imshow('RealSense', color_image)
        color_img_back
        cv2.imshow("Input", color_img_back)
        if saveVideo_on:
            color_img_back = cv2.flip(color_img_back,180)
            out.write(color_img_back)

        key = cv2.waitKey(1)
        if key == 27: # ESC
            break
        elif key == 32: # space
            saveVideo_on = True
            print("space")
        elif key == 115: # s
            saveVideo_on = False
            print('stop')
finally:
    # Stop streaming
    pipeline.stop()
    # Release everything if job is finished
    # cap.release()
    out.release()
    cv2.destroyAllWindows()