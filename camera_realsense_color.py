## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import os 
import sys
import pdb
import rospy

try:
    sys.path = ['/home/rainbow/workspace/aruco_marker', '/home/rainbow/anaconda3/envs/aruco/lib/python36.zip', '/home/rainbow/anaconda3/envs/aruco/lib/python3.6', '/home/rainbow/anaconda3/envs/aruco/lib/python3.6/lib-dynload', '/home/rainbow/anaconda3/envs/aruco/lib/python3.6/site-packages']
except:
    pdb.set_trace()
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from datetime import datetime
import copy


def deproject(center,depth,K,pose=None):
    """
    center.shape = [1,2]
    depth.shape = [1,1]
    K.shape = [3,3]
    """
    out_gt = center * depth
    out_gt = np.concatenate((out_gt, depth), 1)
    # out_gt = [1,3]
    inv_K = np.linalg.inv(K.T)
    xyz = np.dot(out_gt, inv_K)
    return xyz

def project(xyz, K):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
W = 640
H = 480

# video capture for test video backup
# cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# filename = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
# out = cv2.VideoWriter(filename+'.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (W,H))
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

        cam_matrix = np.array([[color_intrin.fx,0,color_intrin.ppx],[0,color_intrin.fy,color_intrin.ppy],[0,0,1]])

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image), cv2.COLORMAP_BONE)
        # gray_colormap = cv2.applyColorMap(cv2.convertScaleAbs(gray), cv2.COLORMAP_BONE)

        ###############################################################################
        ###### red marker detection
        
        ##################################
        # change color value
        low_red = np.array([161, 155, 84])   # (161,155,84)
        high_red = np.array([179, 255, 255]) # (179,255,255)
        ##################################

        img_mask = cv2.inRange(img_hsv, low_red, high_red)
        img_mask = cv2.medianBlur(img_mask,7)
        img_result = cv2.bitwise_and(color_image, color_image, mask = img_mask)
        # cv2.imshow('img_result', img_result)
       
        ret, gray_th = cv2.threshold(gray,50,255,cv2.THRESH_BINARY_INV)
        ###############################################################################
        ###### houghLineP      # https://076923.github.io/posts/Python-opencv-28/

        # canny = cv2.Canny(img_mask, 5000, 1500, apertureSize = 5, L2gradient = True)
        lines = cv2.HoughLinesP(img_mask, 1, np.pi, 30, minLineLength = 10, maxLineGap = 50)
        pixel_x = 0
        pixel_y = 0
        if lines is not None:
            for i in lines:
                # print("a: {}, b: {}, c: {}, d: {}".format(i[0][0], i[0][1], i[0][2], i[0][3]))
                cv2.line(color_image, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 2)
                pixel_x += np.mean([i[0][0], i[0][2]])
                pixel_y += np.mean([i[0][1], i[0][3]])
            pixel_x = int(round(pixel_x / len(lines)))
            pixel_y = int(round(pixel_y / len(lines)))
            cv2.line(color_image, (pixel_x, 0), (pixel_x, H), (0, 0, 255), 2)
        

        # visualize guide line
        left = int(W*0.4)
        right = int(W*0.6)
        cv2.line(color_image, (left, 0), (left, H), (0, 255, 0), 2)
        cv2.line(color_image, (right, 0), (right, H), (0, 255, 0), 2)
        

        if pixel_x in range(left, right):
            cv2.putText(color_image, "Grap!", (280, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elif pixel_x < left:
            cv2.putText(color_image, "right ->", (170, 64), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        elif pixel_x > right:
            cv2.putText(color_image, "<- left", (350, 64), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(color_image, "", (350, 64), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        

        cv2.circle(color_image, (pixel_x, pixel_y), 10, (255, 0, 0), 4)

        ##############################################################################
        # output results
        depth_pixel = depth_image[pixel_y,pixel_x]/1000.0
        # predict camera coordinate xyz
        # xyz.shape = [3], meter 
        xyz = deproject(np.array([pixel_x,pixel_y]).reshape(1,2),np.array(depth_pixel).reshape(1,1),K=cam_matrix)[0]
        ##############################################################################

        # cm 
        visual_xyz = np.round(xyz * 1000)
        # proj_xy = project(xyz,K=cam_matrix)
        # print("a: {}, b: {}, c: {}".format(visual_xyz[0,0], visual_xyz[0,1], xyz[0,2]))

        cv2.putText(color_img_back, "{}, {}, {}".format(pixel_x, pixel_y, depth_pixel ), (350, 64), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(color_img_back, "{}, {}, {}".format(visual_xyz[0], visual_xyz[1], visual_xyz[2]), (300, 150), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('img_mask', img_mask)
        # cv2.imshow('canny', canny)
        ###############################################################################
        
        # Stack both images horizontally
        # images = np.hstack((depth_colormap))
        # images = np.hstack((gray_colormap,color_image))
        # images = cv2.resize(images,(1500,1500))
        # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.moveWindow('RealSense',500,100)
        cv2.imshow('depth image', depth_colormap)
        cv2.imshow('RealSense', color_image)
        color_img_back
        cv2.imshow("Input", color_img_back)
        # if saveVideo_on:
        #     color_img_back = cv2.flip(color_img_back,180)
        #     out.write(color_img_back)

        key = cv2.waitKey(1)
        # if key == 27: # ESC
        #     break
        # elif key == 32: # space
        #     saveVideo_on = True
        #     print("space")
        # elif key == 115: # s
        #     saveVideo_on = False
        #     print('stop')
finally:
    # Stop streaming
    pipeline.stop()
    # Release everything if job is finished
    # cap.release()
    # out.release()
    cv2.destroyAllWindows()