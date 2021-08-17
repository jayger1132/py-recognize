import argparse
import time
import os
import numpy as np
import pandas as pd
import functools
import operator
import requests
import base64
import cv2
import csv
import sys
from sys import platform
import argparse
import pandas

def detection(img):
    
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder
            # (Release/x64 etc.)
            sys.path.append('D:\\openpose\\openpose-master\\build\python\\openpose\\Release')
            os.add_dll_directory('D:\\openpose\\openpose-master\\build\\x64\\Release')
            os.add_dll_directory('D:\\openpose\\openpose-master\\build\\bin')
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder
            # (Release/x64 etc.)
            sys.path.append('../../python')
            # If you run `make install` (default path is `/usr/local/python`
            # for Ubuntu), you can also access the OpenPose/python module from
            # there.  This will install OpenPose and the python library at your
            # desired installation path.  Ensure that this is in your python
            # path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e
    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="./data/imgs/from_video/Biceps_curl/bending_test/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    #parse_known_args()使用時機是argument不只有一個，當命令中傳入之後才會用到的選項時不會報錯而是先存起來保留到之後使用
    args = parser.parse_known_args()
    #print(args[0].image_dir)
    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "./models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1: next_item = args[1][i + 1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    
    datum.cvInputData = img
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    cv2.waitKey(0)

video_path = './data/video/3.mp4'
dim = (720, 720)

cap = cv2.VideoCapture(video_path)
# ret為T/F 代表有沒有讀到圖片 frame 是一偵
#ret, frame = cap.read()
ret, img = cap.read()
# cv2.CAP_PROP_FPS 輸出影片FPS
video_fps = cap.get(cv2.CAP_PROP_FPS)

video_size = (img.shape[1], img.shape[0])
print('Videofps',video_fps)
print(video_size)

start_handle_time = time.time()

img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
## 模糊可能還要測試 越高辨識率會變差，越低誤判率會變高 要找合適的中間值
img = cv2.GaussianBlur(img, (7,7), 0)

detection(img)
cv2.waitKey(0)