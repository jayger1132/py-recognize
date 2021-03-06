
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
import tensorflow as tf
from tensorflow import keras
class MyArgs():
    def __init__(self):
        self.video_path = './data/video/Crunch/4.mp4'
        self.model = './data/model/model_Crunch'
args = MyArgs()
dim = (720, 720)

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

def detection(img):
    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="./data/imgs/from_video/Biceps_curl/bending_test/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    #parse_known_args()???????????????argument?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
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
    if datum.poseKeypoints is None:
            print("Noneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
            return(4,img)
    # ????????????????????????,?????????????????????????????????
    tmp_data = []
    target = []
    for i in range(0,25):
        tmp_data.append({ 'x': str(datum.poseKeypoints[0][i][0]), 'y': str(datum.poseKeypoints[0][i][1]), 'score': str(datum.poseKeypoints[0][i][2])})
        #print(str(datum.poseKeypoints[0][i][0]),str(datum.poseKeypoints[0][i][1]),str(datum.poseKeypoints[0][i][2]))
    df = pd.DataFrame(tmp_data)
    #print(df.values)
    data = np.array(df.values)
    data = data.reshape((1, 75))
    #print(data)
    #cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    img2=datum.cvOutputData
    #??????
    with tf.Graph().as_default():
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
        #????????????load???????????????#
        args = MyArgs()
        model = keras.models.load_model(args.model)
        #reshap[1,1,1,...~75???]
        model.predict(np.ones((1, 75)))
        #model.predict_classes(test)?????????????????? ???model.predict(test) ??????????????????
        rest = model.predict(data,verbose=1)
        print(rest)
        #rest = model.predict_classes(data,verbose=0)
        #print("=================",rest[0])
        #if rest[0] == 0:
        #    print('Straight------------')
        #    return (0, img2)
        #elif rest[0] == 1:
        #    print('90-------------')
        #    return (1, img2)
        #elif rest[0] == 2:
        #    print('Underknee-------------')
        #    return (2, img2)
    #print(df.iloc[:,:])
    return(3,img2)
    
cap = cv2.VideoCapture(args.video_path)
# ret???T/F ??????????????????????????? frame ?????????
#ret, frame = cap.read()
ret, img = cap.read()
# cv2.CAP_PROP_FPS ????????????FPS
video_fps = cap.get(cv2.CAP_PROP_FPS)

video_size = (img.shape[1], img.shape[0])
print('Videofps',video_fps)
print(video_size)

start_handle_time = time.time()
count =-1
while ret :
    
    ret, img = cap.read()
    count += 1
    if (count % 10) !=0:
        continue
    if ret == True:
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        ## ???????????????????????? ??????????????????????????????????????????????????? ????????????????????????
        img = cv2.GaussianBlur(img, (7,7), 0)
        #??????
        h = dim[0]
        w = dim[1]
        center = (h/2,w/2)
        #rotate ?????????????????? , "90"????????????90???, "-90"????????????90???, 1?????????1???
        R = cv2.getRotationMatrix2D(center,-90,1)
        rotate = cv2.warpAffine(img,R,(w,h))
        ## ??????
        (is_okay,target2)=detection(rotate)
        print(is_okay,"===============================================")
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", target2)
        cv2.waitKey(100)
    else :
        break