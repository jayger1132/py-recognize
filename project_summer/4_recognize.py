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
import numpy as np
from sys import platform
import argparse
import pandas
import tensorflow as tf
from tensorflow import keras
class MyArgs():
    def __init__(self):
        self.video_path = './data/video/Biceps_curl/8.mp4'
        self.model = './data/model/model_Biceps_curl'
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
    params = dict()
    params["model_folder"] = "./models/"

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # 將節點放入陣列中,可以在這裡做節點的處理
    tmp_data = []
    tmp_AVGx = []
    tmp_AVGy = []
    target = []
    offsetneck = float(datum.poseKeypoints[0][1][0] - (img.shape[1] / 2))
    #print (offsetneck,img.shape)
    for i in range(0,25):
        if i != 1:
            if float(datum.poseKeypoints[0][i][0]) != 0.0:
                #print(datum.poseKeypoints[0][i][0])
                tempX = float(datum.poseKeypoints[0][i][0]) - offsetneck
                #print(tempX)
                #datum.posekeypoint 是const 下面的寫法是不行的
                #datum.poseKeypoints[0][i][0]=X+offsetneck
            else:
                None
        else:
            tempX = (img.shape[1]) / 2

        tmp_data.append({ 'x': str(datum.poseKeypoints[0][i][0]), 'y': str(datum.poseKeypoints[0][i][1]), 'score': str(datum.poseKeypoints[0][i][2])})
        #存AVG
        tmp_AVGx.extend([tempX])
        tmp_AVGy.extend([float(datum.poseKeypoints[0][i][1])])
        #print(str(datum.poseKeypoints[0][i][0]),str(datum.poseKeypoints[0][i][1]),str(datum.poseKeypoints[0][i][2]))
    #輸出AVG值
    #print("AVGx :",tmp_AVGx,"  AVGy :",tmp_AVGy)
        
    df = pd.DataFrame(tmp_data)
    #print(df.values)
    data = np.array(df.values)
    data = data.reshape((1, 75))
    #print(data)
    #cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    img2 = datum.cvOutputData

    #AVG 計算
    path = "./data/csv/Biceps_curl/AVG"
    with open(path + "/AVG.csv", newline='') as csvfile:
            # 以冒號分隔欄位，讀取檔案內容
            rows = csv.reader(csvfile)
            # 讀取 CSV 檔內容，將每一列轉成一個 dictionary
            #rows = csv.DictReader(csvfile)
            #用panda 知道csv的列數
            df = pd.read_csv(path + "/AVG.csv")
            #print (len(df))
            column, row = 5, 25
            Ax = np.zeros((column,row))
            Ay = np.zeros((column,row))
            j = 1
            #將AVG 的x,y 值存入陣列中
            for row in rows:
                if row[0] == "x0" :
                    continue
                else:
                    o = 0
                    for i in range(0,25):
                        Ax[j][o] = row[2 * i]
                        #print(Ax[j][o])
                        Ay[j][o] = row[2 * i + 1]
                        #print(Ay[j][o])
                        o+=1
                    j = j + 1
            
            Score1 = np.array([np.linalg.norm(Ax[1] - tmp_AVGx),np.linalg.norm(Ay[1] - tmp_AVGy)])
            Score2 = np.array([np.linalg.norm(Ax[2] - tmp_AVGx),np.linalg.norm(Ay[2] - tmp_AVGy)])
            Score3 = np.array([np.linalg.norm(Ax[3] - tmp_AVGx),np.linalg.norm(Ay[3] - tmp_AVGy)])
            Score4 = np.array([np.linalg.norm(Ax[4] - tmp_AVGx),np.linalg.norm(Ay[4] - tmp_AVGy)])
            
            #flag紀錄等級
            LV = 0
            if(min(Score1[1],Score2[1],Score3[1],Score4[1])) == Score1[1]:
                LV = 1
            elif(min(Score1[1],Score2[1],Score3[1],Score4[1])) == Score2[1]:
                LV = 2
            elif(min(Score1[1],Score2[1],Score3[1],Score4[1])) == Score3[1]:
                LV = 3
            elif(min(Score1[1],Score2[1],Score3[1],Score4[1])) == Score4[1]:
                LV = 4
    #預測
    with tf.Graph().as_default():
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #每次都要load還需要再改#
        args = MyArgs()
        model = keras.models.load_model(args.model)
        #reshap[1,1,1,...~75個]
        model.predict(np.ones((1, 75)))
        #model.predict_classes(test)預測的是類別 ，model.predict(test) 預測的是數值
        rest = model.predict_classes(data,verbose=0)
        #print("=================",rest[0])
        if rest[0] == 0:
            #print('Bending------------')
            return ('Bending', img2 ,LV )
        elif rest[0] == 1:
            #print('Straight-------------')
            return ('Straight', img2 , LV)
    #print(df.iloc[:,:])
    return(3,img2)
    
cap = cv2.VideoCapture(args.video_path)
# ret為T/F 代表有沒有讀到圖片 frame 是一偵
#ret, frame = cap.read()
ret, img = cap.read()
# cv2.CAP_PROP_FPS 輸出影片FPS
video_fps = cap.get(cv2.CAP_PROP_FPS)

video_size = (img.shape[1], img.shape[0])
print('Videofps',video_fps)
print(video_size)

start_handle_time = time.time()
count = -1
#flag 判斷是否準備就緒
flag_ready_S = 0
flag_ready_B = 0
#計算時間點
Time = 0
while ret :
    ret, img = cap.read()
    #計算幀數
    count += 1
    
    #%6 => 1幀0.25秒
    if (count % 7) != 0:
        continue
    if ret == True:
        Time = Time+1
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        # 模糊可能還要測試 越高辨識率會變差，越低誤判率會變高 要找合適的中間值
        img = cv2.GaussianBlur(img, (7,7), 0)
        # 檢測
        (flag_action,target_out ,LV_out) = detection(img)
        
        #偵測伸直
        if flag_action == 'Straight' and flag_ready_S < 3:
            flag_ready_S = flag_ready_S + 1
        #連續偵測到兩次為伸直
        elif flag_action == 'Bending' and flag_ready_B < 2:
            flag_ready_B = flag_ready_B+1
        #正式辨識
        if Time >=3 and flag_ready_B >= 2 and flag_ready_S > 2:
            (flag_action,target_out,LV_out) = detection(img)
            print("等級為:",LV_out)

        print("now_action ",flag_action , "flag_S: ",flag_ready_S ,"flag_B: ",flag_ready_B ,"time :" , Time , "count :" ,count)
        #print(is_okay,"===============================================")
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", target_out)
        cv2.waitKey(0)
    else :
        break





#處理 Calling Model.predict in graph mode is not supported when the Model instance was constructed with eager mode enabled
#https://www.codeleading.com/article/42675321680/ 
