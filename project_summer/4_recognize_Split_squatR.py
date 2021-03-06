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
        self.video_path = './data/video/Split_squatR/complete/VID_20211123_162504.mp4'
        self.model = './data/model/model_Biceps_curl'
        self.path = "./data/csv/Split_squatR/AVG"
        self.A = [0,1,2,5,8,9,10,12,13,14]
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

    # ????????????????????????,?????????????????????????????????
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
                #datum.posekeypoint ???const ???????????????????????????
                #datum.poseKeypoints[0][i][0]=X+offsetneck
            else:
                None
        else:
            tempX = (img.shape[1]) / 2

        tmp_data.append({ 'x': str(datum.poseKeypoints[0][i][0]), 'y': str(datum.poseKeypoints[0][i][1]), 'score': str(datum.poseKeypoints[0][i][2])})
        
    #???AVG
    for i in args.A:
        if i != 1:
            if float(datum.poseKeypoints[0][i][0]) != 0.0:
                #print(datum.poseKeypoints[0][i][0])
                tempX = float(datum.poseKeypoints[0][i][0]) - offsetneck
                #print(tempX)
                #datum.posekeypoint ???const ???????????????????????????
                #datum.poseKeypoints[0][i][0]=X+offsetneck
            else:
                None
        else:
            tempX = (img.shape[1]) / 2

        tmp_AVGx.extend([tempX])
        tmp_AVGy.extend([float(datum.poseKeypoints[0][i][1])])
        #print(str(datum.poseKeypoints[0][i][0]),str(datum.poseKeypoints[0][i][1]),str(datum.poseKeypoints[0][i][2]))
    #??????AVG???
    #print("AVGx :",tmp_AVGx,"  AVGy :",tmp_AVGy)
        
    df = pd.DataFrame(tmp_data)
    #print(df.values)
    data = np.array(df.values)
    data = data.reshape((1, 75))
    #print(data)
    #cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    img2 = datum.cvOutputData

    #AVG ??????
     
    with open(args.path + "/AVG.csv", newline='') as csvfile:
            # ??????????????????????????????????????????
            rows = csv.reader(csvfile)
            # ?????? CSV ???????????????????????????????????? dictionary
            #rows = csv.DictReader(csvfile)
            #???panda ??????csv?????????
            df = pd.read_csv(args.path + "/AVG.csv")
            #print (len(df))
            column, row = 5, 10
            Ax = np.zeros((column,row))
            Ay = np.zeros((column,row))
            j = 1
            #???AVG ???x,y ??????????????????
            for row in rows:
                if row[0] == "x0" :
                    continue
                else:
                    o = 0
                    for i in args.A:
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
            #flag????????????
            LV = 0
            if(min(Score1[1],Score2[1],Score3[1],Score4[1])) == Score1[1]:
                LV = 1
            elif(min(Score1[1],Score2[1],Score3[1],Score4[1])) == Score2[1]:
                LV = 2
            elif(min(Score1[1],Score2[1],Score3[1],Score4[1])) == Score3[1]:
                LV = 3
            elif(min(Score1[1],Score2[1],Score3[1],Score4[1])) == Score4[1]:
                LV = 4
    #??????
    #with tf.Graph().as_default():
    #    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #    #????????????load???????????????#
    #    #args = MyArgs()
    #    model = keras.models.load_model(args.model)
    #    #reshap[1,1,1,...~75???]
    #    model.predict(np.ones((1, 75)))
    #    #model.predict_classes(test)?????????????????? ???model.predict(test) ??????????????????
    #    rest = model.predict_classes(data,verbose=0)
    #    #print("=================",rest[0])
    #    if rest[0] == 0:
    #        #print('Bending------------')
    #        return ('Bending', img2 ,LV )
    #    elif rest[0] == 1:
    #        #print('Straight-------------')
    #        return ('Straight', img2 , LV)
    ##print(df.iloc[:,:])
    return (img2 ,LV )
    
cap = cv2.VideoCapture(args.video_path)
# ret???T/F ??????????????????????????? frame ?????????
#ret, frame = cap.read()
ret, img = cap.read()
# cv2.CAP_PROP_FPS ????????????FPS
video_fps = cap.get(cv2.CAP_PROP_FPS)

video_size = (img.shape[1], img.shape[0])
#print('Videofps',video_fps)
#print(video_size)

start_handle_time = time.time()
count = -1


#???????????????
Time = 0
Action_time = 0
Out_put = { '?????????': 0 , '?????????': 0 , '?????????': 0 ,'?????????': 0 }
while ret :
    ret, img = cap.read()
    #????????????
    count += 1
    
    #%6 => 1???0.20???
    if (count % 6) != 0:
        continue
    if ret == True:
        Time = Time+1
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        # ???????????????????????? ??????????????????????????????????????????????????? ????????????????????????
        img = cv2.GaussianBlur(img, (7,7), 0)
        # ??????
        (target_out ,LV_out) = detection(img)
        print("LV" , LV_out)
        #==============????????????????????????================?????????===========
        if( Time%15 != 0):
            if(Time>=6 and Time <=12 ):
                if ( LV_out == 1):
                    Out_put['?????????']+=1
                elif ( LV_out == 2):
                    Out_put['?????????']+=1
                elif ( LV_out == 3):
                    Out_put['?????????']+=1
                elif ( LV_out == 4):
                    Out_put['?????????']+=1
        else:
            Time=0
            Action_time+=1
        #print(is_okay,"===============================================")
        print(Out_put.items(),Action_time)
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", target_out)
        cv2.waitKey(0)
    else :
        break
print(Out_put.items())
if(max(Out_put['?????????'],Out_put['?????????'],Out_put['?????????'],Out_put['?????????']) == Out_put['?????????'] ) :
    LV = 1
elif (max(Out_put['?????????'],Out_put['?????????'],Out_put['?????????'],Out_put['?????????']) == Out_put['?????????'] ):
    LV = 2
elif (max(Out_put['?????????'],Out_put['?????????'],Out_put['?????????'],Out_put['?????????']) == Out_put['?????????'] ):
    LV = 3
elif (max(Out_put['?????????'],Out_put['?????????'],Out_put['?????????'],Out_put['?????????']) == Out_put['?????????'] ):
    LV = 4
print('??????'+str(LV))
#dbhost='justtry.406.csie.nuu.edu.tw'
#dbuser='root'
#dbport=33060
#dbpass='nuuCSIE406'
#dbname='account'
#try:
#    db=pymysql.connect(host=dbhost,user=dbuser,port=dbport,password=dbpass,database=dbname)
#    print("????????????")
#    cursor = db.cursor()
#except pymysql.Error as e:
#    print("????????????"+str(e))
#sql = "SELECT account , password , date FROM account WHERE account = '%s'" % (account)
#try:
#    cursor.execute(sql)
#    results = cursor.fetchone()
#except:
#    db.rollback()




#?????? Calling Model.predict in graph mode is not supported when the Model instance was constructed with eager mode enabled
#https://www.codeleading.com/article/42675321680/ 

