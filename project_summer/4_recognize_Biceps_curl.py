#做法較粗糙 判斷狀態
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
        self.video_path = './data/video/Biceps_curl/complete/VID_20211130_201924.mp4'
        self.model = './data/model/model_Biceps_curl'
        self.path = "./data/csv/Biceps_curl/AVG"
        self.A = [0,1,2,3,4,5,6,7]
args = MyArgs()
dim = (480, 720)

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
    video_size = (img.shape[1], img.shape[0])
    #print(video_size)
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
    #print(tmp_data)
    #存AVG
    for i in args.A:
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

        tmp_AVGx.extend([tempX])
        tmp_AVGy.extend([float(datum.poseKeypoints[0][i][1])])
        
    #輸出AVG值
    #print("AVGx :",tmp_AVGx,"\nAVGy :",tmp_AVGy)
    df = pd.DataFrame(tmp_data)
    #print(df.values)
    data = np.array(df.values)
    data = data.reshape((1, 75))
    #print(data)
    #cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    img2 = datum.cvOutputData

    #AVG 計算
     
    with open(args.path + "/AVG.csv", newline='') as csvfile:
            # 以冒號分隔欄位，讀取檔案內容
            rows = csv.reader(csvfile)
            # 讀取 CSV 檔內容，將每一列轉成一個 dictionary
            #rows = csv.DictReader(csvfile)
            #用panda 知道csv的列數
            df = pd.read_csv(args.path + "/AVG.csv")
            #print (len(df))
            column, row = 6, 8
            Ax = np.zeros((column,row))
            Ay = np.zeros((column,row))
            j = 1
            #將AVG 的x,y 值存入陣列中
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
            Score0 = np.array([np.linalg.norm(Ax[1] - tmp_AVGx),np.linalg.norm(Ay[1] - tmp_AVGy)])
            Score1 = np.array([np.linalg.norm(Ax[2] - tmp_AVGx),np.linalg.norm(Ay[2] - tmp_AVGy)])
            Score2 = np.array([np.linalg.norm(Ax[3] - tmp_AVGx),np.linalg.norm(Ay[3] - tmp_AVGy)])
            Score3 = np.array([np.linalg.norm(Ax[4] - tmp_AVGx),np.linalg.norm(Ay[4] - tmp_AVGy)])
            Score4 = np.array([np.linalg.norm(Ax[5] - tmp_AVGx),np.linalg.norm(Ay[5] - tmp_AVGy)])
            print(Score0,Score1,Score2,Score3,Score4)
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
        #args = MyArgs()
        model = keras.models.load_model(args.model)
        #reshap[1,1,1,...~75個]
        model.predict(np.ones((1, 75)))
        #model.predict_classes(test)預測的是類別 ，model.predict(test) 預測的是數值
        rest = model.predict_classes(data,verbose=0)
        #print("=================",rest[0])
        if rest[0] == 0:
            #print('Bending------------')
            return ('Ready', img2 ,LV )
        elif rest[0] == 1:
            #print('Straight-------------')
            return ('Start', img2 , LV)
    #print(df.iloc[:,:])
    #return (img2 ,LV)
    
cap = cv2.VideoCapture(args.video_path)
# ret為T/F 代表有沒有讀到圖片 frame 是一偵
#ret, frame = cap.read()
ret, img = cap.read()
# cv2.CAP_PROP_FPS 輸出影片FPS
video_fps = cap.get(cv2.CAP_PROP_FPS)
print(img)
video_size = (img.shape[1], img.shape[0])
#print('Videofps',video_fps)
#print(video_size)

start_handle_time = time.time()
count = -1
#計算張數
Time = 0
#計算動作次數
Action_flag = 0
Action_time = 0
tempA = []
Out_put = { '等級一': 0 , '等級二': 0 , '等級三': 0 ,'等級四': 0 }
while ret :
    ret, img = cap.read()
    #計算幀數
    count += 1
    
    #%6 => FPS 5
    if (count % 6) != 0:
        continue
    if ret == True:
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        # 模糊可能還要測試 越高辨識率會變差，越低誤判率會變高 要找合適的中間值
        img = cv2.GaussianBlur(img, (7,7), 0)
        # 檢測
        (Flag_action,target_out ,LV_out) = detection(img)

        #==============偵測是否開始運動================未完成===========
        if(Flag_action == 'Ready'):
            
            if(Action_flag == 1):
                Action_time += 1
                Action_flag = 0
                #分數
                Grade = 0
                for i in range(0,len(tempA)) :
                    if i<=len(tempA)/4:
                        if(tempA[i]==1 or tempA[i]==2):
                            Grade+=1
                    elif i<=3*len(tempA)/4:
                        if(tempA[i]==3 or tempA[i]==4):
                            Grade+=1
                    else:
                        if(tempA[i]==1 or tempA[i]==2):
                            Grade+=1
                print("評分為",Grade/len(tempA)*100,"%")
                #計算時間點
                Time = 0
                tempA = []
        else:
            Time += 1
            tempA.extend([LV_out])
            if(Time>=7 and Time <=12 ):
                if ( LV_out == 1):
                    Out_put['等級一']+=1
                elif ( LV_out == 2):
                    Out_put['等級二']+=1
                elif ( LV_out == 3):
                    Out_put['等級三']+=1
                elif ( LV_out == 4):
                    Out_put['等級四']+=1
                Action_flag = 1
        
        print("等級為 : ",LV_out)
        print("各個等級 : ",Out_put)
        print("狀態 : " , Flag_action)
        print("Time : " , Time , "Action_time : " ,Action_time)
        print(tempA)
        #print(is_okay,"===============================================")
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", target_out)
        cv2.waitKey(100)
    else :
        break

#print(Out_put.items())
#if(max(Out_put['等級一'],Out_put['等級二'],Out_put['等級三'],Out_put['等級四']) == Out_put['等級一'] ) :
#    LV = 1
#elif (max(Out_put['等級一'],Out_put['等級二'],Out_put['等級三'],Out_put['等級四']) == Out_put['等級二'] ):
#    LV = 2
#elif (max(Out_put['等級一'],Out_put['等級二'],Out_put['等級三'],Out_put['等級四']) == Out_put['等級三'] ):
#    LV = 3
#elif (max(Out_put['等級一'],Out_put['等級二'],Out_put['等級三'],Out_put['等級四']) == Out_put['等級四'] ):
#    LV = 4
#print('等級'+str(LV))


#上傳資料
#dbhost='justtry.406.csie.nuu.edu.tw'
#dbuser='root'
#dbport=33060
#dbpass='nuuCSIE406'
#dbname='gordon'
#try:
#    db = pymysql.connect(host=dbhost,user=dbuser,port=dbport,password=dbpass,database=dbname)
#    print("連結成功")
#    cursor = db.cursor()
#except pymysql.Error as e:
#    print("連線失敗"+str(e))
##sql = "SELECT * FROM Identify "
#sql = "INSERT INTO Identify (exercise , grade , suggest ) VALUES ('二頭彎舉', %d ,'普通' ) " % (temp)
#try:
#    cursor.execute(sql)
#    db.commit()
#    print("上傳成功")
#except:
#    db.rollback()




#處理 Calling Model.predict in graph mode is not supported when the Model instance was constructed with eager mode enabled
#https://www.codeleading.com/article/42675321680/ 
