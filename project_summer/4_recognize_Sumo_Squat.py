# 由定義的最好跟最差取距離
import matplotlib.pyplot as plt
import argparse
import time
import os
import math
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
from PIL import Image, ImageDraw, ImageFont
from os import walk


class MyArgs():
    def __init__(self):
        self.video_path = "./data/csv/Side_Lateral_RaiseL/endvideo/test/VID_20211221_193920.mp4"
        self.model = './data/model/model_Sumo_Squat'
        self.path = "./data/csv/Sumo_Squat/AVG/173"
        self.A = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
        self.testpath = './data/imgs/Sumo_Squat/testmodel/'
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

def detection(img ):
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
    if(str(datum.poseKeypoints)=='None'):
        return ("None" , img , 0 , 0 ,0)
    else:
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
                column, row = 6, len(args.A)
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
            
                #Score0 = np.array([np.linalg.norm(Ax[1] - tmp_AVGx),np.linalg.norm(Ay[1] - tmp_AVGy)])
                #Score1 = np.array([np.linalg.norm(Ax[2] - tmp_AVGx),np.linalg.norm(Ay[2] - tmp_AVGy)])
                #Score2 = np.array([np.linalg.norm(Ax[3] - tmp_AVGx),np.linalg.norm(Ay[3] - tmp_AVGy)])
                #Score3 = np.array([np.linalg.norm(Ax[4] - tmp_AVGx),np.linalg.norm(Ay[4] - tmp_AVGy)])
                #Score4 = np.array([np.linalg.norm(Ax[5] - tmp_AVGx),np.linalg.norm(Ay[5] - tmp_AVGy)])

                #Score0 表示與0的距離
                Score0 = math.pow(math.pow(np.linalg.norm(Ax[1] - tmp_AVGx),2)+math.pow(np.linalg.norm(Ay[1] - tmp_AVGy),2),0.5)
                Score1 = math.pow(math.pow(np.linalg.norm(Ax[2] - tmp_AVGx),2)+math.pow(np.linalg.norm(Ay[2] - tmp_AVGy),2),0.5)
                Score2 = math.pow(math.pow(np.linalg.norm(Ax[3] - tmp_AVGx),2)+math.pow(np.linalg.norm(Ay[3] - tmp_AVGy),2),0.5)
                Score3 = math.pow(math.pow(np.linalg.norm(Ax[4] - tmp_AVGx),2)+math.pow(np.linalg.norm(Ay[4] - tmp_AVGy),2),0.5)
                Score4 = math.pow(math.pow(np.linalg.norm(Ax[5] - tmp_AVGx),2)+math.pow(np.linalg.norm(Ay[5] - tmp_AVGy),2),0.5)
                # print("與等級0比較",Score0,"\n與等級1比較",Score1,"\n與等級2比較",Score2,"\n與等級3比較",Score3,"\n與等級4比較",Score4)
                # print(score01,score12,score23,score34)
                #flag紀錄等級
                LV = 0
                ScoreU = 0
                ScoreD = 0
                ScoreDU = 0 
                ScoreDD = 0
                if(min(Score0,Score1,Score2,Score3,Score4)) == Score0:
                    LV = 0
                elif(min(Score0,Score1,Score2,Score3,Score4)) == Score1:
                    #判斷出是等級1
                    LV = 1

                    #判斷是在 等級1的上還是下
                #在 等級1下面 => 與起始點0比較距離 往上時ScoreU = 0 ; ScoreD = 1
                    if(Score0 <= Score0_1):
                        
                        ScoreU = Score0
                        ScoreD = Score1
                        print("在LV" , LV ,"下面")
                        print("SCU , SCD : ",ScoreU ,ScoreD )
                        if (ScoreU<=Score01):
                            ScoreDU = 1
                        elif (ScoreU>Score01 and ScoreU <= 2*Score01):
                            ScoreDU = 2
                        elif (ScoreU>2*Score01 and ScoreU<= 3*Score01):
                            ScoreDU = 3
                        if (ScoreD<=Score01):
                            ScoreDD = 3
                        elif (ScoreD>Score01 and ScoreD <= 2*Score01):
                            ScoreDD = 2
                        elif (ScoreD>2*Score01 and ScoreD<= 3*Score01):
                            ScoreDD = 1
                    else:
                        #在 等級1上面 => 與起始點1比較距離 往上時ScoreU = 1 ; ScoreD = 2
                        ScoreU = Score1
                        ScoreD = Score2
                        print("在LV" , LV ,"上面")
                        print("SCU , SCD : ",ScoreU ,ScoreD )
                        if(ScoreU <= Score12):
                            ScoreDU = 3
                        elif (ScoreU>Score12 and ScoreU <= 2*Score12):
                            ScoreDU = 2
                        elif (ScoreU>2*Score12 and ScoreU<= 3*Score12):
                            ScoreDU = 1
                        if (ScoreD<=Score12):
                            ScoreDD = 3
                        elif (ScoreD>Score12 and ScoreD <= 2*Score12):
                            ScoreDD = 2
                        elif (ScoreD>2*Score12 and ScoreD<= 3*Score12):
                            ScoreDD = 1

                elif(min(Score0,Score1,Score2,Score3,Score4)) == Score2:
                    LV = 2
                    #判斷是在 等級2的上還是下
                    if(Score0 <= Score0_2):
                        #在 等級2下面 => 與起始點1比較距離 往上時ScoreU = 1 ; ScoreD = 2
                        ScoreU = Score1
                        ScoreD = Score2
                        print("在LV" , LV ,"下面")
                        print("SCU , SCD : ",ScoreU ,ScoreD )
                        if (ScoreU<=Score12):
                            ScoreDU = 1
                        elif (ScoreU>Score12 and ScoreU <= 2*Score12):
                            ScoreDU = 2
                        elif (ScoreU>2*Score12 and ScoreU<= 3*Score12):
                            ScoreDU = 3
                        if (ScoreD<=Score12):
                            ScoreDD = 3
                        elif (ScoreD>Score12 and ScoreD <= 2*Score12):
                            ScoreDD = 2
                        elif (ScoreD>2*Score12 and ScoreD<= 3*Score12):
                            ScoreDD = 1
                    else:
                        #在 等級2上面 => 與起始點3比較距離 往上時ScoreU = 2 ; ScoreD = 3
                        ScoreU = Score2
                        ScoreD = Score3
                        print("在LV" , LV ,"上面")
                        print("SCU , SCD : ",ScoreU ,ScoreD )
                        if(ScoreU <= Score23):
                            ScoreDU = 3
                        elif (ScoreU>Score23 and ScoreU <= 2*Score23):
                            ScoreDU = 2
                        elif (ScoreU>2*Score23 and ScoreU<= 3*Score23):
                            ScoreDU = 1
                        if (ScoreD<=Score23):
                            ScoreDD = 3
                        elif (ScoreD>Score23 and ScoreD <= 2*Score23):
                            ScoreDD = 2
                        elif (ScoreD>2*Score23 and ScoreD<= 3*Score23):
                            ScoreDD = 1
               
                elif(min(Score0,Score1,Score2,Score3,Score4)) == Score3:
                    LV = 3
                
                    if(Score0 <= Score0_3):
                        ScoreU = Score2
                        ScoreD = Score3
                        print("在LV" , LV ,"下面")
                        print("SCU , SCD : ",ScoreU ,ScoreD )
                        if (ScoreU<=Score23):
                            ScoreDU = 1
                        elif (ScoreU>Score23 and ScoreU <= 2*Score23):
                            ScoreDU = 2
                        elif (ScoreU>2*Score23 and ScoreU<= 3*Score23):
                            ScoreDU = 3
                        if (ScoreD<=Score23):
                            ScoreDD = 3
                        elif (ScoreD>Score23 and ScoreD <= 2*Score23):
                            ScoreDD = 2
                        elif (ScoreD>2*Score23 and ScoreD<= 3*Score23):
                            ScoreDD = 1
                    else:
                        ScoreU = Score3
                        ScoreD = Score4
                        print("在LV" , LV ,"上面")
                        print("SCU , SCD : ",ScoreU ,ScoreD )
                        if(ScoreU <= Score34):
                            ScoreDU = 3
                        elif (ScoreU>Score34 and ScoreU <= 2*Score34):
                            ScoreDU = 2
                        elif (ScoreU>2*Score34 and ScoreU<= 3*Score34):
                            ScoreDU = 1
                        if (ScoreD<=Score34):
                            ScoreDD = 3
                        elif (ScoreD>Score34 and ScoreD <= 2*Score34):
                            ScoreDD = 2
                        elif (ScoreD>2*Score34 and ScoreD<= 3*Score34):
                            ScoreDD = 1
               
                elif(min(Score0,Score1,Score2,Score3,Score4)) == Score4:
                    LV = 4
                    if(Score0 <= Score0_4):
                        ScoreU = Score3
                        ScoreD = Score4
                        print("在LV" , LV ,"下面")
                        print("SCU , SCD : ",ScoreU ,ScoreD )
                        if (ScoreU<=Score34):
                            ScoreDU = 1
                        elif (ScoreU>Score34 and ScoreU <= 2*Score34):
                            ScoreDU = 2
                        elif (ScoreU>2*Score34 and ScoreU<= 3*Score34):
                            ScoreDU = 3
                        if (ScoreD<=Score34):
                            ScoreDD = 3
                        elif (ScoreD>Score34 and ScoreD <= 2*Score34):
                            ScoreDD = 2
                        elif (ScoreD>2*Score34 and ScoreD<= 3*Score34):
                            ScoreDD = 1
                    else:
                        print("在LV" , LV ,"上面")
                        ScoreU =  ScoreD = Score4
                        if(ScoreU <= 2*Score34 or ScoreD <=2*Score34):
                            ScoreDU = 2
                            ScoreDD = 3
                    
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
                return ('Ready', img2 , LV , ScoreDU , ScoreDD)
            elif rest[0] == 1:
                return ('Start', img2 , LV , ScoreDU , ScoreDD)

#Cv2輸出中文
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判斷是否OpenCV圖片類型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
#AVG 計算
with open(args.path + "/AVG.csv", newline='') as csvfile:
        # 以冒號分隔欄位，讀取檔案內容
        rows = csv.reader(csvfile)
        # 讀取 CSV 檔內容，將每一列轉成一個 dictionary
        #rows = csv.DictReader(csvfile)
        #用panda 知道csv的列數
        df = pd.read_csv(args.path + "/AVG.csv")
        #print (len(df))
        column, row = 6, len(args.A)
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
        #Score01 = np.array([(np.linalg.norm(Ax[1] - Ax[2])),(np.linalg.norm(Ay[1] - Ay[2]))])
        #Score12 = np.array([(np.linalg.norm(Ax[2] - Ax[3])),(np.linalg.norm(Ay[2] - Ay[3]))])
        #Score23 = np.array([(np.linalg.norm(Ax[3] - Ax[4])),(np.linalg.norm(Ay[3] - Ay[4]))])
        #Score34 = np.array([(np.linalg.norm(Ax[4] - Ax[5])),(np.linalg.norm(Ay[4] - Ay[5]))])

        Score01 = (math.pow(math.pow(np.linalg.norm(Ax[1] - Ax[2]),2)+math.pow(np.linalg.norm(Ay[1] - Ay[2]),2),0.5))/3
        Score12 = (math.pow(math.pow(np.linalg.norm(Ax[2] - Ax[3]),2)+math.pow(np.linalg.norm(Ay[2] - Ay[3]),2),0.5))/3
        Score23 = (math.pow(math.pow(np.linalg.norm(Ax[3] - Ax[4]),2)+math.pow(np.linalg.norm(Ay[3] - Ay[4]),2),0.5))/3
        Score34 = (math.pow(math.pow(np.linalg.norm(Ax[4] - Ax[5]),2)+math.pow(np.linalg.norm(Ay[4] - Ay[5]),2),0.5))/3
        Score0_1 = Score01
        Score0_2 = math.pow(math.pow(np.linalg.norm(Ax[1] - Ax[3]),2)+math.pow(np.linalg.norm(Ay[1] - Ay[3]),2),0.5)
        Score0_3 = math.pow(math.pow(np.linalg.norm(Ax[1] - Ax[4]),2)+math.pow(np.linalg.norm(Ay[1] - Ay[4]),2),0.5)
        Score0_4 = math.pow(math.pow(np.linalg.norm(Ax[1] - Ax[5]),2)+math.pow(np.linalg.norm(Ay[1] - Ay[5]),2),0.5)
        
        #print("等級0~1距離 :" , Score01)
        #print("等級1~2距離 :" , Score12 , "等級0~2距離 : ",Score0_2)
        #print("等級2~3距離 :" , Score23 , "等級0~3距離 : ",Score0_3)
        #print("等級3~4距離 :" , Score34 , "等級0~4距離 : ",Score0_4)
cap = cv2.VideoCapture(args.video_path)
# ret為T/F 代表有沒有讀到圖片 frame 是一偵
#ret, frame = cap.read()
ret, img = cap.read()
# cv2.CAP_PROP_FPS 輸出影片FPS
video_fps = cap.get(cv2.CAP_PROP_FPS)
#print(img)
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
#作影片用
EndTime = 1 
tempA = []
tempSCDU = []
tempSCDD = []
#Out_put = { '等級一': 0 , '等級二': 0 , '等級三': 0 ,'等級四': 0 }

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
        (Flag_action,target_out ,LV_out , SCDUout ,SCDDout ) = detection(img )

        #==============偵測是否開始運動================未完成===========
        if (Flag_action == 'None'):
            continue
        elif(Flag_action == 'Ready'):
            cv2.rectangle(target_out, (280, 10), (470, 60), (255, 255, 255), -1)
            cv2.putText(target_out, 'Stationary', (285, 45), 4, 1, (255, 0, 0), 1, 35)
            #cv2.imwrite('./img/Side_Lateral_RaiseL/'+str(Action_time)+"-"+str(Time)+"-"+str(LV_out)+'.jpg', target_out)
            if(Action_flag == 1):
                Action_flag = 0
                Time += 1
                #分數
                Grade = 0
                for i in range(0,len(tempA)) :
                    if tempSCDU[i]>tempSCDD[i]:
                        Grade += tempSCDU[i]
                    else :
                        Grade += tempSCDD[i]
                cv2.rectangle(target_out, (280, 10), (470, 90), (255, 255, 255), -1)
                cv2.putText(target_out, 'Stationary', (285, 45), 4, 1, (255, 0, 0), 1, 35)
                target_out=cv2ImgAddText(target_out, "本次動作評分為 "+str(round(Grade/len(tempA), 2)), 290, 50, (255, 0, 0), 20)
                print("評分為",Grade/len(tempA))
                #儲存圖片
                #cv2.imwrite('./img/Side_Lateral_RaiseL/'+str(Action_time)+"-"+str(Time)+"-"+str(LV_out)+'.jpg', target_out)
                #計算時間點,emdtime
                Endtime =0
                Time = 0
                tempA = []
                tempSCDU = []
                tempSCDD = []
                Action_time += 1
        else:
            cv2.rectangle(target_out, (350, 10), (470, 90), (255, 255, 255), -1)
            target_out=cv2ImgAddText(target_out, "階段 "+str(LV_out), 360, 15, (255, 0, 0), 35)
            target_out=cv2ImgAddText(target_out, "分數 "+str(max(SCDUout,SCDDout)), 360, 50, (255, 0, 0), 35)
            Time += 1
            print ("SCDU : ",SCDUout )
            print ("SCDD : ",SCDDout )
            tempA.extend([LV_out])
            tempSCDU.extend([SCDUout])
            tempSCDD.extend([SCDDout])
            #儲存圖片   
            #cv2.imwrite('./img/Side_Lateral_RaiseL/'+str(Action_time)+"-"+str(Time)+"-"+str(LV_out)+'.jpg', target_out)
            if(Time>=7):
                Action_flag = 1
        
        print("等級為 : ",LV_out)
        print("狀態 : " , Flag_action)
        print("Time : " , Time , "Action_time : " ,Action_time)
        print("TempA : ",tempA ,"\nTempSCDU : " ,tempSCDU , "\nTempSCDD : " ,tempSCDD)
        #print(is_okay,"===============================================")
        cv2.imwrite(args.testpath+str(int(count/6))+'.jpg', target_out)    
        #target_out=(cv2.cvtColor(target_out, cv2.COLOR_BGR2RGB))
        #cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", target_out)
        #cv2.waitKey(0)
    else :
        break

#處理 Calling Model.predict in graph mode is not supported when the Model instance was constructed with eager mode enabled
#https://www.codeleading.com/article/42675321680/ 




