# From Python
# It requires OpenCV installed for Python
import csv
import sys
import cv2
import os
import time
from sys import platform
import argparse
import pandas

import cv2
import numpy as np
import os
import sys
import os.path
import time
import math
from os import walk

# config
path = "./data/csv/Side_Lateral_RaiseL/endvideo/Side_Lateral_RaiseL/"
paths = []
#paths=["./data/csv/Side_Lateral_RaiseL/endvideo/Biceps_curl/testb.mp4" ]
imgpath = "./data/csv/Side_Lateral_RaiseL/"
dir = "endimg/"
name = "D"


for root, dirs ,files in walk(path):
    
   #print("路徑：", root)
   #print("  目錄：", dirs)
   #print("  檔案：", files)
   for file in files:
        
       paths.extend([path+'/'+str(file)])



def toDHash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h = []
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                h.append(1)
            else:
                h.append(0)
    return h

# hamming distance
def hDist(hash1, hash2):
    num = 0
    for idx in range(len(hash1)):
        if hash1[idx] != hash2[idx]:
            num += 1
    return num

def dHashDiff(i1, i2):
    h1 = toDHash(i1)
    h2 = toDHash(i2)
    return hDist(h1, h2)

def convertVideoToImage(video_paths, save_title, save_root=None , img_path =None , crop_size = None):
    start_time = time.time()
    
    if save_root is None:
        save_root = save_title
    save_path = img_path + save_root
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    fps_fix = 3 #fps15，fps_fix取3，一秒5張
    dim = (480, 720)
    # jpeg_flag = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    diff_threshold = 0
    
    subDir = ''
    count = -1
    img_count = 1
    #subDir_flag = True
    
    for video_path in video_paths:
        
        cam = cv2.VideoCapture(video_path)
        start_flag = False
        ret_val = True
        pre_frame = None
        while ret_val:
            ret_val, image = cam.read()
            count = count + 1
            if count % fps_fix != 0:
                continue

            #if img_count % 100 == 0 and subDir_flag:
            #    subDir = '/img_' + str(img_count) + '-' + str(img_count + 99)
            #    if not os.path.exists(save_path + subDir):
            #        os.mkdir(save_path + subDir)
            #        subDir_flag = False

            if not start_flag:
                if image is None:
                    print("image \"%s\" is not exist" % video_path)
                    break
                print('video image: (%d, %d)' % (image.shape[1], image.shape[0]))
                start_flag = True

            if image is not None:
                # 輸出放入的影片
                cv2.imshow('test',image)
                image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                # image = cv2.GaussianBlur(image, (31, 31), 0)
                # crop
                if crop_size is not None:                
                    (x1, y1, x2, y2) = crop_size
                    if x2 is None:
                        x2 = image.shape[1]
                    if y2 is None:
                        y2 = image.shape[0]
                    # crop
                    image = image[y1:y2, x1:x2]
                if pre_frame is None or dHashDiff(image, pre_frame) > diff_threshold:
                    # _path = (save_path + subDir + '/' + save_title +
                    # '-%d.jpg') % img_count
                    # cv2.imwrite(_path, image, jpeg_flag)
                    _path = (save_path + subDir + save_title + '-%d.jpg') % img_count
                    print('save image at: ' + _path)
                    cv2.imwrite(_path, image)
                    img_count = img_count + 1
                    #subDir_flag = True
                    pre_frame = image
            else:
                print('frame is empty...')

            if cv2.waitKey(1) == 27:
                break
    print('Finish in %.4f secs.' % (time.time() - start_time))
    print('Number of images: %d/%d' % (img_count, math.floor(count / fps_fix) + 1))
    cv2.destroyAllWindows()

convertVideoToImage(paths, name , dir , imgpath)
#--------------------------------------------------------------------------#
#enviroment
parser = argparse.ArgumentParser()
testpath = './data/csv/Side_Lateral_RaiseL/endtest/'
csv_path = "./data/csv/Side_Lateral_RaiseL/endtest.csv"
#圖片位置
parser.add_argument("--image_dir", default="./data/csv/Side_Lateral_RaiseL/endimg/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
#def write_csv()
count = 1
try:
    # Import Openpose (Windows/Ubuntu/OSX)
    #dir_path = os.path.dirname(os.path.realpath(__file__))
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
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--image_dir",
    #default="./data/imgs/from_video/Biceps_curl/straight_test/", help="Process
    #a directory of images.  Read all standard formats (jpg, png, bmp, etc.).")
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

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read frames on directory
    imagePaths = op.get_images_on_directory(args[0].image_dir)
    start = time.time()
    print("Body keypoints: \n")
    
    with open(csv_path, 'w', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            tmp_data = []
            for i in range(0,25):
                #extend 是[0],[1],[2]這樣+ append是[012],[012],[012]的+
                tmp_data.extend(['x' + str(i),'y' + str(i)])
            #print(tmp_data)
            # 寫入一列資料
            writer.writerow(tmp_data)
    # Process and display images
    for imagePath in imagePaths:
        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        #print(str(datum.poseKeypoints))
        #要先判斷圖片是否判斷得出來 會有None的問題
        if datum.poseKeypoints is None:
            print(str(imagePath),"Noneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
            try:
                os.remove(imagePath)
                continue
            except OSError as e:
                print(e)
        else :
            print(imagePath)
        #video_size = (imageToProcess.shape[1],imageToProcess.shape[0])
        #偏移量 normorlize
        print("neck: ",datum.poseKeypoints[0][1][0]," imgsize: ",imageToProcess.shape)
        offsetneck = datum.poseKeypoints[0][1][0] - (imageToProcess.shape[1] / 2) 
        print("offsetneck: ",offsetneck)
        
        
        #write csv
        # 'a+' 是附加在後面 'a'是在前面
        with open(csv_path, 'a+', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            # 寫入另外幾列資料
            data = [] #儲存成一列(橫排)
            temp = (0,0,0)
            # range(0,25) => 0~24
            for i in range(0,25):
                # 不去偏移neck
                if i != 1 :
                    if float(str(datum.poseKeypoints[0][i][0])) != 0.0:
                        print(datum.poseKeypoints[0][i][0])
                        temp = [datum.poseKeypoints[0][i][0] - offsetneck,str(datum.poseKeypoints[0][i][1])]
                        print(temp)
                    # 如果x值為0代表沒有抓到點
                    else :
                        temp = [str(datum.poseKeypoints[0][i][0]),str(datum.poseKeypoints[0][i][1])]
                #neck
                else :
                    temp = [imageToProcess.shape[1] / 2,str(datum.poseKeypoints[0][i][1])]
                
                data.extend(temp)
                #print(data)
            
            writer.writerow(data)
            
        #    #writer.writerow(["","",""]) #nan

        if not args[0].no_display:
            count+=1
            cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
            cv2.imwrite(testpath+str(count)+'.jpg', datum.cvOutputData)
            cv2.waitKey(100)

    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")

    #cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    cv2.waitKey(0)


except Exception as e:
    print(e)
    sys.exit(0)

