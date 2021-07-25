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


#def write_csv()
try:
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

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read frames on directory
    imagePaths = op.get_images_on_directory(args[0].image_dir);
    start = time.time()
    print("Body keypoints: \n")
    #enviroment
    csv_path = "./data/csv/Biceps_curl_Bending_Test.csv"
    with open(csv_path, 'w', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            # 寫入一列資料
            writer.writerow(['X', 'Y', 'Score'])
    # Process and display images
    for imagePath in imagePaths:
        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        print(str(datum.poseKeypoints))

        #write csv
        # 'a+' 是附加在後面 'a'是在前面
        with open(csv_path, 'a+', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            # 寫入另外幾列資料
            data = [] #儲存成一列(橫排)
            for i in range(0,24):
                data.extend([str(datum.poseKeypoints[0][i][0]),str(datum.poseKeypoints[0][i][1]),str(datum.poseKeypoints[0][i][2])])
                #print(data)
            writer.writerow(data)
            
            #writer.writerow(["","",""]) #nan

        if not args[0].no_display:
            cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
            key = cv2.waitKey(15)
            if key == 27: break

    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
    
    

    
            
    #cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    cv2.waitKey(0)


except Exception as e:
    print(e)
    sys.exit(0)
