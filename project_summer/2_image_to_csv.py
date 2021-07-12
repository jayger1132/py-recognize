# From Python
# It requires OpenCV installed for Python
import csv
import sys
import cv2
import os
from sys import platform
import argparse

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
    parser.add_argument("--image_path", default="./data/imgs/from_video/Biceps_curl/img_0-99/Biceps_curl-1.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

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

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # Display Image
    print("Body keypoints: \n" + str(datum.poseKeypoints))
    for i in range(0,24):
        print(str(datum.poseKeypoints[0][i]),"\n")

    #write csv
    with open('./data/csv/Biceps_curl.csv', 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        # 寫入一列資料
        writer.writerow(['X', 'Y', 'Score'])
        # 寫入另外幾列資料
        for i in range(0,24):
            writer.writerow([str(datum.poseKeypoints[0][i][0]),str(datum.poseKeypoints[0][i][1]),str(datum.poseKeypoints[0][i][2])])
            
    #cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    cv2.waitKey(0)


except Exception as e:
    print(e)
    sys.exit(0)
