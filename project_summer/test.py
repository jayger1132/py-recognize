import numpy as np
import csv
import pandas as pd
import numpy as np
from os import walk

path = "./data/video/Split_squatL/L1"
paths=[]
imgpath = "./data/imgs/from_video/Split_squatL/"
dir = "AVG1"
name = "/split_squat"
def convertVideoToImage(video_paths, save_title, save_root=None , img_path =None , crop_size = None):
    start_time = time.time()
    
    if save_root is None:
        save_root = save_title
    save_path = img_path + save_root
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    fps_fix = 3 #fps15，fps_fix取3，一秒5張
    dim = (720, 720)
    # jpeg_flag = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    diff_threshold = 0
    
    subDir = ''
    count = -1
    img_count = 0
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