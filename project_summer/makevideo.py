import numpy as np
import cv2
from os import walk
#讀取一張圖片
dim = (480,720)
path = "./temp/normal/"
#完成寫入物件的建立，第一個引數是合成之後的影片的名稱，第二個引數是可以使用的編碼器，第三個引數是幀率即每秒鐘展示多少張圖片，第四個引數是圖片大小資訊
videowrite = cv2.VideoWriter(path+'V2.mp4',-1,2,dim)#20是幀數，size是圖片尺寸
img_array=[]

for root, dirs ,files in walk(path):
    len=(len(files))
    #print("路徑：", root)
    #print("  目錄：", dirs)
    #print("  檔案：", files)
        
for filename in [(path+"V2"+'{0}.jpg').format(i) for i in range(len-1)]:
    img = cv2.imread(filename)
    if img is None:
        print(filename + " is error!")
        continue
    img_array.append(img)
#print(img_array)
for i in img_array :
    videowrite.write(i)


