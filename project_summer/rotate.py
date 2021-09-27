import cv2
import os

path='./data/imgs/from_video/Crunch\\Underknee'
files=os.listdir(path)

p = 0

def Rotate(img):
    
    h = img.shape[0]
    w = img.shape[1]
    center = (h/2,w/2)
    #rotate 旋轉中心座標 , "90"為逆時針90度, "-90"為順時針90度, 1為縮放1倍
    R = cv2.getRotationMatrix2D(center,-90,1)
    rotate = cv2.warpAffine(img,R,(w,h))
    return rotate
    
for i in files :
    oldname=path+"/"+i
    img = cv2.imread(oldname)
    #cv2.imshow("show",rotate)
    #cv2.waitKey(100)
    #print (center)
    Result = Rotate(img)
    #儲存
    cv2.imwrite(path+"/"+str(p)+".jpg",Result)
    p = p+1