#Python 3.9.6
from watchdog.observers import Observer
from watchdog.events import *
from os import walk
from ftplib import FTP
import numpy as np
import cv2
import os
import csv
import shutil
import time
unrecognize_path = './unrecognize/'
action = None
recognize_path = None
class FileEventHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        pass
    def on_moved(self, event):
        if event.is_directory:
            print("directory moved from {0} to {1}".format(event.src_path,event.dest_path))
        else:
            print("file moved from {0} to {1}".format(event.src_path,event.dest_path))

    def on_created(self, event):
        if event.is_directory:
            print("directory created:{0}".format(event.src_path))
        else:
            for root, dirs ,files in walk(unrecognize_path):
                #print("路徑：", root)
                #print("  目錄：", dirs)
                #print("  檔案：", files)

                ######辨識code#######
                for file in files:
                    print("now")
                    #print(root + '/' + str(file))
                    if ("Side_Lateral_RaiseL" in str(file)):
                        
                        action = 'Side_Lateral_RaiseL'
                        recognize_path = "./img/"+action+"/"
                        #清空上一個辨識的檔案 刪除檔案用 os.path.is~~~
                        if os.path.isdir("./img/"+action):
                            shutil.rmtree("./img/"+action)
                        print("~~")
                        os.system("C:/Users/sos22/AppData/Local/Programs/Python/Python39/python recognize_Side_Lateral_RaiseL.py")
                    
                    elif ("Side_Lateral_RaiseR" in str(file)):
                        action = "Side_Lateral_RaiseR"
                        recognize_path = "./img/"+action+"/"
                        if os.path.isdir("./img/"+action):
                            shutil.rmtree("./img/"+action)
                        print("~~")
                        os.system("C:/Users/sos22/AppData/Local/Programs/Python/Python39/python recognize_Side_Lateral_RaiseR.py")

                    elif ("Biceps_Curl" in str(file)):
                        action = "Biceps_Curl"
                        recognize_path = "./img/"+action+"/"
                        if os.path.isdir("./img/"+action):
                            shutil.rmtree("./img/"+action)
                        print("~~")
                        os.system("C:/Users/sos22/AppData/Local/Programs/Python/Python39/python recognize_Biceps_Curl.py")
                    elif ("Sumo_Squat" in str(file)):
                        action = "Sumo_Squat"
                        recognize_path = "./img/"+action+"/"
                        if os.path.isdir("./img/"+action):
                            shutil.rmtree("./img/"+action)
                        print("~~")
                        os.system("C:/Users/sos22/AppData/Local/Programs/Python/Python39/python recognize_Sumo_Squat.py")
            if action != None:
                #print("~~")
                for root, dirs ,files in walk(unrecognize_path):
                    #source = './unrecognize/'+files
                    print("路徑：", root)
                    print("  目錄：", dirs)
                    print("  檔案：", files)
                if recognize_path != None:
                    #print("~~~~")
                    # txtpath = sign 有幾張 txtpath2 網頁端用
                    txtpath = recognize_path + 'sign.txt'
                    
                    txtpath2 = recognize_path + 'flag.txt'
                    
                    len = 0
                    for file in os.listdir(recognize_path):
           
                        print("檔案：", file)
                        len+=1
                    f = open(txtpath, 'w')
                    f.write(str(len-2))
                    f.close()
                    f = open(txtpath2, 'w')
                    f.write(str(len-2))
                    f.close()
            ######################## FTP ##########################
            ftp = FTP()                                           #
            HostName = '120.105.129.164'                          #
            UserName = 'root'                                     #
            PassWord = 'nuuCSIE406'                               #
            timeout = 30                                          #
            port = 21                                             #
            ftp_path = recognize_path                #影像檔案路徑 #
            ftp_patherror = recognize_path + "error" #錯誤檔案路徑 #
            #######################################################
            ftp.connect(HostName,port) # 連線FTP伺服器
            ftp.login(UserName,PassWord) # 登入
            #list = ftp.nlst()       # 獲得目錄列表
            #print(list)
            for file in os.listdir(ftp_path):
                print(file)
                if(file == 'error'):
                    continue
                fp = open(ftp_path + '/' + str(file), 'rb')
                ftp.storbinary('STOR '+file, fp ,1024) # 上傳FTP檔案
                print(ftp_path + '/' + str(file))
            ftp.cwd("error/")
            for file in os.listdir(ftp_patherror):
                print(file)
                fp = open(ftp_patherror + '/' + str(file), 'rb')
                ftp.storbinary('STOR '+file, fp ,1024) # 上傳FTP檔案
                print(ftp_patherror + '/' + str(file))
            fp.close()
            ftp.quit()                  # 退出FTP伺服器

            #清空 unrecognize
            if os.path.isfile(unrecognize_path+action+".mp4"):
                os.remove(unrecognize_path+action+".mp4")

            print("file created:{0}".format(event.src_path))

    def on_deleted(self, event):
        if event.is_directory:
            print("directory deleted:{0}".format(event.src_path))
        else:
            print("file deleted:{0}".format(event.src_path))

    def on_modified(self, event):
        if event.is_directory:
            print("directory modified:{0}".format(event.src_path))
        else:
            print("file modified:{0}".format(event.src_path))


if __name__ == "__main__":
    
    import time
    observer = Observer()
    event_handler = FileEventHandler()
    observer.schedule(event_handler, unrecognize_path, True)
    observer.start()
    try:
        while True:

            time.sleep(1)
            
    except KeyboardInterrupt:
        observer.stop()

