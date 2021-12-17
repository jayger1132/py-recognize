from watchdog.observers import Observer
from watchdog.events import *
from os import walk
from ftplib import FTP
import numpy as np
import cv2
import os
import csv
import shutil
action = 'Side_Lateral_RaiseL'
recognize_path = "./img/"+action+"/"

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
            print("file created:{0}".format(event.src_path))

    def on_deleted(self, event):
        if event.is_directory:
            print("directory deleted:{0}".format(event.src_path))
        else:
            ftp.connect(HostName,port) # 連線FTP伺服器
            ftp.login(UserName,PassWord) # 登入
            list = ftp.nlst()       # 獲得目錄列表
            #print(list)
            #print(path)
            for file in os.listdir(ftp_patherror):
                print(file)
                fp = open(ftp_patherror + '/' + str(file), 'rb')
                ftp.storbinary('STOR '+file, fp ,1024) # 上傳FTP檔案
                print(ftp_patherror + '/' + str(file))
            for file in os.listdir(ftp_path):
                print(file)
                if(file == 'error'):
                    continue
                fp = open(ftp_path + '/' + str(file), 'rb')
                ftp.storbinary('STOR '+file, fp ,1024) # 上傳FTP檔案
                print(ftp_path + '/' + str(file))
            fp.close()
            ftp.quit()                  # 退出FTP伺服器
            print("file deleted:{0}".format(event.src_path))

    def on_modified(self, event):
        if event.is_directory:
            print("directory modified:{0}".format(event.src_path))
        else:
            print("file modified:{0}".format(event.src_path))

