import numpy as np
import csv
import pandas as pd
import numpy as np
from os import walk
path = "./data/csv/Biceps_curl/AVG"
with open(path+"/AVG.csv", newline='') as csvfile:
            # 以冒號分隔欄位，讀取檔案內容
            rows = csv.reader(csvfile)
            # 讀取 CSV 檔內容，將每一列轉成一個 dictionary
            #rows = csv.DictReader(csvfile)
            #用panda 知道csv的列數
            df = pd.read_csv(path+"/AVG.csv")
            #print (len(df))
            column, row = 5, 25


            Ax = np.zeros((column,row))
            Ay = np.zeros((column,row))
            j=1
            #將AVG 的x,y 值存入陣列中
            for row in rows:
                if row[0] == "x0" :
                    continue
                else:
                    o=0
                    for i in range (0,25):
                        Ax[j][o] = row[2*i]
                        #print(Ax[j][o])
                        Ay[j][o] = row[2*i+1]
                        #print(Ay[j][o])
                        o+=1
                    j=j+1
           
            print("1級",np.linalg.norm(Ax[1]-Ax[1]))
            print("2級",np.linalg.norm(Ax[2]-Ay[2]))
            print("3級",np.linalg.norm(Ax[3]-Ay[3]))
            print("4級",np.linalg.norm(Ax[4]-Ay[4]))