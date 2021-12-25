import csv
import pandas as pd
import numpy as np
from os import walk

path = "./data/csv/Sumo_Squat/AVG/173"
def writecsv():
    with open(path+'./AVG.csv','w',newline='') as csvf:
        writer = csv.writer(csvf)
        tmp_data=[]
        for i in range(0,25):
            #extend 是[0],[1],[2]這樣+ append是[012],[012],[012]的+
            tmp_data.extend(['x'+str(i),'y'+str(i)])
        #print(tmp_data)
        # 寫入一列資料
        writer.writerow(tmp_data)
    return()

writecsv()

for root, dirs ,files in walk(path):
    
    #print("路徑：", root)
    #print("  目錄：", dirs)
    #print("  檔案：", files)
    for file in files:
        if file =='AVG.csv':
            continue
        with open(root+"/"+file, newline='') as csvfile:
            # 以冒號分隔欄位，讀取檔案內容
            #rows = csv.reader(csvfile, delimiter=':')
            # 讀取 CSV 檔內容，將每一列轉成一個 dictionary
            rows = csv.DictReader(csvfile)
            #用panda 知道csv的列數
            df = pd.read_csv(root+"/"+file)
            #print (len(df))

            #x0=0.0
            x0=x1=x2=x3=x4=x5=x6=x7=x8=x9=x10=x11=x12=x13=x14=x15=x16=x17=x18=x19=x20=x21=x22=x23=x24=0.0
            y0=y1=y2=y3=y4=y5=y6=y7=y8=y9=y10=y11=y12=y13=y14=y15=y16=y17=y18=y19=y20=y21=y22=y23=y24=0.0
            Ax = []
            Ay = [] 

            #總和
            for row in rows:
                x0 = x0 + float(row['x0'])
                x1 = x1 + float(row['x1'])
                x2 = x2 + float(row['x2'])
                x3 = x3 + float(row['x3'])
                x4 = x4 + float(row['x4'])
                x5 = x5 + float(row['x5'])
                x6 = x6 + float(row['x6'])
                x7 = x7 + float(row['x7'])
                x8 = x8 + float(row['x8'])
                x9 = x9 + float(row['x9'])
                x10 = x10 + float(row['x10'])
                x11 = x11 + float(row['x11'])
                x12 = x12 + float(row['x12'])
                x13 = x13 + float(row['x13'])
                x14 = x14 + float(row['x14'])
                x15 = x15 + float(row['x15'])
                x16 = x16 + float(row['x16'])
                x17 = x17 + float(row['x17'])
                x18 = x18 + float(row['x18'])
                x19 = x19 + float(row['x19'])
                x20 = x20 + float(row['x20'])
                x21 = x21 + float(row['x21'])
                x22 = x22 + float(row['x22'])
                x23 = x23 + float(row['x23'])
                x24 = x24 + float(row['x24'])
                y0 = y0 + float(row['y0'])
                y1 = y1 + float(row['y1'])
                y2 = y2 + float(row['y2'])
                y3 = y3 + float(row['y3'])
                y4 = y4 + float(row['y4'])
                y5 = y5 + float(row['y5'])
                y6 = y6 + float(row['y6'])
                y7 = y7 + float(row['y7'])
                y8 = y8 + float(row['y8'])
                y9 = y9 + float(row['y9'])
                y10 = y10 + float(row['y10'])
                y11 = y11 + float(row['y11'])
                y12 = y12 + float(row['y12'])
                y13 = y13 + float(row['y13'])
                y14 = y14 + float(row['y14'])
                y15 = y15 + float(row['y15'])
                y16 = y16 + float(row['y16'])
                y17 = y17 + float(row['y17'])
                y18 = y18 + float(row['y18'])
                y19 = y19 + float(row['y19'])
                y20 = y20 + float(row['y20'])
                y21 = y21 + float(row['y21'])
                y22 = y22 + float(row['y22'])
                y23 = y23 + float(row['y23'])
                y24 = y24 + float(row['y24'])
            
            Ax.extend([x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24])
            Ay.extend([y0,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20,y21,y22,y23,y24])
            for i in range(0,25):
                Ax[i]=Ax[i]/len(df)
                Ay[i]=Ay[i]/len(df)

            print(Ax,"\n",Ay)
            with open(root+'./AVG.csv','a+',newline='') as csvf:
                writer = csv.writer(csvf)
                data=[]
                for i in range(0,25):
                    data.extend([str(Ax[i]),str(Ay[i])])
                #print(tmp_data)
                # 寫入一列資料
                writer.writerow(data)
           
            
