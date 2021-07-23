import numpy as np
import pandas as pd 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM

TARGET_STRAIGHT = 1
TARGET_BENDING = 0
TARGET_DIM = 2
def handleData(datas):
    #將資料都合併在一起 axis=0通常是直的合併 1為橫的
    d = pd.concat(datas,axis=0)
    #columns[:-1] 少一行
    ds = pd.DataFrame(d,columns=d.columns[:-1])
    tx = ds
    ty = keras.utils.to_categorical(d['target'], TARGET_DIM) #用陣列的方式儲存
    return (tx,ty)
df_straight = pd.read_csv('./data/csv/Biceps_curl_Straight.csv')
df_straight['target'] = TARGET_STRAIGHT
df_bending = pd.read_csv('./data/csv/Biceps_curl_Bending.csv')
df_bending['target'] = TARGET_BENDING 

X ,y = handleData([df_straight,df_bending])
#DF用法 , loc/iloc 差在iloc通常都是用數字(index去取資料) loc是用"name"
#print (X.iloc[:,0:2]) 這樣是 columns0~2
#print (X.iloc[:,::-1]) 這樣是 columns反過來輸出
#print (X.iloc[::-1,:]) 這樣是 index反過來輸出
#print (X.loc[:,"X"])
#單純輸出X沒辦法看到全部的資料，因為是concat合併的，就已經依照輸入的data來排列順序了
print(X)

