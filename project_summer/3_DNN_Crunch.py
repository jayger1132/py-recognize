import numpy as np
import pandas as pd 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM
from sklearn.model_selection import train_test_split
TARGET_STRAIGHT = 0
TARGET_90 = 1
TARGET_UNDERKNEE = 2
TARGET_DIM = 3
def handleData(datas):
    #將資料都合併在一起 axis=0通常是直的合併 1為橫的
    d = pd.concat(datas,axis=0)
    #columns[:-1] 少一行
    ds = pd.DataFrame(d,columns=d.columns[:-1])
    tx = ds
    ty = keras.utils.to_categorical(d['target'], TARGET_DIM) #用陣列的方式儲存
    return (tx,ty)
def build_model_1():
    model = Sequential()
    # 全連階層(Dense)
    model.add(Dense(64, input_shape=(75, ), activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(4, activation='tanh'))
    #Softmax 運算使得最後一層輸出的機率分佈總和為
    model.add(Dense(TARGET_DIM, activation='softmax'))
    return model

df_straight = pd.read_csv('./data/csv/Crunch/Crunch_Straight.csv')
df_straight['target'] = TARGET_STRAIGHT
df_90 = pd.read_csv('./data/csv/Crunch/Crunch_90.csv')
df_90['target'] = TARGET_90
df_underknee = pd.read_csv('./data/csv/Crunch/Crunch_Underknee.csv')
df_underknee['target'] = TARGET_UNDERKNEE

df_straight_test = pd.read_csv('./data/csv/Crunch/Crunch_Straight_Test.csv')
df_straight_test['target'] = TARGET_STRAIGHT
df_90_test = pd.read_csv('./data/csv/Crunch/Crunch_90_Test.csv')
df_90_test['target'] = TARGET_90
df_underknee_test = pd.read_csv('./data/csv/Crunch/Crunch_Underknee_Test.csv')
df_underknee_test['target'] = TARGET_UNDERKNEE

#有時候格式跑掉可以先去第一排+個字在重新run 我是在Score旁+一行
X ,y = handleData([df_straight,df_90,df_underknee,df_straight_test,df_90_test,df_underknee_test])
#test_size=0.2 -> training : test = 8 : 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
#DF用法 , loc/iloc 差在iloc通常都是用數字(index去取資料) loc是用"name"
#print (X.iloc[:,0:2]) 這樣是 columns0~2
#print (X.iloc[:,::-1]) 這樣是 columns反過來輸出
#print (X.iloc[::-1,:]) 這樣是 index反過來輸出
#print (X.loc[:,"X"])
#單純輸出X沒辦法看到全部的資料，因為是concat合併的，就已經依照輸入的data來排列順序了
#print(X.iloc[:,:])


model = build_model_1()
model.summary()
# 使用categorical_crossentropy損失時，目標值應該是分類格式假設分為3，那就是3維的向量，除了表示類別index為1其他都為0
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

#開始訓練模型
#X_train.shape (列,行)
#train_history = model.fit(X_train, y_train, batch_size=X_train.shape[0], validation_data=(X_test, y_test), epochs=1500, verbose=0)
#verbose=2 為每個epoch輸出一行記錄 =1顯示進度條
train_history = model.fit(X_train, y_train, batch_size=X_train.shape[0], validation_data=(X_test, y_test), epochs=1000, verbose=1)
#顯示訓練結果
score = model.evaluate(X_train, y_train)
print ('Train Acc:', score[1] ,'Loss :' , score[0])
score = model.evaluate(X_test, y_test)
print ('Test Acc:', score[1], 'Loss :' , score[0])

model.save('./data/model/model_Crunch')