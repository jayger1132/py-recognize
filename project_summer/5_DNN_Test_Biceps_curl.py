import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils.np_utils import *

TARGET_READY = 0
TARGET_START = 1
TARGET_DIM = 2

def handleData(datas, frac ):
    od = pd.concat(datas, axis=0)
    
    if frac is not None:
        d = od.sample(frac = float(frac))
    else:
        d = od
    ds = pd.DataFrame(d, columns=d.columns[:-1])
    tx = ds
    ty = to_categorical(d['target'], TARGET_DIM)
    return (tx, ty)

def testAcc(_model, frac = None):
    tx, ty = handleData([df_start, df_ready], frac)
    
    score = _model.evaluate(tx, ty , batch_size=1)
    print ('Acc:', score[1])
    return score[1]*100

df_start = pd.read_csv('./data/csv/Biceps_curl/Analys_Start.csv')
df_start['target'] = TARGET_START
df_ready = pd.read_csv('./data/csv/Biceps_curl/Analys_Ready.csv')
df_ready['target'] = TARGET_READY

tmodel = keras.models.load_model('./data/model/model_Biceps_curl')
scores = []
rgs = np.arange(1, 0, -0.1)


for i in rgs:
    acc = testAcc(tmodel, i)
    scores.append(acc)
#(x,y,color)
plt.plot(rgs, scores,color=(0/255,0/255,0/255),linewidth=3.0)
#調整Y軸顯示範圍
plt.ylim(90, 100)
# 設定刻度字型大小
plt.xticks(fontsize=20,family = "Times New Roman")
plt.yticks(fontsize=20,family = "Times New Roman")
#顯示網格
plt.grid(True)
plt.xlabel("Sampling rate" ,family = "Times New Roman", size = 25)
plt.ylabel("Accuracy (%)" ,family = "Times New Roman", size = 25)
plt.show()

print(scores)