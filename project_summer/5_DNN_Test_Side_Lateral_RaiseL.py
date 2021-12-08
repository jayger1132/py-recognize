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
    return score[1]

df_start = pd.read_csv('./data/csv/Biceps_curl/AVG/173/Biceps_curl4.csv')
df_start['target'] = TARGET_START
df_ready = pd.read_csv('./data/csv/Biceps_curl/AVG/173/Biceps_curl0.csv')
df_ready['target'] = TARGET_READY

tmodel = keras.models.load_model('./data/model/model_Biceps_curl')
scores = []
rgs = np.arange(1, 0, -0.2)

for i in rgs:
    acc = testAcc(tmodel, i)
    scores.append(acc)
#(x,y,color)
plt.plot(rgs, scores,color=(255/255,100/255,100/255))
#調整Y軸顯示範圍
plt.ylim(0, 1)
#顯示網格
plt.grid(True)
plt.xlabel("Sampling rate")
plt.ylabel("Accuracy (%)")
plt.show()

print(scores)