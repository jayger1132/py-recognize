#import numpy as np
#import pandas as pd 
#import keras
#from keras.models import Sequential
#from keras.layers.core import Dense, Activation, Dropout
#from keras.optimizers import Adam
#from keras.layers.recurrent import LSTM
#from sklearn.model_selection import train_test_split
#data = pd.read_csv('./data/csv/Biceps_curl_Bending.csv')

#TARGET_LYING = 1
#TARGET_DIM = 2

#data['target'] = TARGET_LYING
##print(data)

#ds = pd.DataFrame(data, columns=data.columns[:-1])
###print(ds)
#ty = keras.utils.to_categorical(data['target'], TARGET_DIM)
##print (ds,ty)
#ds_train, ds_test, ty_train, ty_test = train_test_split(ds, ty, test_size=0.33, random_state=0)
#print(ds_train)
#print(ds_test)
#print(ty_train)
#print(ty_test)

a = [1,2,3]
b = [2,4,5]
a.extend(b)
print(a)