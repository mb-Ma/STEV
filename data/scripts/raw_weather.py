import numpy as np
import pickle as pkl
import pandas as pd

with open('test.pkl', 'rb') as f:
    test = pkl.load(f)

test_list = []
for i in range(test['x'].shape[0]):
    test_list.append(test['x'][i, :, ::5, 0])
    test_list.append(test['y'][i, :, ::5, 0])
test_log = np.concatenate(test_list, axis=0)

with open('val.pkl', 'rb') as f:
    val = pkl.load(f)

val_list = []
for i in range(val['x'].shape[0]):
    val_list.append(val['x'][i, :, ::5, 0])
    val_list.append(val['y'][i, :, ::5, 0])
val_log = np.concatenate(val_list, axis=0)

with open('trn.pkl', 'rb') as f:
    train = pkl.load(f)

train_list = []
for i in range(train['x'].shape[0]):
    train_list.append(train['x'][i, :, ::5, 0])
    train_list.append(train['y'][i, :, ::5, 0])
train_log = np.concatenate(train_list, axis=0)

time = pd.date_range(start="2010-01-01", end="2018-12-31", freq="H")

data = np.concatenate([train_log, val_log, test_log], axis=0)

df = pd.DataFrame(data)
df.index = time[:-1]

df.to_csv('./weather_temp.csv', index=True)