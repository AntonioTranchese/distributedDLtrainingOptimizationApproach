import numpy as np
import pandas as pd
import matplotlib as plt
import tensorflow as tf

#Read file containing dataset
filename = '/home/picocluster/005/ch_ID005Accel.csv'
df = pd.read_csv(filename)

#Variables to handle the temporal value
titles = ['AccelerationX(g)','AccelerationY(g)','AccelerationZ(g)']
feature_keys = ['Accel X (g)','Accel Y (g)', 'Accel Z (g)']
date_time_key = 'Timestamp (ms)'

#Variables for preprocessing
split_fraction = 0.7
train_split = int(split_fraction * int(df.shape[0]))

#Variables for timeseries generation
step = 6
past = 720
future = 72

#Preprocessing function
def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

print(
    "The selcted parameters are:",
    ", ".join([titles[i] for i in [0, 1, 2]]),
)
selected_features = [feature_keys[i] for i in [0, 1, 2]]
features = df[selected_features]
features.index = df[date_time_key]

features = normalize(features.values, train_split)
features = pd.DataFrame(features)

print('DataFrame length:')
print(int(features.shape[0]))

#Train_test split
start = past + future
end = start + train_split

train_data = features.loc[0 : train_split - 1]
test_data = features.loc[train_split:]

x_train = train_data[[i for i in range(3)]].values
y_train = features.iloc[start:end][[1]]

print('x_train and y_train shapes:')
print(x_train.shape)
print(y_train.shape)


x_end = len(test_data) - past - future

label_start = train_split + past + future

x_test = test_data[:x_end][[i for i in range(3)]].values
y_test = features.iloc[label_start:][[1]]

print('x_test and y_test shapes:')
print(x_test.shape)
print(y_test.shape)

#Save
features.to_csv('/home/picocluster/005/ch_ID005Accel_normalized.csv')

#Conversion in DataFrame
df_x_train = pd.DataFrame(x_train)
df_x_test = pd.DataFrame(x_test)

df_x_train.to_csv('/home/picocluster/005/ch_ID005Accel_normalized_train.csv')
df_x_test.to_csv('/home/picocluster/005/ch_ID005Accel_normalized_test.csv')

#Split the numpy array into num_chunks equal parts
num_chunks = 10

chunks_train = np.array_split(x_train, num_chunks)
chunks_test = np.array_split(x_test, num_chunks)

for i, chunk in enumerate(chunks_train):
 print(f'Saving {i}-th chunk_train')
 chunk_df = pd.DataFrame(chunk)
 chunk_df.to_csv(f'/nfs/train/split_file_train_{i}.csv')

for i, chunk in enumerate(chunks_test):
 print(f'Saving {i}-th chunk_test')
 chunk_df = pd.DataFrame(chunk)
 chunk_df.to_csv(f'/nfs/test/split_file_test_{i}.csv')


