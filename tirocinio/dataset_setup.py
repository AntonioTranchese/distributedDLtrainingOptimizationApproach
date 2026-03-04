import os
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from tensorflow import keras
from tensorflow.keras import layers
import tempfile


"""
# Variables for splitting
past = 720
future = 72
# Variables for timeseries generation
step = 6
sequence_length = int(past / step)
shift = 1
split_fraction = 0.7
train_split = int(split_fraction * 5171400) #shape computation in another script
start = past + future
end = start + train_split
x_end = 1551420 - past - future
label_start = train_split + past + future


def timeseries_generation(x, y, sequence_length, shift, stride):
 x = x.window(size=120, shift=1, stride=6, drop_remainder=True)
 x = x.flat_map(lambda window: window.batch(batch_size=120, drop_remainder=True))
 dataset = tf.data.Dataset.zip((x, y))
 return dataset

def read_file(filename):
 select_columns = [1, 2, 3] #Excluded index
 column_defaults = tf.constant([0.0, 0.0, 0.0], dtype=tf.float64) #Preserved maximum precision
 building_dataset = tf.data.experimental.make_csv_dataset(
         filename,
         batch_size=5, #for simplicity
         column_defaults=column_defaults,
         select_columns=select_columns,
         num_epochs=1,
         shuffle=False)

 def pack_features_vector(features):
         #Pack the features into a single array.
         features = tf.stack([tf.cast(x, tf.float64) for x in list(features.values())], axis=1)
         return features


 building_dataset = building_dataset.map(pack_features_vector)
 building_dataset = building_dataset.unbatch() #the final resulst must have shape (120, 3)

 return building_dataset

#HAR = human activity recognition
def train_HAR_dataset(batch_size):
 filename = '/home/picocluster/ch_ID005Accel_normalized.csv'
 building_dataset = read_file(filename)

 train_data = building_dataset.take(train_split)

#choose the inputs and target
 x_train = train_data
#this line of code works beacuse i and data are named in this way!
 y_train = building_dataset.map(lambda x: x[1]).enumerate().skip(start).take_while(lambda i, _: i < end).map(lambda i, data: data)
 y_train = y_train.map(lambda x: tf.reshape(x, [1])) #no problem with custom training loop

#PUT here timeseries generation
 train_dataset = timeseries_generation(x_train, y_train, sequence_length, shift, step)

 return train_dataset

def test_HAR_dataset(batch_size):
 filename = '/home/picocluster/ch_ID005Accel_normalized_test.csv'

 building_dataset = read_file(filename)

 test_data = building_dataset

#choose the inputs and target
 x_test = test_data
#this line of code works beacuse i and data are named in this way!
 y_test = building_dataset.map(lambda x: x[1]).enumerate().skip(start).take_while(lambda i, _: i < end).map(lambda i, data: data)
 y_test = y_test.map(lambda x: tf.reshape(x, [1])) #no problem with custom training loop

#PUT here timeseries generation
 test_dataset = timeseries_generation(x_test, y_test, sequence_length, shift, step)

 return test_dataset


def train_dataset_fn(global_batch_size, input_context):
 batch_size = input_context.get_per_replica_batch_size(global_batch_size)
 dataset = train_HAR_dataset(batch_size)
 dataset = dataset.shard(input_context.num_input_pipelines,
			 input_context.input_pipeline_id)
 dataset = dataset.batch(batch_size)
 dataset = dataset.prefetch(tf.data.AUTOTUNE)
 return dataset

def test_dataset_fn(global_batch_size, input_context):
 batch_size = input_context.get_per_replica_batch_size(global_batch_size)
 dataset = test_HAR_dataset(batch_size)
 dataset = dataset.shard(input_context.num_input_pipelines,
                         input_context.input_pipeline_id)
 dataset = dataset.batch(batch_size)
 dataset = dataset.prefetch(tf.data.AUTOTUNE)
 return dataset


#Utility for TFRecord files generation
def chunk_dataset(filename):
 building_dataset = read_file(filename)

#choose the inputs and target
 x_test = building_dataset
#this line of code works beacuse i and data are named in this way!
 y_test = building_dataset.map(lambda x: x[1]).enumerate().skip(start).take_while(lambda i, _: i < end).map(lambda i, data: data)
 y_test = y_test.map(lambda x: tf.reshape(x, [1])) #no problem with custom training loop

#PUT here timeseries generation
 tsg_dataset = timeseries_generation(x_test, y_test, sequence_length, shift, step)

 return tsg_dataset
"""

#tf.data.AUTOTUNE = I let Tensorflow how to use the concerned resources

def simple_LSTM_model():
 time_steps = 120
 num_features = 3

 inputs = keras.Input(shape=(time_steps, num_features))

 lstm_out = layers.LSTM(32)(inputs)

 outputs = layers.Dense(1)(lstm_out)

#Model creation
 model = keras.Model(inputs=inputs, outputs=outputs)
 model.summary()

 return model


#Parsing function for tfrecord file
def map_fn(serialized_example):
    feature_description = {
        'feature': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)

    time_steps = 120  #Timeseries length
    num_features = 3

    #Tensor parsing from bytes
    #out_type = numpy array type!
    feature = tf.io.parse_tensor(example['feature'], out_type=tf.float32)
    label = tf.io.parse_tensor(example['label'], out_type=tf.float32)

    #Shape resetting
    feature.set_shape([time_steps, num_features])
    label.set_shape([1]) #It is important for CTL good working

    return feature, label


def tfrecord_train_dataset(global_batch_size, input_context):
#Local batch size
 batch_size = input_context.get_per_replica_batch_size(global_batch_size)

#Directory preparation
 d = tf.data.Dataset.list_files('/nfs/train/*.tfrecords')

#Synchronous data parallelism
 d = d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)

#TFRecord reading
 d = d.interleave(tf.data.TFRecordDataset,
	cycle_length=1, #Temporal order preserved!
	num_parallel_calls=tf.data.AUTOTUNE, #Parsing can be parallel
	deterministic=True) #Output must be reconstructed in order!

#Parsing
 d = d.map(map_fn,
	num_parallel_calls=tf.data.AUTOTUNE,
	deterministic=True)

#Batching
 d = d.batch(batch_size, drop_remainder=True) #All batches shapes equal

#Prefetching
 d = d.prefetch(tf.data.AUTOTUNE) #CPU prepares data while GPU works

 return d


def tfrecord_test_dataset(global_batch_size, input_context):
#Local batch size
 batch_size = input_context.get_per_replica_batch_size(global_batch_size)

#Directory preparation
 d = tf.data.Dataset.list_files('/nfs/test/*.tfrecords')

#Synchronous data parallelism
 d = d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)

#TFRecord reading
 d = d.interleave(tf.data.TFRecordDataset,
        cycle_length=1, #Temporal order preserved!
        num_parallel_calls=tf.data.AUTOTUNE, #Parsing can be parallel
        deterministic=True) #Output must be reconstructed in order!

#Parsing
 d = d.map(map_fn,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True)

#Batching
 d = d.batch(batch_size, drop_remainder=True) #All batches shapes equal

#Prefetching
 d = d.prefetch(tf.data.AUTOTUNE) #CPU prepares data while GPU works

 return d


