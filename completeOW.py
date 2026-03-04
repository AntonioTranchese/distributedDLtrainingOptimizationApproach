import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras import layers
import os
import time

TRAIN_DIR = 'percorso/alla/cartella/train/'
TEST_DIR = 'percorso/alla/cartella/test/'

BATCH_SIZE = 128
EPOCHS = 1

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

def tfrecord_train_dataset():
    
    d = tf.data.Dataset.list_files('/nfs/train/*.tfrecords')
    
    #TFRecord reading
    d = d.interleave(tf.data.TFRecordDataset,
	cycle_length=1, #Temporal order preserved!
	num_parallel_calls=tf.data.AUTOTUNE, #Parsing can be parallel
	deterministic=True) #Output must be reconstructed in order!
    
    #Parsing
    d = d.map(map_fn,
	num_parallel_calls=tf.data.AUTOTUNE,
	deterministic=True)
    
    d = d.batch(BATCH_SIZE)
    d = d.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return d

def tfrecord_test_dataset():
    
    d = tf.data.Dataset.list_files('/nfs/test/*.tfrecords')
    
    #TFRecord reading
    d = d.interleave(tf.data.TFRecordDataset,
	cycle_length=1, #Temporal order preserved!
	num_parallel_calls=tf.data.AUTOTUNE, #Parsing can be parallel
	deterministic=True) #Output must be reconstructed in order!
    
    #Parsing
    d = d.map(map_fn,
	num_parallel_calls=tf.data.AUTOTUNE,
	deterministic=True)
    
    d = d.batch(BATCH_SIZE)
    d = d.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return d


train_dataset = tfrecord_train_dataset()
test_dataset = tfrecord_test_dataset()

#METTI IL MODELLO QUA.
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

model = simple_LSTM_model()

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

start_time = time.perf_counter() #training time measurement
print('Training started')
print(f"Avvio test rapido - Batch Size: {BATCH_SIZE}, Epoche: {EPOCHS}...")
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS
)
end_time = time.perf_counter()
#print(f'Total training time: {end_time - start_time:.2f} seconds')
print(f'Total training time: {((end_time-start_time)/60):.2f} minutes.')

model.save('modello_lstm_test.h5')
print("Test completato con successo.")
