import json
import os
import sys
import tensorflow as tf
import dataset_setup
import time
import matplotlib.pyplot as plt
import tempfile
from multiprocessing import util

#tf_config must be placed here!
tf_config = {
    'cluster': {
        'worker': ['192.168.31.110:11111', '192.168.31.111:11111', '192.168.31.112:11111', '192.168.31.113:11111', '192.168.31.114:11111']
    },
    'task': {'type': 'worker', 'index': 0}
} #Only 'index' is different between workers

#Local batch size
per_worker_batch_size = 128 #256 #128 #16 #64

#tf_config loading
os.environ['TF_CONFIG'] = json.dumps(tf_config)
tf_config = json.loads(os.environ['TF_CONFIG'])

#Util variable for global batch size
num_workers = len(tf_config['cluster']['worker'])

global_batch_size = per_worker_batch_size * num_workers

num_epochs = 3


train_loss_results = []

# Checkpoint saving and restoring
def _is_chief(task_type, task_id, cluster_spec):
  return (task_type is None
          or task_type == 'chief'
          or (task_type == 'worker'
              and task_id == 0
              and 'chief' not in cluster_spec.as_dict()))

"""
def _get_temp_dir(dirpath, task_id):
  base_dirpath = 'workertemp_' + str(task_id)
  temp_dir = os.path.join(dirpath, base_dirpath)
  tf.io.gfile.makedirs(temp_dir)
  return temp_dir

def write_filepath(filepath, task_type, task_id, cluster_spec):
  dirpath = os.path.dirname(filepath)
  base = os.path.basename(filepath)
  if not _is_chief(task_type, task_id, cluster_spec):
    dirpath = _get_temp_dir(dirpath, task_id)
  return os.path.join(dirpath, base)
"""

#Checkpoint directory for saving
checkpoint_dir = os.path.join(tempfile.gettempdir(), 'ckpt')

#Define Strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
#Model building and compiling need to be within scope
    multi_worker_model = dataset_setup.simple_LSTM_model()

#Training dataset ditribution
    train_dist_dataset = strategy.distribute_datasets_from_function(
        lambda input_context: dataset_setup.tfrecord_train_dataset(global_batch_size, input_context))

#Testing dataset distribution
    test_dist_dataset = strategy.distribute_datasets_from_function(
        lambda input_context: dataset_setup.tfrecord_test_dataset(global_batch_size, input_context))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')
    test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')

#Set reduction to 'NONE' so I can do the reduction yourself
    loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        loss = tf.nn.compute_average_loss(per_example_loss)
        return loss

    test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss')

#Checkpoint that tracks the model with optimizer
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=multi_worker_model)

#Input function to 'strategy.run'
def train_step(inputs):
    x, y = inputs

    with tf.GradientTape() as tape:
        predictions = multi_worker_model(x, training=True)
        loss = compute_loss(y, predictions)

    gradients = tape.gradient(loss, multi_worker_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, multi_worker_model.trainable_variables))

    train_accuracy.update_state(y, predictions)
    return loss

def test_step(inputs):
    x, y = inputs

    predictions = multi_worker_model(x, training=False)
    t_loss = loss_object(y, predictions)

    test_loss.update_state(y, predictions) #update_state(t_loss) does not work
    test_accuracy.update_state(y, predictions)

#`run` replicates the provided computation and runs it
#with the distributed input.
#Functions for distributed computation
@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

@tf.function
def distributed_test_step(dataset_inputs):
    return strategy.run(test_step, args=(dataset_inputs,))

start_time = time.perf_counter() #training time measurement
print('Training started')
for epoch in range(num_epochs):
    #train loop
    total_loss = 0.0
    num_batches = 0
    for x in train_dist_dataset:
        total_loss += distributed_train_step(x)
        num_batches += 1
    train_loss = total_loss / num_batches

    #test loop
    for x in test_dist_dataset:
        distributed_test_step(x)

    checkpoint.save(checkpoint_dir)

    template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
              "Test Accuracy: {}")
    print(template.format(epoch, train_loss,
                            train_accuracy.result() * 100, test_loss.result(),
                            test_accuracy.result() * 100))

    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()

end_time = time.perf_counter()
#print(f'Total training time: {end_time - start_time:.2f} seconds')
print(f'Total training time: {((end_time-start_time)/60):.2f} minutes.')
