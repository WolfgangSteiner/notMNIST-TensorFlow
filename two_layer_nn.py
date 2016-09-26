
# coding: utf-8

# Deep Learning
# =============
#
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import sys

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_data = save['train_dataset']
  train_labels = save['train_labels']
  validation_data = save['valid_dataset']
  validation_labels = save['valid_labels']
  test_data = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_data.shape, train_labels.shape)
  print('Validation set', validation_data.shape, validation_labels.shape)
  print('Test set', test_data.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

def weight_variable(dim):
    return tf.Variable(tf.truncated_normal(dim))

def bias_variable(length):
    return tf.Variable(tf.truncated_normal([length]))

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act = tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            weights = weight_variable([input_dim, output_dim])
        with tf.name_scope("biases"):
            biases = bias_variable([output_dim])
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            activations = act(preactivate, 'activation')
            return activations


train_data, train_labels = reformat(train_data, train_labels)
validation_data, validation_labels = reformat(validation_data, validation_labels)
test_data, test_labels = reformat(test_data, test_labels)

print('Training set', train_data.shape, train_labels.shape)
print('Validation set', validation_data.shape, validation_labels.shape)
print('Test set', test_data.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (1.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))  / predictions.shape[0])

def feed_dict_test(test_data, test_labels):
    return {
        tf_input : test_data,
        tf_labels : test_labels,
        tf_regularization : 0.0,
        tf_learning_rate: 0.0 }

def feed_dict_train(batch_data, batch_labels, regularization, learning_rate):
    return {
        tf_input : batch_data,
        tf_labels : batch_labels,
        tf_regularization : 0.01,
        tf_learning_rate: learning_rate }

image_size = 28
num_labels = 10
batch_size = 128
num_neurons_1 = 392
num_neurons_2 = num_neurons_1 / 2

tf_graph = tf.Graph()
with tf_graph.as_default():
    # initialize_data
    tf_input = tf.placeholder(tf.float32, shape=(None, image_size * image_size))
    tf_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
    tf_regularization = tf.placeholder(tf.float32, shape=())
    tf_learning_rate = tf.placeholder(tf.float32, shape=())

    w1 = weight_variable([image_size * image_size, num_neurons_1])
    b1 = bias_variable(num_neurons_1)
    w2 = weight_variable([num_neurons_1, num_neurons_2])
    b2 = bias_variable(num_neurons_2)
    w3 = weight_variable([num_neurons_2, num_labels])
    b3 = bias_variable(num_labels)

    tf_logits = tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf_input, w1) + b1), w2) + b2), w3) + b3
    tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf_logits, tf_labels)) \
        + tf_regularization * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3))
    tf_optimizer = tf.train.AdagradOptimizer(0.1).minimize(tf_loss)
    tf_prediction = tf.nn.softmax(tf_logits)

num_steps = 40000
with tf.Session(graph=tf_graph) as session:
    tf.initialize_all_variables().run()
    learning_rate = 0.25
    print("Initialized")

    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_data[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        _, loss, predictions = session.run([tf_optimizer, tf_loss, tf_prediction], feed_dict=feed_dict_train(batch_data, batch_labels, 0.01, learning_rate))

        if (step % 100 == 0):
            print("")
            print("Step: %d" % (step))
            print("Minibatch loss: %f" % (loss))
            print("Minibatch accuracy: %.1f%%" % (accuracy(predictions, batch_labels) * 100.0))

            validation_predictions, = session.run([tf_prediction], feed_dict = feed_dict_test(validation_data, validation_labels))
            validation_accuracy = accuracy(validation_predictions, validation_labels)
            print("Validation accuracy: %.1f%%" % (validation_accuracy * 100.0))
            print("Learning rate: %.2f" % learning_rate)

            if (validation_accuracy > 0.85):
                learning_rate = 0.125 / 4
            elif (validation_accuracy > 0.80):
                learning_rate = 0.125 / 2
            elif (validation_accuracy > 0.75):
                learning_rate = 0.125

    test_predictions, = session.run([tf_prediction], feed_dict = feed_dict_test(test_data, test_labels))
    print("Test accuracy: %.1f%%" % (accuracy(test_predictions, test_labels) * 100.0))
