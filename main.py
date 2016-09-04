
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
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


num_steps = 801

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

class Classifier:
  image_size = 28
  num_labels = 10
  batch_size = 128
  tf_graph = None
  tf_optimizer = None
  tf_train_data = None
  tr_train_labels = None
  tf_validation_data = None
  tf_test_data = None
  tf_loss = None
  tf_train_prediction = None
  train_data = None
  train_labels = None
  test_data = None
  test_labels = None
  validation_data = None
  validation_labels = None

  def __init__(self, a_train_data, a_train_labels, a_validation_data, a_validation_labels, a_test_data, a_test_labels):
    self.train_data = a_train_data
    self.train_labels = a_train_labels
    self.validation_data = a_validation_data
    self.validation_labels = a_validation_labels
    self.test_data = a_test_data
    self.test_labels = a_test_labels
    self.initialize_graph()

  def initialize_parameters(self):
    pass

  def initialize_graph(self):
    pass

  def initialize_data(self):
    self.tf_train_data = tf.placeholder(tf.float32, shape=(self.batch_size, self.image_size * self.image_size))
    self.tr_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_labels))
    self.tf_validation_data = tf.constant(self.validation_data)
    self.tf_test_data = tf.constant(self.test_data)

  def train(self, aNumSteps):
    with tf.Session(graph=self.tf_graph) as session:
      tf.initialize_all_variables().run()
      print("Initialized")
      for step in range(aNumSteps):
        offset = (step * self.batch_size) % (self.train_labels.shape[0] - self.batch_size)
        batch_data = train_dataset[offset:(offset + self.batch_size), :]
        batch_labels = train_labels[offset:(offset + self.batch_size), :]
        feed_dict = {self.tf_train_data : batch_data, self.tr_train_labels : batch_labels}
        _, l, predictions = session.run([self.tf_optimizer, self.tf_loss, self.tf_train_prediction], feed_dict=feed_dict)

        if (step % 500 == 0):
          print("")
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(self.TfValidationPrediction.eval(), self.validation_labels))
        else:
          if (step % 10 == 0):
            sys.stdout.write('.')
            sys.stdout.flush()
          if (step % 50 == 0):
            sys.stdout.write(' ' + str(step % 500) + ' ')
            sys.stdout.flush()

      print("Test accuracy: %.1f%%" % accuracy(self.TfTestPrediction.eval(), test_labels))


class LogisticClassifier(Classifier):
  def initialize_parameters(self):
    self.tf_weights = tf.Variable(tf.truncated_normal([self.image_size * self.image_size, self.num_labels]))
    self.tf_bias = tf.Variable(tf.zeros([self.num_labels]))

  def initialize_graph(self):
    self.tf_graph = tf.Graph()
    with self.tf_graph.as_default():
      self.initialize_data()
      self.initialize_parameters()

      self.tf_logits = tf.matmul(self.tf_train_data, self.tf_weights) + self.tf_bias
      self.tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.tf_logits, self.tr_train_labels))
      self.tf_optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.tf_loss)
      self.tf_train_prediction = tf.nn.softmax(self.tf_logits)
      self.TfValidationPrediction = tf.nn.softmax(tf.matmul(self.tf_validation_data, self.tf_weights) + self.tf_bias)
      self.TfTestPrediction = tf.nn.softmax(tf.matmul(self.tf_test_data, self.tf_weights) + self.tf_bias)


class NeuralNetworkOneLayer(Classifier):
  NumNeurons = 1024;

  def initialize_parameters(self):
    self.w1 = tf.Variable(tf.truncated_normal([self.image_size * self.image_size, self.NumNeurons]))
    self.b1 = tf.Variable(tf.zeros([self.NumNeurons]))
    self.w2 = tf.Variable(tf.truncated_normal([self.NumNeurons, self.num_labels]))
    self.b2 = tf.Variable(tf.zeros([self.num_labels]))

  def initialize_graph(self):
    self.tf_graph = tf.Graph()
    with self.tf_graph.as_default():
      self.initialize_data()
      self.initialize_parameters()

      self.tf_logits = tf.matmul(tf.nn.relu(tf.matmul(self.tf_train_data, self.w1) + self.b1), self.w2) + self.b2;

      self.tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.tf_logits, self.tr_train_labels))
      self.tf_optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.tf_loss)
      self.tf_train_prediction = tf.nn.softmax(self.tf_logits)

      self.TfValidationPrediction = tf.nn.softmax(
        tf.matmul(tf.nn.relu(tf.matmul(self.tf_validation_data, self.w1) + self.b1), self.w2) + self.b2)

      self.TfTestPrediction = tf.nn.softmax(
        tf.matmul(tf.nn.relu(tf.matmul(self.tf_test_data, self.w1) + self.b1), self.w2) + self.b2)


c = NeuralNetworkOneLayer(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)
c.train(1000)
