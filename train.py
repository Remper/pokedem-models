from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import argparse
import gzip
import numpy as np
import time

"""
    Basic deep neural network that works with SVM input
"""

parser = argparse.ArgumentParser(description='Basic deep neural network that works with SVM input')
parser.add_argument('--train', default='', required=True, help='Location of the training set', metavar='#')
parser.add_argument('--layers', default=5, help='Amount of hidden layers', metavar='#')
parser.add_argument('--units', default=256, help='Amount of hidden units per layer', metavar='#')
parser.add_argument('--batch_size', default=256, help='Amount of samples in a batch', metavar='#')
args = parser.parse_args()

args.batch_size = int(args.batch_size)
args.units = int(args.units)
args.layers = int(args.layers)

print("Initialized with settings:")
print(vars(args))


def train_set_metadata(filename):
    """
        Figures out amount of labels and features
    """
    max_index = 0
    labels = {}
    with open(filename, 'rb') as reader:
        for line in reader:
            row = line.split(' ')
            if row[0] not in labels:
                labels[row[0]] = len(labels)
            for ele in row[1:]:
                index = int(ele.split(':')[0])
                if index > max_index:
                    max_index = index
    return max_index, labels


epoch_counter = 0
def train_set_reader(filename, max_index, labels):
    """
        Reads training set one by one
    """
    global epoch_counter
    epoch_counter = 0
    while True:
        with open(filename, 'rb') as reader:
            for line in reader:
                row = line.split(' ')
                label = np.zeros(len(labels), dtype=np.float32)
                label[int(row[0])] = 1.0
                features = np.zeros(max_index, dtype=np.float32)
                for ele in row[1:]:
                    feature = ele.split(':')
                    features[int(feature[0])-1] = float(feature[1])
                yield label, features
        print("Epoch %d concluded" % epoch_counter)
        epoch_counter += 1


def produce_batch(reader, batch_size):
    """
        Produces full batch ready to be input in NN
    """
    labels = list()
    batch = list()
    while len(labels) < batch_size:
        label, features = reader.next()
        labels.append(label)
        batch.append(features)
    return batch, labels


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
  return tf.Variable(initial)


# Restoring embeddings and preparing reader to produce batches
print("Figuring out training set metadata")
timestamp = time.time()
max_index, labels = train_set_metadata(args.train)
print("Done in", ("%.5f" % (time.time() - timestamp)) + "s")
print("Initializing training set reader")
reader = train_set_reader(args.train, max_index, labels)

print("Test batch:")
batch, labels = produce_batch(reader, 2)
for idx, _ in enumerate(labels):
    print("  Features: ", batch[idx])
    print("  Label: ", labels[idx])
    print("")

# Defining dataflow graph
graph = tf.Graph()

with graph.as_default():
    # Graph begins with input. tf.placeholder tells Tensorflow that we will input those variables at each iteration
    train_features = tf.placeholder(tf.float32, shape=[args.batch_size, max_index])
    train_labels = tf.placeholder(tf.float32, shape=[args.batch_size, len(labels)])


    # Multiple dense layers (fully connected linear + tanh nonlinearity with random dropouts to help with overfitting)
    input_size = max_index
    hidden_units = args.units
    layer = train_features
    for idx in range(args.layers):
        with tf.name_scope("dense_layer"):
            weights = weight_variable([input_size, hidden_units])
            biases = bias_variable([hidden_units])
            hidden = tf.nn.tanh(tf.matmul(layer, weights) + biases)
            layer = tf.nn.dropout(hidden, 0.8)
            input_size = hidden_units

    # Linear layer before softmax
    weights = weight_variable([input_size, 2])
    biases = bias_variable([2])
    layer = tf.matmul(layer, weights) + biases

    # Softmax and cross entropy in the end
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(layer, train_labels))
    loss_summary = tf.scalar_summary("loss", loss)
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)


# Main execution
with tf.Session(graph=graph) as session:
    # Initializing everythin
    writer = tf.train.SummaryWriter("logs", graph)
    print("Initializing variables")
    timestamp = time.time()
    tf.initialize_all_variables().run()
    print("Done (" + ("%.5f" % (time.time() - timestamp)) + "s)")
    print("Starting training")

    # Main execution loop. Right now it is a fixed amount of iterations, you might want to change it to epochs later
    average_loss = 0
    timestamp = time.time()
    for step in xrange(100000):
        # Batch production is done on python side, normally you input word indices
        #  and then use Tensorflow to convert it to proper word vectors
        features, labels = produce_batch(reader, args.batch_size)
        # We are telling Tensorflow to perform 1 full forward-backpropagation step by including optimizer to the
        #  list of variables that we want to compute.
        _, loss_value, summary = session.run([optimizer, loss, loss_summary], feed_dict={
            train_features: features,
            train_labels: labels
        })
        # Writes loss_summary to log. Each call represents a single point on the plot
        writer.add_summary(summary, step)
        # Output average loss periodically
        average_loss += loss_value
        if step % 500 == 0:
            print("Average loss at step ("+str(step)+", "+("%.5f" % (time.time() - timestamp)) + "s): ", average_loss)
            timestamp = time.time()
            average_loss = 0

print("Done")