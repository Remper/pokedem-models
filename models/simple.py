import time
import tensorflow as tf
import numpy as np
from flask import json

from models.model import Model
from tensorflow.contrib import slim

DEFAULT_LAYERS = 5
DEFAULT_UNITS = 256
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_EPOCHS = 100

DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_DROPOUT_RATE = 0.8


class SimpleModel(Model):
    def __init__(self, name, inputs, classes):
        Model.__init__(self, name)
        self._inputs = inputs
        self._classes = classes
        self.batch_size(DEFAULT_BATCH_SIZE).units(DEFAULT_UNITS).layers(DEFAULT_LAYERS)\
            .max_epochs(DEFAULT_MAX_EPOCHS).learning_rate(DEFAULT_LEARNING_RATE).dropout_rate(DEFAULT_DROPOUT_RATE)

    def batch_size(self, batch_size):
        self._batch_size = batch_size
        return self

    def units(self, units):
        self._units = units
        return self

    def layers(self, layers):
        self._layers = layers
        return self

    def max_epochs(self, max_epochs):
        self._max_epochs = max_epochs
        return self

    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate
        return self

    def dropout_rate(self, dropout_rate):
        self._dropout_rate = dropout_rate
        return self

    def _definition(self):
        graph = tf.Graph()
        with graph.as_default():
            # Graph begins with input. tf.placeholder tells TF that we will input those variables at each iteration
            self._train_features = tf.placeholder(tf.float32, shape=[self._batch_size, self._inputs])
            self._train_labels = tf.placeholder(tf.float32, shape=[self._batch_size, self._classes])

            # Multiple dense layers
            input_size = self._inputs
            hidden_units = self._units
            layer = self._train_features
            for idx in range(self._layers):
                with tf.name_scope("dense_layer"):
                    weights = self.weight_variable([input_size, hidden_units])
                    biases = self.bias_variable([hidden_units])
                    hidden = tf.nn.relu(tf.matmul(layer, weights) + biases)
                    layer = tf.nn.dropout(hidden, self._dropout_rate)
                    input_size = hidden_units

            # Linear layer before softmax
            weights = self.weight_variable([input_size, self._classes])
            biases = self.bias_variable([self._classes])
            layer = tf.matmul(layer, weights) + biases

            # Softmax and cross entropy in the end
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self._train_labels, logits=layer)
            self._loss = tf.reduce_mean(losses)
            self._prediction = tf.nn.softmax(layer)
            tf.summary.scalar("loss", self._loss)
            self._global_step = tf.train.get_or_create_global_step()
            self._optimizer = slim.optimize_loss(loss=self._loss, global_step=self._global_step, learning_rate=None,
                                                 optimizer=tf.train.AdamOptimizer(), clip_gradients=5.0)
            self._saver = tf.train.Saver()
        return graph

    def train(self, batch_producer):
        self._init()
        with self._session as sess:
            # Main execution
            check_interval = 500
            # Initializing everything
            writer = tf.summary.FileWriter(logdir="logs")
            g_summary = tf.summary.merge_all()
            print("Initializing variables")
            timestamp = time.time()
            tf.global_variables_initializer().run()
            print("Done in %.5fs" % (time.time() - timestamp))
            print("Starting training")

            # Main execution loop
            average_loss = 0
            timestamp = time.time()
            tolerance_margin = 10
            tolerance = tolerance_margin + 1
            min_loss = -1
            while batch_producer.current_epoch < self._max_epochs and tolerance > 0:
                features, labels = batch_producer.produce(self._batch_size)
                _, loss_value, global_step, summary_v = self._session.run(
                    [self._optimizer, self._loss, self._global_step, g_summary],
                    feed_dict={
                        self._train_features: features,
                        self._train_labels:   labels
                    })

                # Writes loss_summary to log. Each call represents a single point on the plot
                writer.add_summary(summary=summary_v, global_step=global_step)

                # Output average loss periodically
                average_loss += loss_value
                if global_step % check_interval == 0 and global_step > 0:
                    average_loss /= check_interval
                    if min_loss < average_loss:
                        tolerance -= 1
                    else:
                        if tolerance < tolerance_margin:
                            tolerance += 1
                    if min_loss > average_loss or min_loss == -1:
                        min_loss = average_loss
                    print("[+] step: %d, %.2f steps/s, tol: %2d, epoch: %2d, avg.loss: %.5f, min.loss: %.5f"
                          % (global_step, float(check_interval) / (time.time() - timestamp),
                             tolerance, batch_producer.current_epoch, average_loss, min_loss))
                    timestamp = time.time()
                    average_loss = 0
            if batch_producer.current_epoch >= self._max_epochs:
                print("Amount of epochs reached")
            if tolerance <= 0:
                print("Tolerance margin reached")
        self._ready = True

    def predict(self, features):
        self._init()
        self._check_if_ready()
        features = np.array(features).reshape(self._batch_size, self._inputs)
        return self._prediction.eval(session=self._session, feed_dict={self._train_features: features})

    @staticmethod
    def restore_definition(filename):
        params = json.load(open(filename + '.json', 'rb'))
        model = SimpleModel(params["name"], params["inputs"], params["classes"])
        return model

    def save_to_file(self, filename):
        Model.save_to_file(self, filename)
        json.dump({
            'name': self._name,
            'inputs': self._inputs,
            'classes': self._classes
        }, open(filename+'.json', 'wb'))