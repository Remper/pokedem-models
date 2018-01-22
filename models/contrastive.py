import time
import tensorflow as tf
import numpy as np
import csv
from flask import json

from models.model import Model

DEFAULT_LAYERS = 5
DEFAULT_UNITS = 256
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_EPOCHS = 10

DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_DROPOUT_RATE = 0.9


class ContrastiveModel(Model):
    def __init__(self, name, inputs, classes=1):
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
            # Graph begins with input. tf.placeholder tells Tensorflow that we will input those variables at each iteration
            self._train_label = tf.placeholder(tf.float32, shape=[self._batch_size])
            self._train_features = tf.placeholder(tf.float32, shape=[self._batch_size, self._inputs])
            train_input1, train_input2 = tf.split(1, 2, self._train_features, name='input_split')

            # Multiple dense layers
            input_size = self._inputs/2
            hidden_units = self._units
            layers = [train_input1, train_input2]
            hist_summaries = []
            for idx in range(self._layers):
                with tf.name_scope("dense_layer"):
                    weights = self.weight_variable([input_size, hidden_units])
                    biases = self.bias_variable([hidden_units])
                    hist_summaries.append(tf.histogram_summary("dense_weights_"+str(idx), weights))
                    hist_summaries.append(tf.histogram_summary("dense_biases_"+str(idx), biases))
                    new_layers = []
                    for layer in layers:
                        hidden = tf.nn.relu(tf.matmul(layer, weights) + biases)
                        hidden = tf.nn.dropout(hidden, self._dropout_rate)
                        new_layers.append(hidden)
                    layers = new_layers
                    input_size = hidden_units

            # Linear layer before softmax
            with tf.name_scope("pre_softmax_linear_layer"):
                weights = self.weight_variable([input_size, self._classes])
                biases = self.bias_variable([self._classes])
                new_layers = []
                for layer in layers:
                    layer = tf.matmul(layer, weights) + biases
                    new_layers.append(tf.squeeze(layer, [1]))
                layers = new_layers

            # Logits to probabilities
            with tf.name_scope("cost"):
                score_diff = layers[0] - layers[1]
                exp_res = tf.exp(score_diff)
                cost = - self._train_label * score_diff + tf.log(1 + exp_res)

            # Softmax and custom objective in the end
            #  This is not a correct cross entropy
            #  batch_losses = tf.nn.sigmoid_cross_entropy_with_logits(layers[0] - layers[1], self._train_label)
            self._loss = tf.reduce_mean(cost)
            self._prediction = layers[0]
            self._pair_prediction = tf.div(exp_res, 1 + exp_res)
            self._loss_summary = tf.scalar_summary("loss", self._loss)
            self._hist_summaries = tf.merge_summary(hist_summaries)
            self._optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)
            self._saver = tf.train.Saver()
        return graph

    def train(self, train_prod, eval_prod=None):
        self._init()
        # Main execution
        check_interval = 500
        # Initializing everything
        writer = tf.train.SummaryWriter("logs", self._graph)
        print("Initializing variables")
        timestamp = time.time()
        with self._graph.as_default():
            tf.initialize_all_variables().run(session=self._session)
        print("Done in %.5fs" % (time.time() - timestamp))
        print("Starting training")

        # Main execution loop
        average_loss = 0
        timestamp = time.time()
        step = 0
        tolerance_margin = 20
        tolerance = tolerance_margin + 1
        min_loss = -1
        while train_prod.current_epoch < self._max_epochs and tolerance > 0:
            features, labels = train_prod.produce(self._batch_size)
            _, loss_value, summary = self._session.run([self._optimizer, self._loss, self._loss_summary], feed_dict={
                self._train_features: features,
                self._train_label: labels
            })
            # Writes loss_summary to log. Each call represents a single point on the plot
            writer.add_summary(summary, step)
            # Output average loss periodically
            average_loss += loss_value
            #if step % 10 == 0 and step > 0:
            #    writer.add_summary(self._session.run(self._hist_summaries), step)
            if step % check_interval == 0 and step > 0:
                average_loss /= check_interval
                if min_loss < average_loss:
                    tolerance -= 1
                else:
                    if tolerance < tolerance_margin:
                        tolerance += 1
                if min_loss > average_loss or min_loss == -1:
                    min_loss = average_loss
                print("[+] step: %d, %.2f steps/s, tol: %2d, epoch: %2d, avg.loss: %.5f, min.loss: %.5f"
                      % (step, float(check_interval) / (time.time() - timestamp),
                         tolerance, train_prod.current_epoch, average_loss, min_loss))
                timestamp = time.time()
                average_loss = 0
            step += 1
        if train_prod.current_epoch >= self._max_epochs:
            print("Amount of epochs reached")
        if tolerance <= 0:
            print("Tolerance margin reached")
        self._ready = True

    def pair(self, features):
        self._init()
        self._check_if_ready()
        features = np.array(features).reshape(self._batch_size, self._inputs)
        return self._pair_prediction.eval(session=self._session, feed_dict={self._train_features: features})

    def predict(self, features):
        self._init()
        self._check_if_ready()
        half_input = self._inputs/2
        features = np.array(features).reshape(self._batch_size, half_input)
        features = np.pad(features, ((0, 0), (0, half_input)), 'constant')
        return self._prediction.eval(session=self._session, feed_dict={self._train_features: features})

    @staticmethod
    def restore_definition(filename):
        params = json.load(open(filename + '.json', 'rb'))
        model = ContrastiveModel(params["name"], params["inputs"])
        return model

    def save_to_file(self, filename):
        Model.save_to_file(self, filename)
        json.dump({
            'name': self._name,
            'inputs': self._inputs
        }, open(filename+'.json', 'wb'))

    @staticmethod
    def get_producer(filename):
        return CSVBatchProducer(filename)


class CSVBatchProducer():
    def __init__(self, filename):
        self.current_epoch = 0
        self.max_index = 0
        self.labels = {'result': 0}
        self._reader = self._train_set_reader(filename)

    def produce(self, batch_size):
        """
            Produces full batch ready to be input in NN
        """
        labels = list()
        batch = list()
        while len(labels) < batch_size:
            label, features = self._reader.next()
            labels.append(label)
            batch.append(features)
        return batch, labels

    def _train_set_reader(self, filename):
        """
            Reads training set one by one
        """
        while True:
            with open(filename, 'rb') as reader:
                csvreader = csv.reader(reader, delimiter=',')
                for row in csvreader:
                    label = float(row[0])
                    if self.max_index == 0:
                        self.max_index = len(row) - 1
                    features = np.zeros(self.max_index, dtype=np.float32)
                    for idx in range(self.max_index):
                        features[idx] = float(row[idx+1])
                    yield label, features
            self.current_epoch += 1