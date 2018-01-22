from sklearn.utils import shuffle

import tensorflow as tf
import time
import numpy as np


class Model:
    def __init__(self, name):
        self._name = name
        self._graph = None
        self._session = None
        self._saver = None
        self._ready = False

    def _definition(self):
        raise NotImplementedError("Definition function must be implemented for the model")

    def _init(self):
        if self._session is not None:
            return

        self._graph = self._definition()
        if self._saver is None:
            raise NotImplementedError("Definition wasn't properly implemented: missing saver")
        self._session = tf.Session(graph=self._graph)

    def train(self, train_prod, eval_prod=None):
        raise NotImplementedError("Training function has to be defined")

    def predict(self, features):
        raise NotImplementedError("Predict function has to be defined")

    def restore_from_file(self, filename):
        self._init()
        self._saver.restore(self._session, filename)
        self._ready = True

    def _check_if_ready(self):
        if not self._ready:
            raise ValueError("Model isn't ready for prediction yet, train it or restore from file first")

    def save_to_file(self, filename):
        self._check_if_ready()
        print("Saving model")
        timestamp = time.time()
        self._saver.save(sess=self._session, save_path=filename+'.cpkt')
        print("Done in %.2fs" % (time.time() - timestamp))

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


class BatchProducer:
    def __init__(self, filename):
        self.max_index = 0
        self.labels = {}
        self.set_size = 0
        self.filename = filename

        print("Figuring out dataset metadata")
        timestamp = time.time()
        self.max_index, self.labels, self.set_size = self._get_dataset_metadata(filename)
        print("Done in %.2fs" % (time.time() - timestamp))
        self._print_stats()

    def produce(self, batch_size: int) -> (np.ndarray, np.ndarray, int):
        """
            Produces full batch ready to be input in NN
        """
        pass

    def _print_stats(self):
        print("Features: %d. Classes: %d. Training set size: %d"
              % (self.max_index, len(self.labels), self.set_size))

    @staticmethod
    def _get_dataset_metadata(filename):
        """
            Figures out amount of labels and features
        """
        max_index = 0
        labels = {}
        set_size = 0
        with open(filename, 'r') as reader:
            for line in reader:
                set_size += 1
                row = line.split(' ')
                if row[0] not in labels:
                    labels[row[0]] = len(labels)
                for ele in row[1:]:
                    index = int(ele.split(':')[0])
                    if index == 0:
                        raise ValueError("This code doesn't support zero-based SVM features")
                    if index > max_index:
                        max_index = index
        return max_index, labels, set_size


class SVMBatchProducer(BatchProducer):
    def __init__(self, filename):
        super().__init__(filename)

    def produce(self, batch_size: int) -> (np.ndarray, np.ndarray, int):
        """
            Produces full batch ready to be input in NN
        """
        labels = list()
        batch = list()
        reader = self._train_set_reader()
        cur_sample = 0
        while True:
            try:
                while len(labels) < batch_size:
                    label, features = reader.__next__()
                    labels.append(label)
                    batch.append(features)
                yield np.vstack(batch), np.vstack(labels), cur_sample
                cur_sample += len(batch)
                labels = list()
                batch = list()
            except:
                break
        return np.vstack(batch), np.vstack(labels), cur_sample

    def _train_set_reader(self) -> (np.ndarray, np.ndarray):
        """
            Reads training set one sample at the time
        """
        with open(self.filename, 'r') as reader:
            for line in reader:
                row = line.split(' ')
                label = np.zeros(len(self.labels), dtype=np.float32)
                label[int(row[0])] = 1.0
                features = np.zeros(self.max_index, dtype=np.float32)
                for ele in row[1:]:
                    feature = ele.split(':')
                    features[int(feature[0])-1] = float(feature[1])
                yield label, features


class PreloadedSVMBatchProducer(BatchProducer):
    def __init__(self, filename):
        super().__init__(filename)

        print("Preloading dataset")
        timestamp = time.time()
        self.X, self.Y = self._load_dataset()
        print("Done in %.2fs" % (time.time() - timestamp))

    def produce(self, batch_size: int) -> (np.ndarray, np.ndarray, int):
        """
            Produces full batch ready to be input in NN
        """

        X, Y = shuffle(self.X, self.Y)
        size = Y.shape[0]

        pointer = 0
        while pointer + batch_size < size:
            yield X[pointer:pointer + batch_size], Y[pointer:pointer + batch_size], pointer
            pointer += batch_size
        yield X[pointer:], Y[pointer:], pointer

    def _load_dataset(self):
        """
            Reads training set one by one
        """
        X = []
        Y = []
        X_batch = []
        Y_batch = []
        cutoff = 100000
        with open(self.filename, 'r') as reader:
            for line in reader:
                row = line.split(' ')
                labels = np.zeros(len(self.labels), dtype=np.float32)
                labels[int(row[0])] = 1.0
                features = np.zeros(self.max_index, dtype=np.float32)
                for ele in row[1:]:
                    feature = ele.split(':')
                    features[int(feature[0])-1] = float(feature[1])

                X_batch.append(features)
                Y_batch.append(labels)

                if len(X_batch) >= cutoff:
                    X.append(np.vstack(X_batch))
                    Y.append(np.vstack(Y_batch))
                    X_batch = []
                    Y_batch = []
        return np.vstack(X), np.vstack(Y)