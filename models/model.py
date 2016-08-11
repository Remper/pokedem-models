import tensorflow as tf
import time


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

    def train(self, batch_producer):
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
        self._saver.save(self._session, filename)
        print("Done in %.2fs" % (time.time() - timestamp))

    @staticmethod
    def get_producer(filename):
        return SVMBatchProducer(filename)

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)


class SVMBatchProducer():
    def __init__(self, filename):
        self.current_epoch = 0
        print("Figuring out training set metadata")
        timestamp = time.time()
        self.max_index, self.labels, self.set_size = self._train_set_metadata(filename)
        print("Done in %.2fs" % (time.time() - timestamp))
        print("Features: %d. Classes: %d. Training set size: %d"
              % (self.max_index, len(self.labels), self.set_size))
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

    def _train_set_metadata(self, filename):
        """
            Figures out amount of labels and features
        """
        max_index = 0
        labels = {}
        set_size = 0
        with open(filename, 'rb') as reader:
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

    def _train_set_reader(self, filename):
        """
            Reads training set one by one
        """
        while True:
            with open(filename, 'rb') as reader:
                for line in reader:
                    row = line.split(' ')
                    label = np.zeros(len(self.labels), dtype=np.float32)
                    label[int(row[0])] = 1.0
                    features = np.zeros(self.max_index, dtype=np.float32)
                    for ele in row[1:]:
                        feature = ele.split(':')
                        features[int(feature[0])-1] = float(feature[1])
                    yield label, features
            self.current_epoch += 1