import tensorflow as tf
import time


class Model():
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