from __future__ import absolute_import
from __future__ import print_function

import argparse
import numpy as np
import time

from models.simple import SimpleModel

"""
    Basic deep neural network that works with SVM input
"""

parser = argparse.ArgumentParser(description='Basic deep neural network that works with SVM input')
parser.add_argument('--train', default='', required=True, help='Location of the training set', metavar='#')
parser.add_argument('--output', default='model', help='Save model to', metavar='#')
parser.add_argument('--max_epochs', default=100, help='Maximum amount of epochs', metavar='#')
parser.add_argument('--layers', default=5, help='Amount of hidden layers', metavar='#')
parser.add_argument('--units', default=256, help='Amount of hidden units per layer', metavar='#')
parser.add_argument('--batch_size', default=256, help='Amount of samples in a batch', metavar='#')
args = parser.parse_args()

args.batch_size = int(args.batch_size)
args.units = int(args.units)
args.layers = int(args.layers)
args.max_epochs = int(args.max_epochs)

print("Initialized with settings:")
print(vars(args))


class BatchProducer():
    def __init__(self, filename):
        self.current_epoch = 0
        print("Figuring out training set metadata")
        timestamp = time.time()
        self.max_index, self.labels, self.set_size = self._train_set_metadata(filename)
        print("Done in %.2fs" % (time.time() - timestamp))
        print("Features: %d. Classes: %d. Training set size: %d (%.2f steps/epoch)"
              % (self.max_index, len(self.labels), self.set_size, float(self.set_size) / args.batch_size))
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


# Restoring embeddings and preparing reader to produce batches
print("Initializing training set reader")
batch_producer = BatchProducer(args.train)
print("Test batch:")
batch, labels = batch_producer.produce(2)
for idx, _ in enumerate(labels):
    print("  Features: ", batch[idx])
    print("  Label: ", labels[idx])
    print("")

model = SimpleModel("Model", batch_producer.max_index, len(batch_producer.labels))
model.units(args.units).layers(args.layers).batch_size(args.batch_size).max_epochs(args.max_epochs)
model.train(batch_producer)
model.save_to_file(args.output)

print("Done")
