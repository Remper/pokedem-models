from __future__ import absolute_import
from __future__ import print_function

import argparse
import numpy as np
import time

from models.simple import SimpleModel
from models.contrastive import ContrastiveModel

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
parser.add_argument('--contrastive', default=False, action='store_true', help='Use contrastive model for training')
args = parser.parse_args()

args.batch_size = int(args.batch_size)
args.units = int(args.units)
args.layers = int(args.layers)
args.max_epochs = int(args.max_epochs)

print("Initialized with settings:")
print(vars(args))

# Restoring embeddings and preparing reader to produce batches
print("Initializing training set reader")
SelectedModel = SimpleModel
if args.contrastive:
    SelectedModel = ContrastiveModel
batch_producer = SelectedModel.get_producer(args.train)
print("Test batch:")
batch, labels = batch_producer.produce(2)
for idx, _ in enumerate(labels):
    print("  Features: ", batch[idx])
    print("  Label: ", labels[idx])
    print("")

model = SelectedModel("Model", batch_producer.max_index, len(batch_producer.labels))
model.units(args.units).layers(args.layers).batch_size(args.batch_size).max_epochs(args.max_epochs)
model.train(batch_producer)
model.save_to_file(args.output)

print("Done")
