import random

from keras.datasets import cifar10
import matplotlib.pyplot as plt
from cs231n.classifiers import KNearestNeighbor
import numpy as np

# classes contained in the cifar10 dataset
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# x_train is a collection of 50,000 32x32 3 channel array images, for training
# y_train is a collection of 50,000 labels, associated with x_train images by index

# x_test is a collection of 10000 32x32 3 channel array images, for testing
# y_test is a collection of 10000 labels, associated with x_test images by index
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# shuffle all samples
x_train_new, y_train_new, x_test_new, y_test_new = [], [], [], []

train_order = list(range(len(x_train)))
test_order = list(range(len(x_test)))

random.shuffle(train_order)
random.shuffle(test_order)

for i in range(len(train_order)):
    x_train_new.append(x_train[train_order[i]])
    y_train_new.append(y_train[train_order[i]])

for i in range(len(test_order)):
    x_test_new.append(x_test[test_order[i]])
    y_test_new.append(y_test[test_order[i]])

x_test = x_test_new
y_test = y_test_new

# subsample the training data to allow for faster training
num_training = 2000
mask = list(range(num_training))
x_train = x_train[mask]
y_train = y_train[mask]

# subsample the testing data to allow for faster testing
num_testing = 20
mask = list(range(num_testing))

x_test = np.asarray([x_test[i] for i in mask])
y_test = np.asarray([y_test[i] for i in mask])

# train the knn classifier with the training data and labels
classifier = KNearestNeighbor()
classifier.train(x_train, y_train)

# testing implementation
dists = classifier.compute_distances_two_loops(x_test)
# plt.imshow(dists, interpolation="none")

classifier.predict_labels(dists, x_test)

# plt.show()