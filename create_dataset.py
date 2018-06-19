import ROOT
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np


def get_data():
    # Download dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # The data is loaded as flat array with 784 entries (28x28),
    # we need to reshape it into an array with shape:
    # (num_images, pixels_row, pixels_column, color channels)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Convert the uint8 PNG greyscale pixel values in range [0, 255]
    # to floats in range [0, 1]
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    # Convert digits to one-hot vectors, e.g.,
    # 2 -> [0 0 1 0 0 0 0 0 0 0]
    # 0 -> [1 0 0 0 0 0 0 0 0 0]
    # 9 -> [0 0 0 0 0 0 0 0 0 1]
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def create_tree(file_, x, y, name):
    file_.cd()

    # Setup tree and branches with arrays
    tree = ROOT.TTree(name, name)

    x_array = np.empty((28 * 28 * 1), dtype="float32")
    x_branch = tree.Branch("x", x_array, "x[{}]/F".format(28 * 28 * 1))

    y_array = np.empty((1), dtype="float32")
    y_branch = tree.Branch("y", y_array, "y/F")

    for x_, y_ in zip(x, y):
        # Reshape x_ to flat array
        x_ = x_.reshape(28 * 28 * 1)

        # Copy inputs and outputs to correct addresses
        x_array[:] = x_[:]
        y_array[:] = np.argmax(y_)

        # Fill tree
        tree.Fill()

    tree.Write()

    # Copy tree for different classes
    file_.mkdir(name + "_digits")
    file_.cd(name + "_digits")
    for i in range(10):
        new_tree = tree.CopyTree("y=={}".format(i))
        new_tree.SetName(name + "_digit{}".format(i))
        new_tree.Write()


if __name__ == "__main__":
    # Get dataset
    x_train, y_train, x_test, y_test = get_data()

    # Convert dataset to ROOT file
    file_ = ROOT.TFile("mnist.root", "RECREATE")
    create_tree(file_, x_train, y_train, "train")
    create_tree(file_, x_test, y_test, "test")
    file_.Write()
    file_.Close()
