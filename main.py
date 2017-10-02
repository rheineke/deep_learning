import os
import struct
import time
from pathlib import Path

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import numpy as np

from neuralnet import NeuralNetMLP


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels


def read_np_mnist(path, npz_filename='mnist_scaled.npz'):
    filepath = path + npz_filename
    if not Path(filepath).exists():
        X_train, y_train = load_mnist(path, kind='train')
        print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

        X_test, y_test = load_mnist(path, kind='t10k')
        print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

        np.savez_compressed(
            filepath,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

    mnist = np.load(filepath)

    return [mnist[f] for f in ['X_train', 'y_train', 'X_test', 'y_test']]


if __name__ == '__main__':
    # mnist_sk = fetch_mldata('MNIST original')

    X_train, y_train, X_test, y_test = read_np_mnist('data/')

    n_epochs = 200

    nn = NeuralNetMLP(
        n_hidden=100,
        l2=0.01,
        epochs=n_epochs,
        eta=0.0005,
        minibatch_size=100,
        shuffle=True,
        seed=1
    )

    start_time = time.time()
    nn.fit(
        X_train=X_train[:55000],
        y_train=y_train[:55000],
        X_valid=X_train[55000:],
        y_valid=y_train[55000:]
    )
    print('Fit took {:.1}s'.format(time.time() - start_time))

    plt.plot(range(nn.epochs), nn.eval_['cost'])
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    plt.savefig('images/12_07.png', dpi=300)

    plt.plot(range(nn.epochs), nn.eval_['train_acc'], label='training')
    plt.plot(range(nn.epochs), nn.eval_['valid_acc'], label='validation',
             linestyle='--')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('images/12_08.png', dpi=300)

    y_test_pred = nn.predict(X_test)
    acc = (np.sum(y_test == y_test_pred).astype(np.float) / X_test.shape[0])

    print('Test accuracy: %.2f%%' % (acc * 100))

    miscl_img = X_test[y_test != y_test_pred][:25]
    correct_lab = y_test[y_test != y_test_pred][:25]
    miscl_lab = y_test_pred[y_test != y_test_pred][:25]

    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(25):
        img = miscl_img[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_title(
            '%d) t: %d p: %d' % (i + 1, correct_lab[i], miscl_lab[i]))

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.savefig('images/12_09.png', dpi=300)
