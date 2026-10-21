import struct

import numpy as np
import pytest

from main import load_mnist, read_np_mnist
from neuralnet import NeuralNetMLP


def blobs(n_per_class=60, seed=0):
    """Three well separated clusters in 4 dimensions."""
    rng = np.random.RandomState(seed)
    centers = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0]], dtype=float)
    X = np.vstack([c + 0.3 * rng.randn(n_per_class, 4) for c in centers])
    y = np.repeat(np.arange(3), n_per_class)
    return X, y


def test_onehot():
    onehot = NeuralNetMLP()._onehot(np.array([0, 2, 1]), 3)
    np.testing.assert_array_equal(onehot, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])


def test_sigmoid_is_bounded_and_centered():
    s = NeuralNetMLP()._sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert s[1] == 0.5
    assert 0 < s[0] < 1e-100 and 1 - 1e-12 < s[2] <= 1


def test_fit_learns_separable_classes():
    X, y = blobs()
    nn = NeuralNetMLP(n_hidden=10, epochs=50, eta=0.05, minibatch_size=10, seed=1)
    nn.fit(X_train=X, y_train=y, X_valid=X[::5], y_valid=y[::5])
    assert nn.eval_["cost"][-1] < nn.eval_["cost"][0]
    assert nn.eval_["train_acc"][-1] == pytest.approx(1.0)
    np.testing.assert_array_equal(nn.predict(X), y)


def test_fit_is_reproducible_with_seed():
    X, y = blobs()
    costs = []
    for _ in range(2):
        nn = NeuralNetMLP(n_hidden=5, epochs=3, minibatch_size=10, seed=7)
        costs.append(nn.fit(X, y, X, y).eval_["cost"])
    assert costs[0] == costs[1]


def write_idx(path, kind, images, labels):
    with open(path / f"{kind}-labels-idx1-ubyte", "wb") as fp:
        fp.write(struct.pack(">II", 2049, len(labels)) + bytes(labels))
    with open(path / f"{kind}-images-idx3-ubyte", "wb") as fp:
        fp.write(struct.pack(">IIII", 2051, len(labels), 28, 28))
        fp.write(images.astype(np.uint8).tobytes())


def test_load_mnist_scales_pixels(tmp_path):
    images = np.zeros((2, 784), dtype=np.uint8)
    images[1] = 255
    write_idx(tmp_path, "train", images, [3, 7])
    X, y = load_mnist(tmp_path, kind="train")
    assert X.shape == (2, 784)
    np.testing.assert_array_equal(y, [3, 7])
    assert X[0].max() == -1.0 and X[1].min() == 1.0


def test_read_np_mnist_caches_npz(tmp_path):
    images = np.arange(3 * 784).reshape(3, 784) % 256
    for kind in ("train", "t10k"):
        write_idx(tmp_path, kind, images, [0, 1, 2])
    first = read_np_mnist(f"{tmp_path}/")
    assert (tmp_path / "mnist_scaled.npz").exists()
    (tmp_path / "train-images-idx3-ubyte").unlink()  # now only the cache remains
    second = read_np_mnist(f"{tmp_path}/")
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
