import numpy as np
from copy import deepcopy


np.random.seed(42)


class ReLU:
    def forward(self, X):
        self.out = np.maximum(0, X)
        return self.out

    def backward(self, grad):
        return grad * (self.out > 0)


class Softmax:
    def forward(self, X):
        shifted = X - np.max(X, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.out = exp / np.sum(exp, axis=1, keepdims=True)
        return self.out

    def backward(self, grad):
        s = np.sum(grad * self.out, axis=1, keepdims=True)
        return self.out * (grad - s)


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def initialize(self, weights, bias):
        pass

    def update(self, weights, bias, dw, db):
        weights -= self.lr * dw
        bias -= self.lr * db


class Momentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.vdw = None
        self.vdb = None

    def initialize(self, weights, bias):
        self.vdw = np.zeros_like(weights)
        self.vdb = np.zeros_like(bias)

    def update(self, weights, bias, dw, db):
        self.vdw = self.beta * self.vdw + (1 - self.beta) * dw
        self.vdb = self.beta * self.vdb + (1 - self.beta) * db
        weights -= self.lr * self.vdw
        bias -= self.lr * self.vdb


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.mw = None
        self.mb = None
        self.vw = None
        self.vb = None

    def initialize(self, weights, bias):
        self.t = 0
        self.mw = np.zeros_like(weights)
        self.mb = np.zeros_like(bias)
        self.vw = np.zeros_like(weights)
        self.vb = np.zeros_like(bias)

    def update(self, weights, bias, dw, db):
        self.t += 1

        self.mw = self.beta1 * self.mw + (1 - self.beta1) * dw
        self.mb = self.beta1 * self.mb + (1 - self.beta1) * db
        self.vw = self.beta2 * self.vw + (1 - self.beta2) * dw**2
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * db**2

        mw_hat = self.mw / (1 - self.beta1**self.t)
        mb_hat = self.mb / (1 - self.beta1**self.t)
        vw_hat = self.vw / (1 - self.beta2**self.t)
        vb_hat = self.vb / (1 - self.beta2**self.t)

        weights -= self.lr * mw_hat / (np.sqrt(vw_hat) + self.eps)
        bias -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


class RMS:
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.sdw = None
        self.sdb = None

    def initialize(self, weights, bias):
        self.sdw = np.zeros_like(weights)
        self.sdb = np.zeros_like(bias)

    def update(self, weights, bias, dw, db):
        self.sdw = self.beta * self.sdw + (1 - self.beta) * dw**2
        self.sdb = self.beta * self.sdb + (1 - self.beta) * db**2

        weights -= self.lr * dw / (np.sqrt(self.sdw) + self.eps)
        bias -= self.lr * db / (np.sqrt(self.sdb) + self.eps)


class L1_Lasso:
    def __init__(self, l1_lambda=0.0):
        self.l1_lambda = l1_lambda


class L2_Ridge:
    def __init__(self, l2_lambda=0.0):
        self.l2_lambda = l2_lambda


class LayerDense:
    def __init__(self, input_shape, output_shape, optimizer=None):
        self.weights = np.random.randn(input_shape, output_shape) * np.sqrt(2 / input_shape)
        self.bias = np.zeros((1, output_shape))
        self.optimizer = optimizer if optimizer is not None else SGD()
        self.optimizer.initialize(self.weights, self.bias)
        self.X = None
        self.out = None
        self.dw = None
        self.db = None

    def forward(self, X):
        self.X = np.asarray(X)
        if self.X.ndim == 1:
            self.X = self.X.reshape(1, -1)
        self.out = self.X @ self.weights + self.bias
        return self.out

    def backward(self, dout):
        self.dw = self.X.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)
        return dout @ self.weights.T

    def step(self):
        self.optimizer.update(self.weights, self.bias, self.dw, self.db)


class Model:
    def __init__(self):
        self.layers = []
        self.optimizer = None
        self.loss = "categorical_crossentropy"
        self.metrics = ["accuracy"]
        self.regularization = None
        self.losses = []
        self.accuracies = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        if not self.layers:
            raise ValueError("No layers configured in the model.")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad, skip_last_softmax=False):
        layers = self.layers
        if skip_last_softmax and layers and isinstance(layers[-1], Softmax):
            layers = layers[:-1]
        for layer in reversed(layers):
            grad = layer.backward(grad)
        return grad

    def compile(self, optimizer, loss="categorical_crossentropy", metrics=None, regularization=None):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = ["accuracy"] if metrics is None else list(metrics)
        self.regularization = regularization

        for layer in self.layers:
            if isinstance(layer, LayerDense):
                layer.optimizer = deepcopy(optimizer)
                layer.optimizer.initialize(layer.weights, layer.bias)

    def get_penalty(self):
        total_penalty = 0.0

        if isinstance(self.regularization, L1_Lasso):
            strength = self.regularization.l1_lambda
            for layer in self.layers:
                if isinstance(layer, LayerDense):
                    total_penalty += strength * np.sum(np.abs(layer.weights))

        elif isinstance(self.regularization, L2_Ridge):
            strength = self.regularization.l2_lambda
            for layer in self.layers:
                if isinstance(layer, LayerDense):
                    total_penalty += strength * np.sum(layer.weights**2)

        return total_penalty

    def _regularization_gradients(self):
        if isinstance(self.regularization, L1_Lasso):
            strength = self.regularization.l1_lambda
            for layer in self.layers:
                if isinstance(layer, LayerDense):
                    layer.dw += strength * np.sign(layer.weights)

        elif isinstance(self.regularization, L2_Ridge):
            strength = self.regularization.l2_lambda
            for layer in self.layers:
                if isinstance(layer, LayerDense):
                    layer.dw += 2 * strength * layer.weights

    @staticmethod
    def _prepare_targets(y):
        y = np.asarray(y)
        if y.ndim == 1:
            return y.astype(int), False
        if y.ndim == 2:
            return np.argmax(y, axis=1), True
        raise ValueError("Targets must be class indices or one-hot encoded rows.")

    def _loss_and_gradient(self, predictions, y):
        if self.loss != "categorical_crossentropy":
            raise ValueError("Version 1.0 supports categorical_crossentropy only.")

        p = np.clip(predictions, 1e-15, 1 - 1e-15)
        labels, one_hot = self._prepare_targets(y)
        n = p.shape[0]

        if one_hot:
            if y.shape != p.shape:
                raise ValueError("One-hot targets must have the same shape as predictions.")
            loss = -np.mean(np.sum(y * np.log(p), axis=1))
            grad = (p - y) / n
        else:
            if np.any(labels < 0) or np.any(labels >= p.shape[1]):
                raise ValueError("Class indices are outside the prediction range.")
            loss = -np.mean(np.log(p[np.arange(n), labels]))
            grad = p.copy()
            grad[np.arange(n), labels] -= 1
            grad /= n

        return loss, grad, labels

    def fit(self, X_train, y_train, batch_size=1, lr=None, e=1000, shuffle=False, verbose=False):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        if X_train.ndim != 2:
            raise ValueError("X_train must have shape (samples, features).")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must contain the same number of samples.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if e < 1:
            raise ValueError("e must be at least 1.")
        if self.optimizer is None:
            self.compile(SGD() if lr is None else SGD(lr=lr))

        num_samples = X_train.shape[0]
        self.losses = []
        self.accuracies = []

        for epoch in range(e):
            if shuffle:
                perm = np.random.permutation(num_samples)
                X_epoch = X_train[perm]
                y_epoch = y_train[perm]
            else:
                X_epoch = X_train
                y_epoch = y_train

            total_loss = 0.0
            total_correct = 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                X_batch = X_epoch[start:end]
                y_batch = y_epoch[start:end]

                predictions = self.forward(X_batch)
                data_loss, grad, labels = self._loss_and_gradient(predictions, y_batch)
                batch_count = len(X_batch)
                total_loss += data_loss * batch_count
                total_correct += np.sum(np.argmax(predictions, axis=1) == labels)

                if not isinstance(self.layers[-1], Softmax):
                    raise ValueError(
                        "For categorical_crossentropy, the final layer must be Softmax."
                    )

                grad = self.backward(grad, skip_last_softmax=True)
                self._regularization_gradients()
                self.update()

            epoch_loss = total_loss / num_samples + self.get_penalty()
            epoch_accuracy = 100 * total_correct / num_samples
            self.losses.append(epoch_loss)
            self.accuracies.append(epoch_accuracy)

            if verbose and (epoch == 0 or (epoch + 1) % max(1, e // 10) == 0 or epoch == e - 1):
                print(
                    f"Epoch {epoch + 1}/{e} - "
                    f"loss: {epoch_loss:.6f} - accuracy: {epoch_accuracy:.2f}%"
                )

        return {"loss": self.losses, "accuracy": self.accuracies}

    def update(self):
        for layer in self.layers:
            if isinstance(layer, LayerDense):
                layer.step()

    def evaluate(self, X_test, y_test):
        predictions = self.forward(X_test)
        loss, _, labels = self._loss_and_gradient(predictions, y_test)
        accuracy = 100 * np.mean(np.argmax(predictions, axis=1) == labels)
        results = {"loss": loss + self.get_penalty()}

        if not self.metrics:
            return results
        if "accuracy" in self.metrics:
            results["accuracy"] = accuracy
        return results
