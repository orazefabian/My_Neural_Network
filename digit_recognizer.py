import gzip
import time

import numpy as np
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
import imageio
import pickle
from sklearn.metrics import confusion_matrix as cf
import matplotlib.pyplot as plt
from termcolor import colored


class NeuralNetwork(object):
    def __init__(self, lr=0.1, use_new_weights=False, save_weights=False):
        self.lr = lr
        self.use_new_weights = use_new_weights
        self.save_weights = save_weights
        self.final_pred = 0
        self.epochs = [0]
        self.costs = [0]
        self.accs = [0]

        self.X_train = self.open_images("../mnist/train-images-idx3-ubyte.gz").reshape(-1, 784)
        self.y_train = self.open_labels("../mnist/train-labels-idx1-ubyte.gz")

        oh = OneHotEncoder()
        self.y_train_oh = oh.fit_transform(self.y_train.reshape(-1, 1)).toarray()

        self.X_test = self.open_images("../mnist/t10k-images-idx3-ubyte.gz").reshape(-1, 784)
        self.y_test = self.open_labels("../mnist/t10k-labels-idx1-ubyte.gz")

        if use_new_weights == True:
            self.w0 = np.random.randn(300, 784)
            self.w1 = np.random.randn(50, 300)
            self.w2 = np.random.randn(10, 50)
        elif use_new_weights == False:
            with open("w0.p", "rb") as file:
                self.w0 = pickle.load(file)
            with open("w1.p", "rb") as file:
                self.w1 = pickle.load(file)
            with open("w2.p", "rb") as file:
                self.w2 = pickle.load(file)

    def open_images(self, filename):
        with gzip.open(filename, "rb") as file:
            data = file.read()
            return np.frombuffer(data, dtype=np.uint8, offset=16) \
                .reshape(-1, 28, 28) \
                .astype(np.float32)

    def open_labels(self, filename):
        with gzip.open(filename, "rb") as file:
            data = file.read()
            return np.frombuffer(data, dtype=np.uint8, offset=8)

    def activation(self, x):
        return expit(x)

    def train(self, X, y):
        a0 = self.activation(self.w0 @ X.T)
        a1 = self.activation(self.w1 @ a0)
        pred = self.activation(self.w2 @ a1)

        e2 = y.T - pred
        e1 = e2.T @ self.w2
        e0 = e1 @ self.w1

        dw2 = e2 * pred * (1 - pred) @ a1.T / len(X)
        dw1 = e1.T * a1 * (1 - a1) @ a0.T / len(X)
        dw0 = e0.T * a0 * (1 - a0) @ X / len(X)

        assert dw2.shape == self.w2.shape
        assert dw1.shape == self.w1.shape
        assert dw0.shape == self.w0.shape

        self.w2 = self.w2 + self.lr * dw2
        self.w1 = self.w1 + self.lr * dw1
        self.w0 = self.w0 + self.lr * dw0

    def predict(self, X):
        a0 = self.activation(self.w0 @ X.T)
        a1 = self.activation(self.w1 @ a0)
        pred = self.activation(self.w2 @ a1)
        return pred

    def cost(self, pred, y):
        # SUM((y - pred)^2)
        s = (1 / 2) * (y.T - pred) ** 2
        return np.mean(np.sum(s, axis=0))

    def train_runs(self, runs=50):

        sec_start = float(round(time.time() * 1000))
        cost_before = 1
        for i in range(0, runs):
            for j in range(0, 60000, 1000):
                images = self.X_train[j:(j + 1000), :] / 255.
                shift_x = np.random.randint(-3, 3)
                shift_y = np.random.randint(-3, 3)
                images = np.roll(images.reshape(-1, 28, 28), (shift_x, shift_y), axis=(1, 2)) \
                    .reshape(-1, 784)
                self.train(images, self.y_train_oh[j:(j + 1000), :])

            cost = self.cost(self.predict(self.X_train), self.y_train_oh)
            if cost < cost_before:
                print(colored("Kosten: " + str(cost), "green"))
            else:
                print(colored("Kosten: " + str(cost), "red"))
            cost_before = cost
            print(f"Left runs: {runs - i - 1}")

            y_test_pred = self.predict(self.X_test / 255.)
            y_test_pred = np.argmax(y_test_pred, axis=0)
            acc = np.mean(y_test_pred == self.y_test)

            self.epochs.append(len(self.epochs))
            self.costs.append(cost)
            self.accs.append(acc)

        print(f"{acc} ({round(acc * 100, 1)}% accuracy)")

        self.final_pred = y_test_pred

        sec_end = float(round(time.time() * 1000))
        sec_duration = float("{0:.2f}".format((sec_end - sec_start) / 1000 / 60))
        print(f"Time spent for training: {sec_duration} minutes")

        if self.save_weights:
            with open("w0.p", "wb") as file:
                pickle.dump(self.w0, file)
            with open("w1.p", "wb") as file:
                pickle.dump(self.w1, file)
            with open("w2.p", "wb") as file:
                pickle.dump(self.w2, file)

    def plot_acc_cost(self):
        plt.plot(self.epochs, self.costs, label="Kosten")
        plt.plot(self.epochs, self.accs, label="Genauigkeit")
        plt.legend()
        plt.show()

    def confusion_matrix(self):
        assert self.y_test.shape == self.final_pred.shape
        matrix = cf(self.y_test, self.final_pred)
        print(matrix)

    def error_print(self, num_exp, num_act):
        for i in range(0, len(self.X_test)):
            if self.final_pred[i] == num_exp and self.y_test[i] == num_act:
                plt.imshow(self.X_test[i].reshape(28, 28))
                plt.show()

    def predict_number(self, image):
        return np.argmax(self.predict(image), axis=0).reshape(-1)
