from My_Neural_Network.Zahlenerkennung import NeuralNetwork
import Data
import imageio
import numpy as np
from termcolor import colored

model = NeuralNetwork(lr=0.005, use_new_weights=False, save_weights=False)
model.train_runs(1)
model.confusion_matrix()
model.plot_acc_cost()

# test my 10 numbers from 0 to 9
for i in range(0, 10):
    image = imageio.imread(f"../Data/test_{i}.png")
    image = 255. - np.mean(image, axis=2).reshape(1, -1)

    predict = model.predict_number(image)
    if predict[0] == i:
        print(colored(predict, "green"))
    else:
        print(colored(predict, "red"))
