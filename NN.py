import numpy as np
import scipy.special
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import random
import tensorflow

class neuralNetwork:

    # iniacjalizacja sieci neuronowej
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # ustawienie liczby neuronów w każdej z warstw
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes

        # ustawienie współczynnika uczenia
        self.lr = learningRate

        # losowanie wag
        self.wih = np.random.normal(0.0, pow(self.iNodes, -0.5), (self.hNodes, self.iNodes))
        self.who = np.random.normal(0.0, pow(self.hNodes, -0.5), (self.oNodes, self.hNodes))

        # funkcja aktywacji - sigmoid
        self.activation_fun = lambda x: scipy.special.expit(x)

    # metoda ucząca
    def train(self, input_list, targets_list):
        # zamiana list na macierz 2d
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # obliczanie sygnału do warstwy ukrytej
        hidden_inputs = np.dot(self.wih, inputs)
        # obliczanie sygnału wychodzącego z warstwy ukrytej
        hidden_outputs = self.activation_fun(hidden_inputs)

        # obliczanie sygnału do warstwy wyjściowej
        final_inputs = np.dot(self.who, hidden_outputs)
        # obliczanie sygnału wychodzącego z warstwy wyjściowej
        final_outputs = self.activation_fun(final_inputs)

        # oblicznie błędu warstwy wyjściowej (target - actual)
        output_err = targets - final_outputs
        # błąd warstwy ukrytej to błąd wyjściowy rozdzielony przez wartość wagi
        hidden_err = np.dot(self.who.T, output_err)

        # aktualizacja wag miedzy warstwą ukrytą, a wyjściem
        self.who += self.lr * np.dot((output_err * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        # aktualizacja wag miedzy wejściem, a warstwą ukrytą
        self.wih += self.lr * np.dot((hidden_err * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass

    # metoda odpowiadająca
    def querry(self, inputs_list):
        # zamiana listy wejść na macierz 2d
        inputs = np.array(inputs_list, ndmin=2).T

        # obliczanie sygnału do warstwy ukrytej
        hidden_inputs = np.dot(self.wih, inputs)
        # obliczanie sygnału wychodzącego z warstwy ukrytej
        hidden_outputs = self.activation_fun(hidden_inputs)

        # obliczanie sygnału do warstwy wyjściowej
        final_inputs = np.dot(self.who, hidden_outputs)
        # obliczanie sygnału wychodzącego z warstwy wyjściowej
        final_outputs = self.activation_fun(final_inputs)
        return final_outputs

input_list = [(0, 0), (1, 0), (1, 1), (0, 1)]
output_list = [(0), (1), (0), (1)]

x = []
y = []
input_nodes = 2
hidden_nodes = 4
output_nodes = 1
learning_rate = 0.02
nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
blad = 0
i = 0
start = timer()
while True:
    w = random.randint(0, 3)
    nn.train(input_list[w], output_list[w])
    if i % 1000 == 0:
        blad = 0
        for z in range(4):
            tmp = nn.querry(input_list[z]) - output_list[z]
            blad += tmp * tmp
        #             print("Actual:", nn.querry(input_list[z])," Target: ", output_list[z])
        blad /= 4
        x.append((i))
        y.append((float(blad)))

    if blad < 0.0005:
        print("Epoka: ", i, " Blad:", blad)
        break
    i += 1
print("Time: ", timer() - start)

for i in range(4):
    print("Actual:", nn.querry(input_list[i % 4]), " Target: ", output_list[i % 4])

plt.plot(x, y, label="Uczące")
plt.ylabel("Błąd %")
plt.xlabel("Liczba epok")
plt.legend()
plt.show()


