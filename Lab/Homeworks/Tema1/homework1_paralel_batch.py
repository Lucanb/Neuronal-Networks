import pickle
import gzip
import numpy as np
import math
import random
from concurrent.futures import ThreadPoolExecutor

class ActivateFunctions:
  
    def __init__(self, array, count_array):
        self.array = array
        self.dim1 = count_array
  
    def identityActivate(self):
        return self.array
  
    def reluActivate(self):
        for index in range(len(self.array[0])):
            if self.array[0][index] < 0:
                self.array[0][index] = 0
        return self.array
        
    def Prelu(self, alpha): 
        for index in range(len(self.array[0])):
            if self.array[0][index] < 0:
                self.array[0][index] = alpha * self.array[0][index]
        return self.array
 
    def binaryStep(self):
        for index in range(len(self.array[0])):
            if self.array[0][index] <= 0:
                self.array[0][index] = 0
            else:
                self.array[0][index] = 1
        return self.array
        
    def SoftStep(self): 
        self.array = 1 / (1 + np.exp(-self.array))
        return self.array

  
    def SoftMax(self):
        e_x = np.exp(self.array[0] - np.max(self.array[0]))
        return e_x / e_x.sum(axis=0)
  
    def softPlus(self):
        for index in range(len(self.array[0])):
            self.array[0][index] = 1 + math.exp(self.array[0][index])
        return self.array
  
    def ELU(self, alpha):
        for index in range(len(self.array[0])):
            if self.array[0][index] < 0:
                self.array[0][index] = alpha * (math.exp(self.array[0][index]) - 1)
        return self.array
  
    def TanH(self):
        for index in range(len(self.array[0])):
            self.array[0][index] = 2 / (1 + math.exp(2 * (-self.array[0][index]))) - 1
        return self.array
       
    def ArcTan(self):    
        for index in range(len(self.array[0])):
            result = math.atan(self.array[0][index])
            self.array[0][index] = math.degrees(result)
        return self.array
   
class RandomDistribution:
  
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
  
    def uniformDistribution(self):
        return np.random.rand(self.input_size, self.output_size)
  
    def xavierDistrib(self):
        xavier_lim = math.sqrt(6.0 / (self.input_size + self.output_size))
        weight_vals = [[random.uniform(-xavier_lim, xavier_lim) for _ in range(self.output_size)] for _ in range(self.input_size)]
        return weight_vals
  
    def initialize_biases_sigmoid(self):
        biases = np.random.randn(self.output_size) * 0.01
        return biases
  
    def xavierDistrib_batch(self, batch_size):
        xavier_lim = math.sqrt(6.0 / (batch_size + self.output_size))
        weight_vals = [[random.uniform(-xavier_lim, xavier_lim) for _ in range(self.output_size)] for _ in range(batch_size)]
        return weight_vals
  
    def initialize_biases_sigmoid_batch(self, batch_size):
        biases = np.random.randn(self.output_size) * 0.01
        return biases
  
    def initPerm(self, size):
        array = list(range(size))
        return array

    def randomShuffle(self, array):
        random.shuffle(array)
        return array

class ClassifierDigits:
 
    def __init__(self, input_size, count_perceptrons, learning_rate):
        self.input_size = input_size
        output_size = 10
        self.count_perceptrons = count_perceptrons
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.randomV = RandomDistribution(input_size, output_size)
        self.mat = np.identity(10)
    
    def weightCalculus_batch(self, input_data, element):
        dotProduct = np.dot(input_data, self.weights) + self.bias
        activatedFunctions = ActivateFunctions(dotProduct, self.count_perceptrons).SoftStep()
        error = (self.mat[element] - activatedFunctions) * self.learning_rate

        # Calculăm errorMatrix
        errorMatrix = np.transpose(np.tile(input_data, (self.count_perceptrons, 1)))
        for i in range(self.count_perceptrons):
            errorMatrix[:, i] *= error[0][i]

        # Calculăm media pentru a obține valorile medii
        errorMatrix_mean = np.mean(errorMatrix, axis=0)

        # Actualizăm ponderile și bias-ul
        self.weights = self.weights + errorMatrix_mean
        self.bias = self.bias + (self.mat[element] - activatedFunctions) * self.learning_rate

    def train_batch(self, train_data, iterations, batch_size):
        randomPerm = self.randomV.initPerm(len(train_data[0]))
        it = 0

        with ThreadPoolExecutor() as executor:
            for _ in range(iterations):
                perm = self.randomV.randomShuffle(randomPerm)
                it += 1

                # Inițializează ponderile și bias-ul cu dimensiunile corecte pentru fiecare iterație
                self.initialize_weights()

                futures = []

                for i in range(len(train_data[0]) // batch_size):
                    start_index = i * batch_size
                    end_index = (i + 1) * batch_size
                    batch_data = (train_data[0][perm[start_index:end_index]], train_data[1][perm[start_index:end_index]])

                    # Apelăm metoda train_batch pentru fiecare batch în mod concurent
                    futures.append(executor.submit(self.weightCalculus_batch, batch_data[0], batch_data[1]))

                # Așteaptă terminarea tuturor execuțiilor concurente
                for future in futures:
                    future.result()

                print('Finished iteration ', it)
    
    # Adaugă o metodă pentru a inițializa ponderile și bias-ul la începutul antrenamentului
    def initialize_weights(self):
        self.weights = self.randomV.xavierDistrib()
        self.bias = self.randomV.initialize_biases_sigmoid_batch(batch_size)

    # Restul codului rămâne neschimbat

def read_data():
    fd = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(fd, encoding="latin")
    fd.close()
    return train_set, test_set

if __name__ == "__main__":
    print('Reading data ...')
    train_data, test_data =  read_data()
    train_x, train_y = train_data
    print('Data successfully loaded!')

    iterations = 30
    batch_size = 100
    MLP = ClassifierDigits(len(train_data[0][0]), 10, 0.001)
    print('Classifier function started iterations, ', iterations)
    MLP.train_batch(train_data, iterations, batch_size)

    trueClassif = sum([1 if (MLP.decision(test_data[0][i]) == test_data[1][i]) else 0 for i in range(len(test_data[0]))])

    print('Number of correct classifications with activation function: ', trueClassif, ' of ', len(test_data[0]))
    print('Success percent activation: ', 100.0 * trueClassif / len(test_data[0]), '%')
