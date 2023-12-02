import numpy as np
import math
import torch
import tensorflow as tf

class ActivateFunctions:
    def __init__(self, array, count_array):
        self.array = array
        self.dim1 = count_array

    def identityActivate(self):
        return torch.tensor(self.array, dtype=torch.float32)

    def derivative_identityActivate(self):
        return torch.ones_like(self.identityActivate())

    def reluActivate(self):
        return F.relu(self.identityActivate())

    def derivative_reluActivate(self):
        return torch.where(self.identityActivate() <= 0, torch.zeros_like(self.identityActivate()), torch.ones_like(self.identityActivate()))

    def preluActivate(self, alpha):
        return F.prelu(self.identityActivate(), torch.tensor(alpha, dtype=torch.float32))

    def derivative_preluActivate(self, alpha):
        return torch.where(self.identityActivate() <= 0, torch.tensor(alpha, dtype=torch.float32), torch.ones_like(self.identityActivate()))

    def binaryStep(self):
        return torch.where(self.identityActivate() <= 0, torch.zeros_like(self.identityActivate()), torch.ones_like(self.identityActivate()))

    def derivative_binaryStep(self):
        raise ValueError("Derivata pentru Binary Step nu este definită în 0")

    def softStep(self):
        return torch.sigmoid(self.identityActivate())

    def derivative_softStep(self):
        sigmoid_result = torch.sigmoid(self.identityActivate())
        return sigmoid_result * (1 - sigmoid_result)

    def softmax(self):
        return F.softmax(self.identityActivate(), dim=1)

    def derivative_softmax(self):
        raise NotImplementedError("Derivata pentru Softmax nu este implementată direct aici.")

    def softPlus(self):
        return F.softplus(self.identityActivate())

    def derivative_softPlus(self):
        return torch.sigmoid(self.identityActivate())

    def eluActivate(self, alpha):
        return F.elu(self.identityActivate(), alpha=torch.tensor(alpha, dtype=torch.float32))

    def derivative_eluActivate(self, alpha):
        return torch.where(self.identityActivate() <= 0, torch.tensor(alpha, dtype=torch.float32) * torch.exp(self.identityActivate()), torch.ones_like(self.identityActivate()))

    def tanhActivate(self):
        return torch.tanh(self.identityActivate())

    def derivative_tanhActivate(self):
        return 1 - torch.tanh(self.identityActivate()) ** 2

    def arcTanActivate(self):
      radians_result = torch.atan(self.identityActivate())
      degrees_result = radians_result * (180.0 / math.pi)
      return degrees_result


    def derivative_arcTanActivate(self):
        return 1 / (1 + self.identityActivate() ** 2)

class RandomDistribution:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def uniformDistribution(self):
        return torch.rand((self.input_size, self.output_size), dtype=torch.float32)

    def xavierDistrib(self):
        xavier_lim = math.sqrt(6.0 / (self.input_size + self.output_size))
        weight_vals = torch.rand((self.input_size, self.output_size), dtype=torch.float32) * 2 * xavier_lim - xavier_lim
        return weight_vals

    def initialize_biases_sigmoid(self):
        biases = torch.randn(self.output_size) * 0.01
        return biases

    def initPerm(self, size):
        array = list(range(size))
        return array

    def randomShuffle(self, array):
        shuffled_array = torch.randperm(len(array))
        return shuffled_array

  
class MLP_Neuronal_Network():
  
    def __init__(self):
  
        return self    

if __name__ == '__main__':
  
 print(torch.sigmoid(torch.tensor([1, 4, 5, 6], dtype=torch.float32)))