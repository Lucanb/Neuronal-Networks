import numpy as np
import tensorflow as tf
import math

class ActivateFunctions:
  def __init__(self, array, count_array):
    self.array = array
    self.dim1 = count_array

  def identityActivate(self):
    return self.array

  def derivative_identityActivate(self):
    return np.ones_like(self.array)

  def reluActivate(self):
    for index in range(len(self.array[0])):
      if self.array[0][index] < 0:
        self.array[0][index] = 0
    return self.array

  def derivative_reluActivate(self):
    return np.where(self.array <= 0, 0, 1)

  def preluActivate(self, alpha):
    for index in range(len(self.array[0])):
      if self.array[0][index] < 0:
        self.array[0][index] = alpha * self.array[0][index]
    return self.array

  def derivative_preluActivate(self, alpha):
    return np.where(self.array <= 0, alpha, 1)

  def binaryStep(self):
    for index in range(len(self.array[0])):
      if self.array[0][index] <= 0:
        self.array[0][index] = 0
      else:
        self.array[0][index] = 1
    return self.array

  def derivative_binaryStep(self):
    raise ValueError("Derivata pentru Binary Step nu este definită în 0")

  def softStep(self):
    for index in range(len(self.array[0])):
      self.array[0][index] = 1 / (1 + math.exp(-self.array[0][index]))
    return self.array

  def derivative_softStep(self):
    sigmoid_result = 1 / (1 + np.exp(-self.array))
    return sigmoid_result * (1 - sigmoid_result)

  def softmax(self):
    e_x = np.exp(self.array - np.max(self.array))
    return e_x / e_x.sum(axis=0)

  def derivative_softmax(self):
    raise NotImplementedError("Derivata pentru Softmax nu este implementată direct aici.")

  def softPlus(self):
    for index in range(len(self.array[0])):
      self.array[0][index] = 1 + math.exp(self.array[0][index])
    return self.array

  def derivative_softPlus(self):
    return 1 / (1 + np.exp(-self.array))

  def eluActivate(self, alpha):
    for index in range(len(self.array[0])):
      if self.array[0][index] < 0:
        self.array[0][index] = alpha * (math.exp(self.array[0][index]) - 1)
    return self.array

  def derivative_eluActivate(self, alpha):
        return np.where(self.array <= 0, alpha * np.exp(self.array), 1)

  def tanhActivate(self):
    for index in range(len(self.array[0])):
      self.array[0][index] = 2 / (1 + math.exp(2 * (-self.array[0][index]))) - 1
    return self.array

  def derivative_tanhActivate(self):
    return 1 - np.tanh(self.array) ** 2

  def arcTanActivate(self):
    for index in range(len(self.array[0])):
      result = math.atan(self.array[0][index])
      self.array[0][index] = math.degrees(result)
    return self.array

  def derivative_arcTanActivate(self):
    return 1 / (1 + self.array ** 2)
   
class RandomDistribution:
  
  def __init__(self,input_size,output_size):
    self.input_size = input_size
    self.output_size = output_size
  
  def uniformDistribution(self):
    return np.random.rand(self.input_size, self.output_size)
  
  def xavierDistrib(self):
    xavier_lim = math.sqrt(6.0 / (self.input_size + self.output_size))
    weight_vals = [[np.random.uniform(-xavier_lim, xavier_lim) for _ in range(self.output_size)] for _ in range(self.input_size)]
    return weight_vals
  
  def initialize_biases_sigmoid(self):

    biases = np.random.randn(self.output_size) * 0.01
    return biases
  
  def initPerm(self,size):
    array=[]
    for i in range(size):
      array.append(i)
    return array

  def randomShuffle(self,array):
    
    np.random.shuffle(array)
    return array


randomD = RandomDistribution(10,786)

def arrayT(array):
   
   for i in range(len(array[0])):
      array[0][i] =  array[0][i] * 2
   return array   

array = [[1,3,5,7]]
print(arrayT(array))

print('THOSE ARE OUR DISTRIBUTIONS : \n')
# print('xavierTensor',randomD.xavierTensor())
# print('xavierDistr',randomD.xavierDistribution())
print('uniforDitrib',randomD.uniformDistribution())
print('initialises bieases',randomD.initialize_biases_sigmoid())
print('\n')

print('Those are our activation Functions :\n')

randomValues = RandomDistribution(1,20).xavierDistrib()
activateF = ActivateFunctions(randomValues,len(randomValues))

print('identity : ',activateF.identityActivate())
print('ArcTan',activateF.ArcTan())
print('BinaryResult',activateF.binaryStep())
print('ELU :',activateF.ELU(1.2))
print('Prelu :',activateF.Prelu(2.4))
print('RELU :',activateF.reluActivate())
print('SoftMax :',activateF.SoftMax())
print('softPlus :',activateF.softPlus())
print('softStep :',activateF.SoftStep())
print('Tahn :',activateF.TanH())