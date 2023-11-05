import numpy as np
import tensorflow as tf
import math

class ActivateFunctions:
  
  def __init__(self,array):
      self.array = array
      self.dim1 = array.shape[0]
      self.dim2 = array.shape[1]
  
  def identityActivate(self):
     return self.array
  
  def reluActivate(self):
    myarray = self.array.copy()
    myarray = myarray.flatten()  
    for element in myarray:
        if element < 0:
           element = 0
    return myarray.reshape(self.dim1,self.dim2)
        
  def Prelu(self,alpha): 
    myarray = self.array.copy()
    myarray = myarray.flatten()  
    for element in myarray:
        if element < 0:
           element = alpha*element
    return myarray.reshape(self.dim1,self.dim2)  
 
  def binaryStep(self):
    myarray = self.array.copy()
    myarray = myarray.flatten()  
    for element in myarray:
        if element <= 0:
            element = 0
        else:
            element =1
    return myarray.reshape(self.dim1,self.dim2)
        
  def SoftStep(self): 
    myarray = self.array.copy()
    myarray = myarray.flatten()  
    for element in myarray:
        element = 1/(1+math.exp(-element))
    return myarray.reshape(self.dim1,self.dim2)  
  
  def SoftMax(self):
    myarray = self.array.copy()
    myarray = myarray.flatten()  
    e_x = np.exp(myarray - np.max(myarray))
    return (e_x / e_x.sum(axis=0)).reshape(self.dim1,self.dim2)
  
  def softPlus(self):
    myarray = self.array.copy()
    myarray = myarray.flatten()  
    for element in myarray:
        element = 1+math.exp(element)
    return myarray.reshape(self.dim1,self.dim2) 
  
  def ELU(self,alpha):
    myarray = self.array.copy()
    myarray = myarray.flatten()  
    for element in myarray:
        if element < 0:
            element = alpha*(math.exp(element) - 1)

    return myarray.reshape(self.dim1,self.dim2)  
  
  def TanH(self):
    myarray = self.array.copy()
    myarray = myarray.flatten()  
    for element in myarray:
        element = 2/(1+math.exp(2*(-element)))-1
    return myarray.reshape(self.dim1,self.dim2)
       
  def ArcTan(self):    
    myarray = self.array.copy()
    myarray = myarray.flatten()  
    for element in myarray:
        result = math.atan(element)
        element = math.degrees(result)
    return myarray.reshape(self.dim1,self.dim2)
   
class RandomDistribution:
  
  def __init__(self,input_size,output_size):
      self.input_size = input_size
      self.output_size = output_size
  
  def uniformDistribution(self):
      return np.random.rand(self.input_size, self.output_size)
  
  def xavierDistribution(self):

    xavier_stddev = np.sqrt(1.0 / (self.input_size + self.output_size))
    weights = np.random.randn(self.input_size, self.output_size) * xavier_stddev + 0.5
    return weights
  
  def xavierTensor(self):
    in_dim, out_dim = self.input_size,self.output_size
    xavier_lim = tf.sqrt(6.)/tf.sqrt(tf.cast(in_dim + out_dim, tf.float32))
    weight_vals = tf.random.uniform(shape=(in_dim, out_dim), 
                                  minval=-xavier_lim, maxval=xavier_lim, seed=22)
    return weight_vals
  
  def initialize_biases_sigmoid(self):

    biases = np.random.randn(self.output_size) * 0.01  # You can adjust the scale as needed
    return np.abs(biases)


randomD = RandomDistribution(10,786)


print('THOSE ARE OUR DISTRIBUTIONS : \n')
print('xavierTensor',randomD.xavierTensor())
print('xavierDistr',randomD.xavierDistribution())
print('uniforDitrib',randomD.uniformDistribution())
print('initialises bieases',randomD.initialize_biases_sigmoid())
print('\n')

print('Those are our activation Functions :\n')

randomValues = RandomDistribution(1,20).xavierDistribution()
activateF = ActivateFunctions(randomValues)

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