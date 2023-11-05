import pickle, gzip, numpy as np
import matplotlib.pyplot as plt
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

    biases = np.random.randn(self.output_size) * 0.01
    return np.abs(biases)


class ClassifierDigits:
 
    def __init__(self,input_size,count_perceptrons,learning_rate):
        
        self.input_size = input_size
        output_size = 10
        self.count_perceptrons = count_perceptrons
        self.learning_rate = learning_rate
    
        randomV = RandomDistribution(input_size,output_size)

        #self.weights = randomV.xavierDistribution()
        self.weights = randomV.xavierTensor()
        self.bias = randomV.initialize_biases_sigmoid()
        self.mat =  np.identity(10)
    


    def weightCalculus(self,input_data,element):
        dotProduct = np.dot(input_data,self.weights) + self.bias
        activatedFunctions = ActivateFunctions(dotProduct).SoftStep()
        error = (self.mat[element] - activatedFunctions) * self.learning_rate


    
    def wightPreactivated():
        return
    
    def activateFunction(self,act,type):
        activate = ActivateFunctions()        
        return activate.type(self)

    def train(self,train_set,count):
        return
    

    def classify():
        return
    
def read_data():
    fd = gzip.open('mnist.pkl.gz','rb')
    train_set, valid_set, test_set = pickle.load(fd, encoding="latin")
    fd.close()

    return (train_set,test_set)

if __name__ == "__main__":
    print('Reading data ...')
    train_data,test_data =  read_data()
    train_x,train_y=train_data
    # print(train_x.shape)
    # print(train_y.shape)
    # plt.imshow(train_x[0].reshape(28, 28))
    # plt.show()
    print('Data successfully loaded!')

 # initialisation parameeters
    classifier_activate = ClassifierDigits()  

