import cv2
import time
import pickle
import numpy as np

sigma = lambda z: 1 / (1 + np.exp(-z))
sigma_prime = lambda z: sigma(z) * (1 - sigma(z))
softmax = lambda z: np.exp(z) / sum(np.exp(z))

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.w = [np.zeros((0, 0))]
        self.b = [np.zeros(0)]
        for size1, size2 in zip(layer_sizes[1:], layer_sizes[:-1]):
            self.w.append(np.random.normal(0, 1 / np.sqrt(size2), (size1, size2)))
            self.b.append(np.random.normal(0, 1, size1))

    def feed_forward(self, x):
        z = [np.zeros(0)]
        a = [x]
        for l in range(1, len(self.w)):
            z.append(self.w[l] @ a[l - 1] + self.b[l])
            a.append((sigma if l < len(self.w) - 1 else softmax)(z[l]))
        return z, a

    def dropout_neurons(self):
        self.dropped_out_neurons = [[]]
        self.dropped_out_weights = []
        for l in range(1, len(self.w) - 1):
            drop = np.random.binomial(1, .5, len(self.w[l]))
            self.dropped_out_neurons.append([i for i, dropped in enumerate(drop) if dropped])
            for i in self.dropped_out_neurons[l]:
                for j in range(len(self.w[l][0])):
                    self.dropped_out_weights.append((l, i, j, self.w[l][i][j]))
                    self.w[l][i][j] = 0
                for j in range(len(self.w[l + 1])):
                    self.dropped_out_weights.append((l + 1, j, i, self.w[l + 1][j][i]))
                    self.w[l + 1][j][i] = 0

    def dropin_weights(self):
        for l, i, j, wlij in self.dropped_out_weights:
            self.w[l][i][j] = wlij

    def dropout_weights(self):
        self.dropped_out_weights = []
        for l in range(2, len(self.w)):
            drop = np.random.binomial(1, .5, len(self.w[l - 1]))
            dropped_out_neurons = [i for i, dropped in enumerate(drop) if dropped]
            for i in range(len(self.w[l])):
                for j in dropped_out_neurons:
                    self.dropped_out_weights.append((l, i, j, self.w[l][i][j]))
                    self.w[l][i][j] = 0

    def back_propagate(self, x, t):
        z, a = self.feed_forward(x)
        delta = [np.zeros(wl.shape) for wl in self.w]
        delta[-1] = a[-1] - t
        for l in range(len(self.w) - 2, 0, -1):
            delta[l] = sigma_prime(z[l]) * (delta[l + 1] @ self.w[l + 1])
            for i in self.dropped_out_neurons[l]:
                delta[l][i] = 0
        nabla_w = [np.zeros((0, 0))] + [delta[l].reshape(-1, 1) @ a[l - 1].reshape(1, -1) for l in range(1, len(self.w))]
        nabla_b = delta
        return nabla_w, nabla_b

    def gradient_descent(self, data_set, batch_size, learning_rate):
        np.random.shuffle(data_set)
        for i in range(0, len(data_set), batch_size):
            self.dropout_neurons()
            nabla_w_sum = [np.zeros(wl.shape) for wl in self.w]
            nabla_b_sum = [np.zeros(bl.shape) for bl in self.b]
            for x, t in data_set[i:(i + batch_size)]:
                nabla_w, nabla_b = self.back_propagate(x, t)
                nabla_w_sum = [nabla_wl_sum + nabla_wl for nabla_wl_sum, nabla_wl in zip(nabla_w_sum, nabla_w)]
                nabla_b_sum = [nabla_bl_sum + nabla_bl for nabla_bl_sum, nabla_bl in zip(nabla_b_sum, nabla_b)]
            self.w = [wl - learning_rate * (nabla_wl_sum / batch_size) for wl, nabla_wl_sum in zip(self.w, nabla_w_sum)]
            self.b = [bl - learning_rate * (nabla_bl_sum / batch_size) for bl, nabla_bl_sum in zip(self.b, nabla_b_sum)]
            self.dropin_weights()

class Classifier:
    def __init__(self, layer_sizes, data_sets, augment_data):
        self.network = NeuralNetwork(layer_sizes)
        self.training_set, self.validation_set, self.testing_set = data_sets
        self.training_set = sum([augment_data(entry) for entry in self.training_set], [])

    def classify(self, x):
        o = self.network.feed_forward(x)[1][-1]
        return max(range(len(self.network.w[-1])), key=(lambda i: o[i]))

    def train(self, batch_size, learning_rate):
        processed_data_set = [(x, [1 if i == t_class else 0 for i in range(len(self.network.w[-1]))]) for x, t_class in self.training_set]
        self.network.gradient_descent(processed_data_set, batch_size, learning_rate)

    def accuracy(self, data_set):
        self.network.dropout_weights()
        ok_count = 0
        for x, t_class in data_set:
            ok_count += self.classify(x) == t_class
        self.network.dropin_weights()
        return ok_count / len(data_set)

    def validation_accuracy(self):
        return self.accuracy(self.validation_set)

    def testing_accuracy(self):
        return self.accuracy(self.testing_set)

def rotate_image(image_2d, angle):
    image_center = tuple(np.array(image_2d.shape) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1)
    return cv2.warpAffine(image_2d, rotation_matrix, image_2d.shape, flags=cv2.INTER_LINEAR)

def image_rotations(entry):
    image_1d, digit = entry
    return [(rotate_image(image_1d.reshape(28, 28), angle).reshape(-1), digit) for angle in range(-15, 20, 5)]

data_sets = pickle.load(open('mnist.pkl', 'rb'), encoding='latin')
data_sets = tuple(list(zip(list(data_set[0]), list(data_set[1]))) for data_set in data_sets)

classifier = Classifier((784, 100, 10), data_sets, image_rotations)
for epoch in range(5):
    t1 = time.time()
    classifier.train(10, 1)
    t2 = time.time()
    print('epoch:', epoch)
    print('accuracy:', classifier.validation_accuracy())
    print('time:', t2 - t1, 'seconds')
    print('=' * 40)
print('final accuracy:', classifier.testing_accuracy())
