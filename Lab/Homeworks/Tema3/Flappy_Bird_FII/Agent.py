import numpy as np
import collections
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2


KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

def preprocess_state(state, new_size=(84, 84)):
        """
        Prelucrarea stării (imagine RGB) pentru a fi compatibilă cu rețeaua neurală.

        Args:
        state (numpy.ndarray): Starea originală returnată de mediu, de obicei o imagine RGB.
        new_size (tuple): Dimensiunea dorită a imaginii (lățime, înălțime).

        Returns:
        torch.Tensor: Starea prelucrată ca tensor.
        """
        # Convertiți starea la imagine PIL și redimensionați
        image = Image.fromarray(state)
        image = image.resize(new_size)

        # Convertiți imaginea la scala de gri
        image = image.convert('L')

        # Convertiți imaginea la numpy array și normalizați
        processed_state = np.array(image) / 255.0

        # Adăugați dimensiuni suplimentare pentru batch și canale
        processed_state = np.expand_dims(processed_state, axis=0)
        processed_state = np.expand_dims(processed_state, axis=0)

        # Convertiți la tensor PyTorch
        return torch.tensor(processed_state, dtype=torch.float32)

class Agent:
    def __init__(self, env, buffer , device):
        self.env = env
        self.exp_buffer = buffer
        self.state = None
        self.total_rewards = 0
        self.actions = range(env.action_space.n)
        self.device = device  

    def reset(self):
        self.total_rewards = 0
        obs = self.env.reset()
        self.state = self.processFrame(obs)

    def processFrame(self,frame):
        frame = frame[0:400, 0:200]

        # plt.imshow(frame, cmap='gray')  # Afișarea imaginii în tonuri de gri
        # plt.show()

        # Salvarea imaginii în tonuri de gri
        # plt.imsave('frame1.png', frame, cmap='gray') 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
        # plt.imshow(frame, cmap='gray')  # Afișarea imaginii în tonuri de gri
        # plt.show()
        # plt.imsave('frame2.png', frame, cmap='gray')
        frame = cv2.filter2D(frame, -1, KERNEL)
        frame = frame.astype(np.float64) / 255.0
        return frame
    
    def choose_action(self, state, net, epsilon):
        ACTIONS = [0, 1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(state, tuple):
            state = state[0]

        if np.random.random() <= epsilon:
            action = np.random.choice(ACTIONS)
        else:
            state_v = preprocess_state(state)
            # plt.imshow(state, cmap='gray')
            # plt.show()
            # plt.imsave('frame3.png', state, cmap='gray')
            state_v = state_v.to(device)
            q_values = net(state_v)
            action = torch.argmax(q_values).item()

        return action
    
    def step(self, net, epsilon=0.1, device='cpu'):
        if np.random.random() <= epsilon:
            action = self.env.action_space.sample()
        else:
            state_v = torch.tensor(np.array([self.state], copy=False), dtype=torch.float32).to(device)
            q_values = net(state_v)
            action = int(torch.argmax(q_values))

        obs, reward, done, _ = self.env.step(action)
        self.total_rewards += reward

        new_state = self._process_frame(obs)
        self.exp_buffer.push((self.state, action, reward, new_state, done))

        self.state = new_state

        if done:
            self._reset()

        return self.total_rewards