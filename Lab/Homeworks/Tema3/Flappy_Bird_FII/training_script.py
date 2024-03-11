import gymnasium as gym
import flappy_bird_gymnasium
import torch
import torch.optim as optim
import torch.nn as nn
from Network import MyNetwork  # Presupunând că ai clasa MyNetwork definită în MyNetwork.py
from Agent import Agent  # Presupunând că ai clasa Agent definită în Agent.py
import random
import numpy as np

ACTIONS = [0, 1]  # Acțiuni disponibile în Flappy Bird
GAMMA = 0.99  # Factorul de discount
EPSILON_START = 0.1  # Valoarea inițială a epsilonului pentru explorare
EPSILON_FINAL = 0.0001  # Valoarea finală a epsilonului
num_episodes = 50000
EPSILON_DECAY_FRAMES = 0.1/num_episodes  # Numărul de pași pentru scăderea epsilonului
LEARNING_RATE = 1e-4 
BATCH_SIZE = 32
BUFFER_CAPACITY = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer: #de imbunatatit cu ce vr reateaua sa invete : sa adaug daca a ajuns la maxim si daca nu sa scot ultimul (coada) --> mereu le scot pe ultimele(flush)
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # Elimină cea mai veche experiență
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, dones, next_states = zip(*batch)

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.uint8)
        next_states = np.array(next_states, dtype=np.float32)

        return states, actions, rewards, dones, next_states

    def __len__(self):
        return len(self.buffer)

env = gym.make("FlappyBird-v0", render_mode="human")
net = MyNetwork(input_shape=(1, 84, 84), actions=len(ACTIONS)).to(device)
tgt_net = MyNetwork(input_shape=(1, 84, 84), actions=len(ACTIONS)).to(device)
agent = Agent(env, None, device)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
buffer = ReplayBuffer(BUFFER_CAPACITY)

def calc_loss(batch, net, tgt_net, device='cpu'):
    states, actions, rewards, dones, next_states = batch
    
    states_v = torch.tensor(states, dtype=torch.float32).to(device)
    actions_v = torch.tensor(actions, dtype=torch.long).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    dones_v = torch.ByteTensor(dones).to(device)
    next_states_v = torch.tensor(next_states, dtype=torch.float32).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_action_values = tgt_net(next_states_v).max(1)[0]
    next_state_action_values[dones_v] = 0.0
    next_state_action_values = next_state_action_values.detach()

    expected_values = rewards_v + next_state_action_values * GAMMA # asta trb sa fie mare si printat 
    return nn.MSELoss()(state_action_values, expected_values)


epsilon = EPSILON_START
SCREEN_SIZE = (288, 512)
threshold = SCREEN_SIZE[1]

def process_state(state):
    return agent.processFrame(state)
    

for episode in range(num_episodes):
    state = env.reset()
    state = env.render()
    # Presupunând că 'state' este un tuple și trebuie prelucrat
    state = process_state(state)
    total_reward = 0
    done = False


    while not done:
        action = agent.choose_action(state, net, epsilon)
        next_state, reward, done, _, info = env.step(action)
        next_state = agent.processFrame(env.render())
# aici :
        if done:
            reward = -1  # Recompensă negativă pentru stări terminale

        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
          
        if len(buffer) > BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            optimizer.zero_grad()
            loss = calc_loss(batch, net, tgt_net)
            loss.backward()
            optimizer.step()

        if not done:
            state_v = torch.tensor([state], dtype=torch.float32).to(device)
            current_q = net(state_v).detach().cpu().numpy()[0, action]
            print(f"Valoare Q pentru starea curentă și acțiunea {action}: {current_q}")

        epsilon = max(EPSILON_FINAL, epsilon - (EPSILON_START - EPSILON_FINAL) / EPSILON_DECAY_FRAMES)

    print(f"Episod: {episode}, Recompensa totală: {total_reward}")

env.close()