import torch
import torch.nn as nn
import torch.optim as optim
import os,sys
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

class MyNetwork(nn.Module):
    def __init__(self, input_shape, actions):
        super(MyNetwork, self).__init__()
        self.actions = actions
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)

        self.action_values = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions)
        )

        self.state_value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def _get_conv_out(self, shape):
        with torch.no_grad():
            conv_out = self.conv(torch.zeros(1, *shape))
            return int(np.prod(conv_out.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        action_values = self.action_values(conv_out)
        state_value = self.state_value(conv_out).expand(x.size(0), self.actions)
        return state_value + action_values - action_values.mean(1).unsqueeze(1).expand(x.size(0), self.actions)