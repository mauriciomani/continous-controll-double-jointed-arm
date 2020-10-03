import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Actor"""
    
    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """
        Using two fully connected layers with 400 and 300 units respectively. Not resetting parameters.
        """
        super(Actor, self).__init__()
        #use for 
        self.seed = torch.manual_seed(seed)
        #for fully connected layer input
        self.fc1 = nn.Linear(state_size, fc1_units)
        #Applying batch normalization
        self.bn1 = nn.BatchNorm1d(fc1_units)
        #fully connected layers
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 =  nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        """Actor policy network to map states to actions, using relus and tahn"""
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
    
class Critic(nn.Module):
    """Critic"""
    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Using same architecture as in Actor network. Two fully connected layers of 400 and 300 units respectively.
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        #in forward function action will be concatenated according to DDGP 
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        
    def forward(self, state, action):
        """Critic value network that maps (state,action) pairs to Q-values"""
        x = F.relu(self.bn1(self.fc1(state)))
        #to concatenate action
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)