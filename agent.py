from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from model import Actor, Critic
import copy

BUFFER_SIZE = 100000  # replay buffer
BATCH_SIZE = 64       # batch size: fixed batch per pass
GAMMA = 0.9           # discount factor
TAU = 1e-4             # for soft update: Not update at once but frequently https://arxiv.org/pdf/1509.02971.pdf
lr_actor = 1e-4        # learning rate actor
lr_critic = 1e-4       # learning rate critic
WEIGHT_DECAY = 0       # L2 weight decay

LEARN_EVERY = 20       # learning timestep interval
learning_num = 10         # number of learning passes
GRAD_CLIPPING = 1.0    # gradient clipping 

# Ornstein-Uhlenbeck: Stochastic stationary Gauss-Markov process
ou_sigma = 0.2
ou_theta = 0.15

EPSILON = 1.0  
EPSILON_DECAY = 1e-6

#gpu if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self, state_size, action_size, seed=12):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.epsilon = EPSILON
        
        #This the actor network. Imported from model.py
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        
        #We will test the critic to see how good is the action. Imported from model.py
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=WEIGHT_DECAY)
        
        # From class UONoise
        self.noise = OUNoise(action_size, seed)
        
        # From class replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        
    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay buffer, and use sample from buffer to learn"""
        
        #from add function in Replay Buffer class: store experience
        self.memory.add(state, action, reward, next_state, done)
        
        # If having enough batch size
        if len(self.memory) > BATCH_SIZE and timestep % LEARN_EVERY == 0:
            for _ in range(learning_num):
                #sample data from sample function Replay Buffer class
                experiences = self.memory.sample()
                #from learn function
                self.learn(experiences, GAMMA)
                
    def act(self, state):
        """Returns actions for given state as per current policy"""
        #make state input
        state = torch.from_numpy(state).float().to(device)
        #in eval mode instead of trainning
        self.actor_local.eval()
        #do not save backprop but increase speed
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        #train actor
        self.actor_local.train()
        
        #from sample noise class
        action += self.epsilon * self.noise.sample()
        
        return np.clip(action, -1, 1)
    
    def reset(self):
        """ Reset from Noise to mean"""
        self.noise.reset()
        
    def learn(self, experiences, gamma):
        """
        Experience is a tuple of states, actions, rewards, next_states, dones
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        """
        
        states, actions, rewards, next_states, dones = experiences
        
        #use target models
        actions_next = self.actor_target(next_states)
        #to feed critic we use actor actions
        Q_targets_next = self.critic_target(next_states, actions_next)
        # compute Q targets
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # minimize critic the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # gradient clipping for critic
        if GRAD_CLIPPING > 0:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), GRAD_CLIPPING)
        #step updates the parameter
        self.critic_optimizer.step()
        
        # actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # update from soft_update function
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        # update epsilon decay
        if EPSILON_DECAY > 0:
            self.epsilon -= EPSILON_DECAY
            self.noise.reset()
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters. θ = τ*θ_local + (1 - τ)*θ_target"""  
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
class OUNoise:
    """Ornsten-Uhlenbeck"""
    
    def __init__(self, size, seed, mu=0., theta=ou_theta, sigma=ou_sigma):
        #array of ones
        self.mu = np.array(mu * size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()
        
    def reset(self):
        """Return a shallow copy of x"""
        self.state = copy.copy(self.mu)
        
    def sample(self):
        """Update internal state and return it as a noise sample"""
        state = self.state
        d_x = self.theta * (self.mu - state) + self.sigma * np.random.standard_normal(self.size)
        self.state = state + d_x
        return self.state
    
    
class ReplayBuffer:
    """Store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        #Deque (Doubly Ended Queue)
        self.memory = deque(maxlen=buffer_size) 
        #size of each training
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        """Add new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        """Random sample of batch from replay buffer"""
        experiences =  random.sample(self.memory, k=self.batch_size)
        
        #vstack: Stack arrays in sequence vertically (row wise)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """"Size of internal memory"""
        return len(self.memory)