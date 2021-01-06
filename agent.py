import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from model import QNetwork
from replay_buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    
    def __init__(self, state_size, action_size, buffer_size=int(1e5), batch_size=64, learning_rate=5e-4, gamma=0.99, tau=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        
        self.q_network = QNetwork(self.state_size, self.action_size)
        self.q_network_target = QNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        
    def act(self, state, epsilon=0.0, verbose=0):
        """
        Given a state predict a new action.
        """
        # predict the action values with the q-network
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()
        
        # decide for action accoring to epsilon-greedy method
        if random.random() > epsilon:
            a = np.argmax(action_values.cpu().data.numpy())
            if verbose>0:
                print(f"\rChoosen action: {a} (greedy action)", end="")
        else:
            a = random.choice(np.arange(self.action_size))
            if verbose>0:
                print(f"\rChoosen action: {a} (random action)", end="")
        return a 
        

    def store_experience(self, state, action, reward, next_state, done, td_error=np.nan):
        """
        Add an experience to learn from.
        """
        self.replay_buffer.add(state, action, reward, next_state, done, td_error=td_error)

        
    def learn(self):
        """
        Learn from stored experiences, by sampling one batch from the replay buffer. Generate 
        the targets for the Q-values by taking the argmax from the target Q-network and train on them.
        After one ackpropagation step update the weights accorinding to the value tau using the following
        update rule:
            target_weights = tau * local_weights + (1-tau) * target_weights
        """
        if len(self.replay_buffer) < self.batch_size:
            return 

        # sample experiences
        states, actions, rewards, next_states, dones, sampling_probs = self.replay_buffer.sample(self.batch_size)

        # calculate the target q_values          
        next_states = torch.from_numpy(next_states).float().to(device)
        Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)

        rewards = torch.from_numpy(rewards).float().to(device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(device)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # get prediction of Q values from local model
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        Q_expected = self.q_network(states).gather(1, actions)

        # compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update weights
        for target_param, local_param in zip(self.q_network_target.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
            
            
    def save_q_network(self, path):
        torch.save(self.q_network.state_dict(), path)
        
        
    def load_q_network(self, path):
        state_dict = torch.load(path)
        self.q_network.load_state_dict(state_dict)
        self.q_network_target.load_state_dict(state_dict)
        