import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class CnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, lr, fname, chkpt_dir = 'models'):
        super(CnnActorCriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, fname)
        os.makedirs(chkpt_dir, exist_ok=True)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_size[0], out_channels=32, kernel_size=5, stride=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )
        self.flattened_size = 64 * 14 * 14
        self.actor = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, output_size)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: torch.Tensor):
        features = self.features(state) 
        flattened = features.view(features.size(0), -1)

        policy_logits = self.actor(flattened)
        policy_dist = Categorical(logits=policy_logits) 
        
        value = self.critic(flattened)
        return policy_dist, value
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        else:
            print(f"Warning: Checkpoint file not found at {self.checkpoint_file}")


class PPOMemory:
    def __init__(self, batch_size):
        self.states = [] # Stores preprocessed torch.Tensors
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        stacked_states = torch.stack(self.states, dim=0) 

        return stacked_states,\
               np.array(self.actions),\
               np.array(self.probs),\
               np.array(self.vals),\
               np.array(self.rewards),\
               np.array(self.dones),\
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        # state is already a preprocessed torch.Tensor
        self.states.append(state) 
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []