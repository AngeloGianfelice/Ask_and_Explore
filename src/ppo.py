import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.fc(x)
 
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, clip_epsilon=0.2):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr)
        self.clip_epsilon = clip_epsilon
    
    def get_action(self, state):
        logits = self.policy(torch.FloatTensor(state))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def update(self, batch):
        states, actions, old_log_probs, advantages, returns = batch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = old_log_probs.detach()
        advantages = torch.FloatTensor(advantages)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        returns = torch.FloatTensor(returns)

        new_logits = self.policy(states)
        dist = torch.distributions.Categorical(logits=new_logits)
        new_log_probs = dist.log_prob(actions)

        ratios = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

        #policy loss
        policy_loss = -torch.min(surr1, surr2).mean()

        #value loss
        value_loss = nn.MSELoss()(self.value(states).squeeze(), returns)
        
        #entropy term
        entropy = dist.entropy().mean()

        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy

        print(f"Total loss: {loss:.4f}, Policy: {policy_loss:.4f}, Value: {value_loss:.4f}, Entropy: {entropy:.4f}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    