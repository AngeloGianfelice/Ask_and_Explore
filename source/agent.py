import numpy as np
import torch
from .ppo import CnnActorCriticNetwork,PPOMemory

class Agent:
    def __init__(self, n_actions,  model_fname, gamma= 0.99, lr = 0.0001, gae_lambda = 0.95,
                 policy_clip = 0.2, batch_size = 64, n_epochs = 4, input_dims = (3,56,56),
                 entropy_coef = 0.01):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef

        self.actorcritic = CnnActorCriticNetwork(input_dims, n_actions, lr, model_fname)
        self.memory = PPOMemory(batch_size)
        self.losses = []
        
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_model(self):
        print('... saving model ...')
        self.actorcritic.save_checkpoint()

    def load_model(self):
        print('... loading model ...')
        self.actorcritic.load_checkpoint()

    def preprocess_obs(self, obs, add_batch = True):
        # Convert HWC (height, width, channel) to CHW (channel, height, width)
        # and normalize pixel values to [0, 1]
        obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2,0,1) / 255.0
        if add_batch:
            obs_tensor = obs_tensor.unsqueeze(0) # Add batch dimension
        return obs_tensor.to(self.actorcritic.device) # Move to device for inference

    def choose_action(self, observation, deterministic = False):
        # Preprocess the observation here, adding batch dimension for network input
        state_tensor = self.preprocess_obs(observation, add_batch=True) 

        with torch.no_grad():
            dist, value = self.actorcritic(state_tensor)
            
            if deterministic:
                action = torch.argmax(dist.probs, dim=-1)
            else:
                action = dist.sample()

            probs = dist.log_prob(action)

        # Return scalar values from the batch of 1
        return action.item(), probs.item(), value.squeeze().item()

    def learn(self):
        self.actorcritic.train()

        state_arr, action_arr, old_prob_arr, vals_arr,\
        reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

        # Convert numpy arrays to tensors for GAE calculation
        actions = torch.tensor(action_arr, dtype=torch.long).to(self.actorcritic.device)
        old_probs = torch.tensor(old_prob_arr, dtype=torch.float).to(self.actorcritic.device)
        values = torch.tensor(vals_arr, dtype=torch.float).to(self.actorcritic.device)
        rewards = torch.tensor(reward_arr, dtype=torch.float).to(self.actorcritic.device)
        dones = torch.tensor(dones_arr, dtype=torch.bool).to(self.actorcritic.device)

        # Optimized GAE calculation (backward pass)
        advantage = torch.zeros_like(rewards).to(self.actorcritic.device)
        last_gae_lam = 0
        for t in reversed(range(len(reward_arr) - 1)):
            if dones[t]: 
                last_gae_lam = 0 
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t].float()) - values[t]
            advantage[t] = delta + self.gamma * self.gae_lambda * last_gae_lam

        #Normalize advantage 
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        for _ in range(self.n_epochs):
            for batch_indices in batches:
                # states_batch is already a torch.Tensor from generate_batches
                states_batch = state_arr[batch_indices].to(self.actorcritic.device)
                old_probs_batch = old_probs[batch_indices]
                actions_batch = actions[batch_indices]
                advantage_batch = advantage[batch_indices]
                values_batch = values[batch_indices] 


                dist, critic_value = self.actorcritic(states_batch)
                
                critic_value = critic_value.squeeze(-1)

                new_probs_batch = dist.log_prob(actions_batch)
                prob_ratio = torch.exp(new_probs_batch - old_probs_batch)

                weighted_probs = advantage_batch * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage_batch
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage_batch + values_batch 
                critic_loss = ((returns - critic_value)**2).mean()

                entropy_loss = dist.entropy().mean()

                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy_loss 
                self.losses.append(total_loss.item())

                self.actorcritic.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actorcritic.parameters(), max_norm=0.5)
                self.actorcritic.optimizer.step()
    
        self.memory.clear_memory() 

    def test(self, env, n_games=10):
        self.load_model()
        self.actorcritic.eval()
        scores=[]
        for i in range(n_games):
            state, info = env.reset()
            done = False
            score = 0
            while not done:
                env.render()  # render the environment
                with torch.no_grad():
                    action, prob, val = self.choose_action(state, deterministic=False)
                new_state, reward, terminated, truncated, info = env.step(action)
                done= terminated or truncated
                score += reward
                state = new_state 
            scores.append(score)   
            #print(f"Test episode {i}: score = {score}")
        print(f"{n_games} Test episodes: avg score = {np.mean(scores)}")