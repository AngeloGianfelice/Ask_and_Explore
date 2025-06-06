import random
import torch
import numpy as np

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

class ANEAgent:
    def __init__(self, env, ppo, M=6, K=6, n=1, alpha=0.3, gamma=0.99, lambd=0.95):
        self.env = env
        self.ppo = ppo
        self.M = M
        self.K = K
        self.n = n
        self.alpha = alpha
        self.S = set()
        self.C = {}
        self.gamma = gamma
        self.lam=lambd
        self.batch_data = []

    def initialize_questions(self):
        # Use the env's internal logic to get all possible questions
        all_questions = self.env.get_questions()

        for q in all_questions:
            if q not in self.S:
                self.S.add(q)
                self.C[q] = 0

        # Make sure S has exactly M questions (init_qs)
        if len(self.S) > self.M:
            self.S = set(random.sample(list(self.S), self.M))


    def generate_D(self):
        D = []
        available_qs = list(self.S)

        for _ in range(self.K):  # K = rollout length
            # Sample n questions per step (qs_per_step)
            if len(available_qs) < self.n:
                # Refill from original S if needed
                available_qs = list(self.S)
            qs = random.sample(available_qs, self.n)
            D.append(qs)
        return D

    def rollout(self, N=128, Nopt=4):
        self.initialize_questions()
        D = self.generate_D()

        for beta in range(N):
            state = self.env.reset()
            self.env.render()
            episode = []
            random.shuffle(D)
            for step in range(self.K):
                qs = D[step]
                answers_t = [self.env.answer_question(state, q) for q in qs]
                #print(f"Question:{qs}")
                #print(f"answer:{answers_t}")

                action, log_prob = self.ppo.get_action(state)

                next_state, re, done = self.env.step(action)
                answers_t1 = [self.env.answer_question(next_state, q) for q in qs]
                
                # Intrinsic reward calculation
                ri = 0
                for i, q in enumerate(qs):
                    if answers_t[i] != answers_t1[i]:
                        ri += 1
                        ##self.C[q] += 1
                        ##if self.C[q] / (beta + 1) >= self.alpha:
                            ##self.S.discard(q)
                            ##new_q = list(self.S)[0]
                            ##qs[i] = new_q
                            ##self.S.remove(new_q)

                episode.append((state, action, log_prob, ri + re))
                state = next_state

                if done:
                    break
            self.update_ppo(episode, Nopt)
            print(f"episode: {beta}")

    def update_ppo(self, episode, Nopt):
        states, actions, log_probs, rewards = zip(*episode)
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.stack(log_probs)

        # Compute values using critic
        with torch.no_grad():
            values = self.ppo.value(states).squeeze()

        # Compute returns and advantages using GAE
        returns, advantages = self.compute_advantages(rewards, values)

        for _ in range(Nopt):
            self.ppo.update((states, actions, old_log_probs, advantages, returns))

    ## ppo with entropy term and Generalized advantage estimation
    def compute_advantages(self, rewards, values):
        advantages = []
        gae = 0
        values = values.view(-1)
        # Add bootstrap value: append 0 at the end
        values = torch.cat([values, torch.tensor([0.0])])

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages)
        returns = advantages + values[:-1]  # value targets

        return returns.detach(), advantages.detach()
    
    def test(self, episodes=10, render=True):
        self.ppo.policy.eval()  # Set policy network to eval mode
        env = self.env  # Assuming your AnE agent has self.env

        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    logits = self.ppo.policy(state_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()

                next_state, reward, done = env.step(action)
                total_reward += reward

                if render:
                    env.render()

                state = next_state

            print(f"Episode {episode + 1} total reward: {total_reward}")