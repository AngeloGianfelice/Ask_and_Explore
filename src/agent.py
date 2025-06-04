import random

class ANEAgent:
    def __init__(self, env, ppo, M=6, K=1, n=5, alpha=0.3):
        self.env = env
        self.ppo = ppo
        self.M = M
        self.K = K
        self.n = n
        self.alpha = alpha
        self.S = set()
        self.C = {}
        self.batch_data = []

    def initialize_questions(self):
        while len(self.S) < self.M:
            state = self.env.reset()
            desc = self.env.get_questions(state)
            for q in desc:
                if q not in self.S:
                    self.S.add(q)
                    self.C[q] = 0

    def generate_D(self):
        S_list = list(self.S)
        D = []
        for _ in range(self.K):
            qs = random.sample(S_list, self.n)
            for q in qs:
                self.S.remove(q)
            D.append(qs)
        return D

    def rollout(self, N=10, Nopt=4):
        self.initialize_questions()
        D = self.generate_D()

        for beta in range(N):
            state = self.env.reset()
            episode = []
            t = 0
            random.shuffle(D)
            for step in range(self.K):
                qs = D[step]
                answers_t = [self.env.answer_question(state, q) for q in qs]
                print(qs)
                print(answers_t)

                action, log_prob = self.ppo.get_action(state)
                print(action)
                next_state, re, done = self.env.step(action)
                answers_t1 = [self.env.answer_question(next_state, q) for q in qs]

                # Render here to see what's happening
                self.env.render()  # note: access the wrapped env's render()

                # Intrinsic reward calculation
                ri = 0
                for i, q in enumerate(qs):
                    if answers_t[i] != answers_t1[i]:
                        ri += 1
                        self.C[q] += 1
                        if self.C[q] / (beta + 1) >= self.alpha:
                            self.S.discard(q)
                            new_q = list(self.S)[0]
                            qs[i] = new_q
                            self.S.remove(new_q)

                episode.append((state, action, log_prob, ri + re))
                state = next_state

                if done:
                    break

            self.update_ppo(episode, Nopt)

    def update_ppo(self, episode, Nopt):
        states, actions, log_probs, rewards = zip(*episode)
        returns, advs = self.compute_advantages(rewards)

        for _ in range(Nopt):
            self.ppo.update((states, actions, log_probs, advs, advs, returns))

    ## ppo with entropy term and Generalized advantage estimation
    def compute_advantages(self, rewards, values, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0
        values = values + [0]  # Add bootstrap value
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return returns, advantages