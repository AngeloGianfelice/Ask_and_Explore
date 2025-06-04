from src.env import MiniGridEnvWrapper
from src.ppo import PPO
from src.agent import ANEAgent


if __name__ == '__main__':

    env = MiniGridEnvWrapper()
    ppo = PPO(state_dim=env.state_dim, action_dim=env.action_dim)
    agent = ANEAgent(env, ppo)
    agent.rollout(N=50, Nopt=5)