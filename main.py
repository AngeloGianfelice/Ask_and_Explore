from src.env import MiniGridEnvWrapper
from src.ppo import PPO
from src.agent import ANEAgent
import yaml


if __name__ == '__main__':

    #Configs initializzation
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # Assign global configs
    env_name = config['env_name']
    rollouts = config['rollouts']
    opt_steps = config['opt_steps']
    init_qs = config['init_qs']
    rollout_length = config['rollout_length']
    qs_per_step = config['qs_per_step']
    alpha = config['alpha']
    lr = config['lr']

    # Assign ppo variables
    gamma = config['ppo']['gamma']
    lambd = config['ppo']['lambda']
    clip_epsilon = config['ppo']['clip_epsilon']

    
    env = MiniGridEnvWrapper(env_name=env_name)
    ppo = PPO(state_dim=env.state_dim, action_dim=env.action_dim,lr=lr, clip_epsilon=clip_epsilon)
    agent = ANEAgent(env, ppo, M=init_qs, K=rollout_length, n=qs_per_step, gamma=gamma, lambd=lambd)
    agent.rollout(N=rollouts, Nopt=opt_steps)
    agent.test(episodes=5, render=True)

