import gymnasium as gym
from gym.wrappers import RecordVideo, RecordEpisodeStatistics
from minigrid.wrappers import RGBImgPartialObsWrapper
from datetime import datetime
import yaml
import argparse
from source.env_wrapper import Minigrid_Wrapper
from source.agent import Agent
from source.train_utils import train_ane_ppo,train_base_ppo
from source.utils import *

def main(mode,algorithm,n,config):

    set_seed(config['seed'])

    if mode == 'train':

        env = gym.make(config['env_name'],render_mode="rgb_array")
        env = RGBImgPartialObsWrapper(env)
        env = Minigrid_Wrapper(env)

        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")

        video_folder = f"./videos/run_{timestamp}_{algorithm}_{n}"

        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: e % config['save_video_rate'] == 0 and e != config['num_games'])
        env = RecordEpisodeStatistics(env)
        n_actions = env.action_space.n

        model_fname=f"ActorCritic_{algorithm}_{n}.pth"
        ext_fname = f'plots/ext_reward_{algorithm}_{n}.png'
        succ_fname= f'plots/success_probability_{algorithm}_{n}.png'

        agent = Agent(n_actions=n_actions, model_fname=model_fname, batch_size=config['batch_size'], 
                lr=config['lr'], n_epochs=config['n_epochs'], input_dims=config['input_dims'],
                entropy_coef=config['entropy_coef'],gamma=config['gamma'],
                gae_lambda=config['lambda'],policy_clip=config['policy_clip']
                )
        
        print(f"Starting training agent {algorithm} on {config['env_name']} with {n_actions} actions...")
        if algorithm == 'ane_ppo':
            results =  train_ane_ppo(env,agent,n=n,num_games=config['num_games'],beta_start=config['beta_start'],beta_min=config['beta_min'],beta_decay=config['beta_decay'])
            int_fname = f'plots/int_reward_{algorithm}_{n}.png'
            plot_metric(results['int_rew_history'], int_fname, xlabel='training episodes', ylabel='intrinsic reward',title='Ask & Explore Intrinsic Reward')
            plot_metric(results['ext_rew_history'], ext_fname, xlabel="training episodes", ylabel="extrinsic reward",title = 'Ask and Explore Extrinsic Reward')
            plot_metric(results['success_history'], succ_fname, xlabel="training episodes", ylabel="success probability",title="Ask & Explore Success Probability")
        else:
            results = train_base_ppo(env,agent,config['num_games'])
            plot_metric(results['ext_rew_history'], ext_fname, xlabel="training episodes", ylabel="extrinsic reward",title = 'Ask and Explore Extrinsic Reward')
            plot_metric(results['success_history'], succ_fname, xlabel="training episodes", ylabel="success probability",title="Ask & Explore Success Probability")

        print("Done!")
        env.close()

        test_env = gym.make(config['env_name'])
        test_env = RGBImgPartialObsWrapper(test_env)
        test_env = Minigrid_Wrapper(test_env)

        print(f"\n--- Testing agent {algorithm} on {config['env_name']} with {n_actions} actions...")
        print(f"Testing {algorithm} agent...")
        agent.test(test_env, n_games=50)
        print("Done!")
        test_env.close()

    else:
        env = gym.make(config['env_name'])
        env = RGBImgPartialObsWrapper(env)
        env = Minigrid_Wrapper(env)
        n_actions = env.action_space.n

        model_fname=f"ActorCritic_{algorithm}_{n}.pth"
        agent = Agent(n_actions=n_actions, model_fname=model_fname, batch_size=config['batch_size'], 
                lr=config['lr'], n_epochs=config['n_epochs'], input_dims=config['input_dims'],
                entropy_coef=0.01,gamma=config['gamma'],
                gae_lambda=config['lambda'],policy_clip=config['policy_clip']
                )

        print(f"\n--- Testing chosen agent on {config['env_name']} with {n_actions} actions...")
        print(f"Testing {algorithm} agent...")
        agent.test(env, n_games=50)
        print("Done!")
        env.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run PPO variant with config options.")

    parser.add_argument("mode", choices=["train", "test"], default='test', help="Mode: train or test")
    parser.add_argument("algorithm", choices=["base_ppo", "ane_ppo"], default='ane_ppo', help="Algorithm variant")
    parser.add_argument("n", type=int, default=1, help="K-hop value (int)")

    args = parser.parse_args()

    print(f"Mode: {args.mode}")
    print(f"Algorithm: {args.algorithm}")
    print(f"K-hop: {args.n}")

    #Configs initializzation
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    print("current config:",config)

    main(args.mode,args.algorithm,args.n,config)
