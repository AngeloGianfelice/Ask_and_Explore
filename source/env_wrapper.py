import gymnasium as gym
from gymnasium import spaces

class Minigrid_Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.observation_space = self.env.observation_space['image']
        
        self.action_space = spaces.Discrete(3) 
        
        self.reward_range = [0, 1]

    def reset(self):
        obs_dict, info = self.env.reset()
        state_image = obs_dict['image']
        return state_image, info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action) 
        return obs_dict['image'], reward, terminated, truncated, info