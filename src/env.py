import gym
import numpy as np
from src.utils  import get_cell_in_direction,is_object_visible

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_


class MiniGridEnvWrapper:
    def __init__(self, env_name="MiniGrid-DoorKey-8x8-v0",render_mode="human"):
        self.env = gym.make(env_name)
        self.base_env = self.env.unwrapped  # gives direct access to grid, agent position, etc.

        self.action_space = self.env.action_space
        self.state_dim = np.prod(self.env.observation_space['image'].shape)
        self.action_dim = self.env.action_space.n

    def render(self):
        return self.env.render()

    def reset(self):
        obs, _ = self.env.reset()
        self.env.render()
        image_obs = obs['image']  # structured obs dict
        return image_obs.astype(np.float32).flatten() / 255.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        image_obs = obs['image']
        done = terminated or truncated
        return image_obs.astype(np.float32).flatten() / 255.0, reward, done

    def get_questions(self):
        questions = []
        colors = ['red', 'green']
        objects = ['key', 'ball']

        for color in colors:
            for obj in objects:
                questions.append(f"Is there a {color} {obj} visible?")

        directions = ['left', 'right', 'front', 'behind']
        for dir in directions:
            questions.append(f"Is there a wall to the {dir}?")

        questions.append("Is the agent holding an object?")
        questions.append("Is the object in front a door?")

        return questions


    def answer_question(self, state, question):
        env = self.base_env

        # Handle visibility questions
        if question.startswith("Is there a") and "visible" in question:
            parts = question.split()
            color = parts[3]
            obj = parts[4]
            return is_object_visible(env,obj, color)

        # Directional wall/object checks
        if question.startswith("Is there a wall to the"):
            direction = question.split()[-1][:-1] if question.endswith("?") else question.split()[-1]
            cell = get_cell_in_direction(env,direction)
            return cell is not None and cell.type == 'wall'

        if question.startswith("Is there a") and "to the" in question:
            parts = question.split()
            obj = parts[3]
            direction = parts[-1][:-1] if question.endswith("?") else parts[-1]
            cell = get_cell_in_direction(env,direction)
            return cell is not None and cell.type == obj

        if question == "Is the object in front a door?":
            cell = get_cell_in_direction(env,'front')
            return cell is not None and cell.type == 'door'

        if question == "Is the door in front open?":
            cell = get_cell_in_direction(env,'front')
            return cell is not None and cell.type == 'door' and cell.is_open

        if question == "Is the agent holding something?":
            return env.carrying is not None

        if question == "Is the agent in the top-left corner?":
            return env.agent_pos == (0, 0)

        return False  # default fallback

