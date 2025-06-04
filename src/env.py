import gymnasium as gym
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper


class MiniGridEnvWrapper:
    def __init__(self, env_name="MiniGrid-DoorKey-8x8-v0"):
        self.env = gym.make(env_name)
        self.env = RGBImgPartialObsWrapper(self.env)
        self.env = ImgObsWrapper(self.env)

        self.action_space = self.env.action_space
        self.state_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = self.env.action_space.n

    def reset(self):
        obs = self.env.reset()
        return (obs[0].astype(np.float32) / 255.0).flatten()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return (obs[0].astype(np.float32) / 255.0).flatten(), reward, done

    def get_questions(self,state=None):
        # state param ignored since we use self.grid
        questions = [
            "Is there a red key visible?",
            "Is there a door in front?",
            "Is the agent holding a key?",
            "Is the door in front open?",
            "Is there a ball in top-left corner?",
            "Is the agent next to a wall on the left?",
        ]
        return questions

    def answer_question(self, state, question):
        # Use self.grid, self.agent_pos, self.agent_dir for answers
    
        def is_object_visible(obj_type, color=None):
            grid = self.env.grid
            for i in range(grid.width):
                for j in range(grid.height):
                    cell = grid.get(i, j)
                    if cell is not None and cell.type == obj_type:
                        if color is None or cell.color == color:
                            return True
            return False

        def get_cell_in_front():
            base_env = self.env.unwrapped  # unwrap to access agent_pos
            x, y = base_env.agent_pos
            dir = base_env.agent_dir
            # agent_dir: 0=right,1=down,2=left,3=up in MiniGrid
            if dir == 0: nx, ny = x + 1, y
            elif dir == 1: nx, ny = x, y + 1
            elif dir == 2: nx, ny = x - 1, y
            else: nx, ny = x, y - 1
            return self.env.grid.get(nx, ny)

        if question == "Is there a red key visible?":
            return is_object_visible('key', 'red')

        elif question == "Is there a door in front?":
            cell = get_cell_in_front()
            return cell is not None and cell.type == 'door'

        elif question == "Is the door in front open?":
            cell = get_cell_in_front()
            return cell is not None and cell.type == 'door' and cell.is_open

        elif question == "Is the agent holding a key?":
        # MiniGrid env stores inventory in env.carrying
            return self.env.carrying is not None and self.env.carrying.type == 'key'

        elif question == "Is there a ball in top-left corner?":
            cell = self.env.grid.get(0, 0)
            return cell is not None and cell.type == 'ball'

        elif question == "Is the agent next to a wall on the left?":
            # Calculate cell to the left of agent
            x, y = self.env.agent_pos
            dir = self.env.agent_dir
            if dir == 0: nx, ny = x, y - 1
            elif dir == 1: nx, ny = x - 1, y
            elif dir == 2: nx, ny = x, y + 1
            else: nx, ny = x + 1, y
            cell = self.env.grid.get(nx, ny)
            return cell is not None and cell.type == 'wall'

        else:
            # Unknown question
            return False

