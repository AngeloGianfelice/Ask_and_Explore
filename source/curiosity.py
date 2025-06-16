import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, IDX_TO_OBJECT, IDX_TO_COLOR
import random 

class GQACuriosityModule():
    def __init__(self, env, num_games, n, beta_start, beta_min, beta_decay):
        self.env = env
        self.n=n
        self.plateau = 0.5 * num_games
        self.beta_start = beta_start
        self.beta_min = beta_min
        self.beta_decay = beta_decay

        self.available_questions = [
            "What is in front of me?",
            "What is to my left?",
            "What is to my right?",
            "Can I move forward?",
            "Is the goal visible?"
        ]

        # dir_vec for standard MiniGrid agent_dir mapping (0:East, 1:South, 2:West, 3:North)
        self.dir_vec = [
            np.array([1, 0]),   # E (0) - MiniGrid default: Facing right
            np.array([0, 1]),   # S (1) - MiniGrid default: Facing down
            np.array([-1, 0]),  # W (2) - MiniGrid default: Facing left
            np.array([0, -1])   # N (3) - MiniGrid default: Facing up
        ]

    def get_relative_object_info(self, relative_offset: np.ndarray,
                                 grid_encoded: np.ndarray, agent_pos: np.ndarray,
                                 agent_dir: int,
                                 width: int, height: int):

        target_pos_absolute = None
        # The `relative_offset` here acts as a "flag" to determine the query type
        # It's not an actual coordinate offset relative to the agent's orientation.
        if np.array_equal(relative_offset, np.array([1,0])): # Asking for 'in front'
            target_pos_absolute = agent_pos + self.dir_vec[agent_dir]
        elif np.array_equal(relative_offset, np.array([0,1])): # Asking for 'right' (relative to agent's facing direction)
            target_pos_absolute = agent_pos + self.dir_vec[(agent_dir + 1) % 4]
        elif np.array_equal(relative_offset, np.array([0,-1])): # Asking for 'left' (relative to agent's facing direction)
            target_pos_absolute = agent_pos + self.dir_vec[(agent_dir - 1 + 4) % 4]
        else:
            return "Cannot interpret relative position."

        if not (0 <= target_pos_absolute[0] < width and 0 <= target_pos_absolute[1] < height):
            return "Out of bounds."

        obj_info_encoded = grid_encoded[target_pos_absolute[0], target_pos_absolute[1]]
        obj_type_idx, color_idx, state_idx = obj_info_encoded

        if obj_type_idx == OBJECT_TO_IDX['empty']:
            return "empty space"

        obj_desc = IDX_TO_OBJECT[obj_type_idx]
        obj_color = IDX_TO_COLOR[color_idx] if color_idx != -1 else ""

        if obj_color:
            obj_desc = f"{obj_color} {obj_desc}"

        return obj_desc

    def answer_question(self, question, grid_encoded, visible_grid, agent_pos,
                        agent_dir, width, height):

        question = question.lower().strip()

        if "is the goal visible" in question:
            for i in range(visible_grid.shape[0]):
                for j in range(visible_grid.shape[1]):
                    if visible_grid[i, j, 0] == OBJECT_TO_IDX['goal']:
                        return "Yes, the goal is visible."
            return "No, the goal is not visible."

        if "what is in front of me" in question or "what is in front" in question:
            return self.get_relative_object_info(np.array([1,0]), grid_encoded, agent_pos, agent_dir, width, height)

        if "what is to my left" in question:
            return self.get_relative_object_info(np.array([0,-1]), grid_encoded, agent_pos, agent_dir, width, height)

        if "what is to my right" in question:
            return self.get_relative_object_info(np.array([0,1]), grid_encoded, agent_pos, agent_dir, width, height)

        if "can i move forward" in question:
            next_pos = agent_pos + self.dir_vec[agent_dir]
            if not (0 <= next_pos[0] < width and 0 <= next_pos[1] < height):
                return "No, you cannot move forward (out of bounds)."

            if grid_encoded[next_pos[0], next_pos[1], 0] == OBJECT_TO_IDX['empty']:
                return "Yes, you can move forward."
            else:
                # Check if it's the goal - moving onto a goal is usually allowed
                if grid_encoded[next_pos[0], next_pos[1], 0] == OBJECT_TO_IDX['goal']:
                    return "Yes, you can move forward." # Can move into the goal square
                else:
                    return "No, there is an obstacle in front."

        return "I don't know the answer to that question."

    def get_intrinsic_reward(self,state_comp,new_state_comp,episode):
        answer_switch_counter = 0

        qs=random.sample(self.available_questions,k=self.n)
        for q in qs:
            state_answer=self.answer_question(q,
                                    state_comp['grid_encoded'],
                                    state_comp['visible_grid'],
                                    state_comp['agent_pos'],
                                    state_comp['agent_dir'],
                                    state_comp['width'],
                                    state_comp['height']
                                    )

            new_state_answer=self.answer_question(q,
                                    new_state_comp['grid_encoded'],
                                    new_state_comp['visible_grid'],
                                    new_state_comp['agent_pos'],
                                    new_state_comp['agent_dir'],
                                    new_state_comp['width'],
                                    new_state_comp['height']
                                    )

            if state_answer != new_state_answer:
                answer_switch_counter += 1

        #Beta exponential decay
        if episode < self.plateau:
            beta = self.beta_start
        else:
            beta = self.beta_start * np.exp(-self.beta_decay * (episode - self.plateau))
            return max(self.beta_min, beta)
    
        intrinsic_reward = beta * answer_switch_counter
        return intrinsic_reward
    
            
    def get_state_components(self):

        components = {
            'grid_encoded': self.env.unwrapped.grid.encode(),
            'visible_grid': self.env.unwrapped.gen_obs()['image'],
            'agent_pos': np.array(self.env.unwrapped.agent_pos),
            'agent_dir': self.env.unwrapped.agent_dir,
            'width': self.env.unwrapped.width,
            'height': self.env.unwrapped.height
        }

        return components