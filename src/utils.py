import numpy as np
import matplotlib.pyplot as plt

def is_object_visible(env, obj_type, color=None):
        for i in range(env.grid.width):
            for j in range(env.grid.height):
                cell = env.grid.get(i, j)
                if cell and cell.type == obj_type and (color is None or cell.color == color):
                    return True
        return False

def get_cell_in_direction(env,direction):
    x, y = env.agent_pos
    dir_map = {
        'front': env.agent_dir,
        'right': (env.agent_dir + 1) % 4,
        'behind': (env.agent_dir + 2) % 4,
        'left': (env.agent_dir + 3) % 4
    }

    d = dir_map[direction]
    if d == 0: nx, ny = x + 1, y     # right
    elif d == 1: nx, ny = x, y + 1   # down
    elif d == 2: nx, ny = x - 1, y   # left
    else: nx, ny = x, y - 1          # up

    return env.grid.get(nx, ny)

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)