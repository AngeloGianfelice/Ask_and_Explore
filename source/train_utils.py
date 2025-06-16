import numpy as np
import csv
from .curiosity import GQACuriosityModule

def train_ane_ppo(env,agent,n,num_games,beta_start,beta_min,beta_decay):

    logs_file='./logs/train_ane_ppo.csv'
    # Write CSV header
    with open(logs_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_num", "ep_ext_reward", "ep_int_reward", "ep_success", "ep_steps"])

    rollout_length = 1024
    
    curiosity_module = GQACuriosityModule(env=env, num_games=num_games, n=n, 
                                          beta_start=beta_start, beta_min=beta_min, beta_decay=beta_decay
                                          )

    best_score = env.reward_range[0]

    int_rew_history = []
    ext_rew_history = []
    success_history = []

    learn_iters = 0
    avg_ext_reward = 0
    n_steps = 0
    state, info = env.reset()
    for i in range(num_games):
        done = False
        ep_int_reward = 0
        while not done:
            state_components = curiosity_module.get_state_components()
            action, prob, val = agent.choose_action(state)
            new_state, ext_reward, terminated, truncated, info = env.step(action)
            done= terminated or truncated
            new_state_components = curiosity_module.get_state_components()
            int_reward = curiosity_module.get_intrinsic_reward(state_components, new_state_components,i)
            total_reward = ext_reward + int_reward

            n_steps += 1
            ep_int_reward += int_reward

            new_state_processed = agent.preprocess_obs(new_state, add_batch=False).cpu()
            agent.remember(new_state_processed, action, prob, val, total_reward, done)

            if n_steps % rollout_length == 0:
                agent.learn()
                learn_iters += 1

            state = new_state

        if avg_ext_reward > best_score and i >= 10:
            best_score = avg_ext_reward
            agent.save_model()

        ep_ext_reward = info["episode"]["r"]
        ep_steps = info["episode"]["l"]
        ep_success = 1 if ep_ext_reward > 0 else 0

        int_rew_history.append(ep_int_reward)
        ext_rew_history.append(ep_ext_reward)
        success_history.append(ep_success)
        avg_ext_reward = np.mean(ext_rew_history[-100:])
        # Append to CSV
        with open(logs_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([i, ep_ext_reward, np.round(ep_int_reward,3), ep_success, ep_steps])

        print(f'Episode {i}: Ext reward {ep_ext_reward:.2f}, Int reward {ep_int_reward:.3f}, avg_ext_reward {avg_ext_reward:.2f},',
              f'episode_steps {ep_steps}, total_steps {n_steps}, learning_steps {learn_iters}, episode_steps {ep_steps}')
        
        state, info = env.reset()
        

    results = {
        'ext_rew_history': ext_rew_history,
        'int_rew_history': int_rew_history,
        'success_history': success_history
    }

    return results

def train_base_ppo(env,agent,num_games):

    logs_file='./logs/train_base_ppo.csv'
    # Write CSV header
    with open(logs_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_num", "ep_ext_reward", "ep_success", "ep_steps"])

    rollout_length = 1024

    best_score = env.reward_range[0]

    ext_rew_history = []
    success_history = []

    learn_iters = 0
    avg_ext_reward = 0
    n_steps = 0

    state, info = env.reset()
    for i in range(num_games):
        done = False
        while not done:
            action, prob, val = agent.choose_action(state)
            new_state, ext_reward, terminated, truncated, info = env.step(action)
            done= terminated or truncated
            n_steps += 1

            new_state_processed = agent.preprocess_obs(new_state, add_batch=False).cpu()
            agent.remember(new_state_processed, action, prob, val, ext_reward, done)

            if n_steps % rollout_length == 0:
                agent.learn()
                learn_iters += 1

            state = new_state

        if avg_ext_reward > best_score and i >= 10:
            best_score = avg_ext_reward
            agent.save_model()

        ep_ext_reward = info["episode"]["r"]
        ep_steps = info["episode"]["l"]
        ep_success = 1 if ep_ext_reward > 0 else 0

        ext_rew_history.append(ep_ext_reward)
        success_history.append(ep_success)
        avg_ext_reward = np.mean(ext_rew_history[-100:])
        # Append to CSV
        with open(logs_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([i, ep_ext_reward, ep_success, ep_steps])

        print(f'Episode {i}: Ext reward {ep_ext_reward:.2f}, avg_ext_reward {avg_ext_reward:.2f},',
              f'episode_steps {ep_steps}, learning_steps {learn_iters}, total_steps {n_steps}')
        

        state, info = env.reset()

    results = {
        'ext_rew_history': ext_rew_history,
        'success_history': success_history
    }

    return results