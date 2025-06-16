# Curiosity-Driven Exploration via Grounded Question Answering in MiniGrid Environments

This project implements curiosity-driven exploration in reinforcement learning through grounded question answering, inspired by the paper **[Ask and Explore: Grounded Question Answering for Curiosity-Driven Exploration](https://arxiv.org/abs/2104.11902)**. The environment used is **MiniGrid**, and the learning algorithm is **Proximal Policy Optimization (PPO)**. The agent learns to ask informative questions about its environment and uses the answers to improve its exploration behavior and decision-making efficiency.

---

## 🧠 Overview

Exploration is a core challenge in reinforcement learning. Rather than relying solely on intrinsic reward mechanisms like novelty or prediction error, this project explores a **grounded question answering (GQA)** strategy. The agent actively asks questions about its surroundings (e.g., "What is in front of me?") and uses the responses to shape its internal knowledge and improve policy learning.

Two main agent types are implemented:

- **Base PPO (`base_ppo`)**: A standard PPO agent without the ability to ask questions.
- **Ask-and-Explore PPO (`ane_ppo`)**: An enhanced PPO agent that asks up to `n` grounded questions per step to guide exploration.

---

## 🛠️ Implementation Details

### Environment
- **MiniGrid**: A lightweight gridworld environment suite ideal for testing generalization and reasoning in agents.
- Custom **curiosity module** allow the agent to interface with a question-answering module.

### Grounded Question Answering
The `ane_ppo` agent uses predefined templates to ask grounded questions such as:
- "Can I move forward?"
- "What is to my left?"
- "Is the goal visible?"

Answers change across steps are used to incentivise agent exploration through an ***intrinsic reward***:

![image](https://github.com/user-attachments/assets/0bd8bc72-f0b4-42b2-b721-df9d9e6664a6)

### Hyperparameters

| Hyperparameter         | Value            |
|------------------------|------------------|
| Learning Rate          | 1e-4             |
| Discount Factor (γ)    | 0.99             |
| GAE Lambda             | 0.95             |
| PPO Clip Range         | 0.2              |
| Entropy Coefficient    | 0.01             |
| Value Loss Coefficient | 0.5              |
| Optimizer              | Adam             |
| Rollout Length         | 1024             |
| Batch Size             | 64               |
| Epochs per Update      | 4                |
| Number of Questions (n)| 1–5(user-defined)|

---

## 📊 Results 

### Training
| Extrinsic Reward                                                                          | Success Probability                                                                       |
|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| ![image](https://github.com/user-attachments/assets/bf836bc0-e30d-46ef-9faf-b7907cf9f96f) | ![image](https://github.com/user-attachments/assets/95765a09-34d0-43dc-b725-5dcb7140cdca) |

### Testing
| Positive Results                                                                          | Negative Results                                                                          |
|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| ![image](https://github.com/user-attachments/assets/bec8aa55-b0b0-4fc5-ba7b-56f6bf500b90) | ![image](https://github.com/user-attachments/assets/dd03d028-26c9-4512-92ec-12879567744b) |
 
## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/AngeloGianfelice/Ask_and_Explore.git
cd Ask_and_Explore
```

### 2. Install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Run the Code
The main script takes the following arguments:

**--mode**: 'train' or 'test'

**--algorithm**: 'ane_ppo' or 'base_ppo'

**--n**: Integer (1–5), number of questions asked per step (used only by ane_ppo)

Train Ask-and-Explore Agent
```bash
python main.py --mode train --algorithm ane_ppo --n 3
```
Test Baseline PPO Agent
```bash
python main.py --mode test --algorithm base_ppo --n 1
```
⚠️ Note: The --n argument is required even for base_ppo, but it is ignored internally.

## 📚 References
1. Kaur, J. N., Jiang, Y., & Liang, P. P. (2021). Ask & Explore: Grounded Question Answering for
Curiosity-Driven Exploration. arXiv preprint arXiv:2104.11902. https://arxiv.org/abs/2104.11902
2. Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018). Exploration by random network distillation. arXiv
preprint arXiv:1810.12894. https://arxiv.org/abs/1810.12894
3. Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-driven exploration by self-supervised
prediction. arXiv preprint arXiv:1705.05363. https://arxiv.org/abs/1705.05363
4. Chevalier-Boisvert, M., Willems, L., & Pal, S. (2018). Minimalistic Gridworld Environment for OpenAI Gym.
https://github.com/Farama-Foundation/Minigrid
5. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization
Algorithms (arXiv:1707.06347). arXiv. https://arxiv.org/abs/1707.06347


