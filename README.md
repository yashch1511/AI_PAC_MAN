# AI_PAC_MAN
# 🎮 Deep Convolutional Q-Learning for Pac-Man

This project implements a **Deep Convolutional Q-Network (DCQN)** to train an AI agent to play **Atari Ms. Pac-Man** using [Gymnasium](https://gymnasium.farama.org/) and PyTorch.  
The agent learns to maximize rewards by eating pellets while avoiding ghosts, and it reaches human-level performance after training.

---

## 📌 Features
- ✅ Uses **Deep Convolutional Neural Networks (CNNs)** for visual input processing  
- ✅ Implements **Experience Replay** for stable learning  
- ✅ Uses a **Target Network** to reduce instability  
- ✅ Trains with **ε-greedy policy** (exploration vs exploitation)  
- ✅ Supports visualization of trained agent gameplay  

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yashch1511/Deep-Q-Learning-Pacman.git
cd Deep-Q-Learning-Pacman

```
---

## 2.Installation 
```
!pip install --upgrade "gymnasium[atari, accept-rom-license]"
!apt-get install -y swig
!pip install gymnasium[box2d]
!pip install torch torchvision imageio
```
Check versions:
```
import gymnasium as gym
import ale_py

print("Gymnasium version:", gym.__version__)
print("ALE-py version:", ale_py.__version__)
```

## 🧠 Neural Network Architecture

The DCQN consists of:

4 Convolutional layers with BatchNorm

Fully connected layers with ReLU

Final output layer → Q-values for all actions
```
Conv2d(3, 32, kernel_size=8, stride=4) → BatchNorm  
Conv2d(32, 64, kernel_size=4, stride=2) → BatchNorm  
Conv2d(64, 64, kernel_size=3, stride=1) → BatchNorm  
Conv2d(64, 128, kernel_size=3, stride=1) → BatchNorm  
FC: (10*10*128 → 512 → 256 → num_actions)

```

## 🎯 Training

Episodes: 2000

Max timesteps per episode: 10,000

Learning rate: 5e-4

Discount factor (γ): 0.99

Batch size: 64

ε-decay: 0.995 (from 1.0 → 0.01)

Training loop:
```
for episode in range(1, number_episodes + 1):
    state, _ = env.reset()
    ...
    action = agent.act(state, epsilon)
    ...
    agent.step(state, action, reward, next_state, done)

```

## 🎥 Results & Visualization

After training, run the agent to see gameplay:
```
show_video_of_model(agent, 'ALE/MsPacman-v5')
show_video()

```

## 📊 Sample Training Output
```
Episode 100  Average Score: 329.70
Episode 200  Average Score: 374.50
Episode 300  Average Score: 486.70
Episode 316  Average Score: 500.40
Environment solved in 216 episodes! Average Score: 500.40

```
## 📦 Dependencies

- Python 3.11+

- Gymnasium
 1.2.0

- ALE-py
 0.11.2

- PyTorch

- Torchvision

- Pillow

- ImageIO
