"""
Tabular Q-learning agent -
"""

import numpy as np
import pickle
from collections import defaultdict

"""
Create a training script in experiments/train_.py to


Gymnasium Wrapper (gymenv_v2)
  ↓  
Q-Learning Agent (qlearning_agent.py)
  ↓ 
Training Script (experiments/train_qlearning.py)
"""

class QLearningAgent:

    def __init__(
        self,
        action_space_size=5,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01       
     ):
        """
        Initialize the Q-learning agent.    
            action_space_size: Number of possible actions (5 for Pacman: STOP, E, N, W, S)
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Initial exploration rate for epsilon-greedy
            epsilon_decay: Decay rate for epsilon after each episode
            epsilon_min: Minimum epsilon value
        """

        self.action_space_size = action_space_size
        self.alpha = learning_rate
        self.gamma = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table, (state, action) - > Q-value
        # Using defaultdict, so unseen states start with Q-value of 0

        self.q_table = defaultdict(lambda: np.zeros(action_space_size))


