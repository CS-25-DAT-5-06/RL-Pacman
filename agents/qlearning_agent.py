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
        action_space_size=4,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01       
     ):
        """
        Initialize the Q-learning agent.    
            action_space_size: Number of possible actions (4 for Pacman: E, N, W, S) STOP REMOVED
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Initial exploration rate for epsilon-greedy
            epsilon_decay: Decay rate for epsilon after each episode
            epsilon_min: Minimum epsilon value
        """

        self.action_space_size = action_space_size
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table, (state, action) - > Q-value
        # Using defaultdict, so unseen states start with Q-value of 0

        self.q_table = defaultdict(lambda: np.zeros(action_space_size))

        # Stats for tracking:
        self.episodes_trained = 0
        self.total_steps = 0

    
    """
    Random thoughts about illigal actions:
    Action STOP removed, illegal/legal actions removed
    """

    def get_action(self, state, training=True):
        """
        Selects an using epsilon-greedy policy
            state: Current state
            GONE: legal_actions: List of legal action indices (None = all actions legal)
            training: If True, use epsilon-greedy. If False, use greedy (for evaluation)
        Returns:
            action: integer action index (0-3)
        """

        # During evaluation, always exploit:
        if not training:
            epsilon = 0.0
        else:
            epsilon = self.epsilon

        # Epsilon-greedy action selection:
        if np.random.random() < epsilon:    # Generates random number between 0 and 1, as epsilon decays, less random actions.
            # Explore: Random action
            return np.random.randint(self.action_space_size) # Just a random action
        else:
            # Exploit: best action according to q-values
            q_values = self.q_table[state] #array of four numbers, q-value for each of the four actions
            return np.argmax(q_values) #returns index of highest q-value
        """
        Note: The argmax function is ´not random, will always pick the first
        value, if q_values are equal. Exploration bias. Could be made better
        """

    def update(self, state, action, reward, next_state, done):
        """
        Update gets called every single action agent takes in the environment
        Update Q-value using the Q-learning update rule

        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

        Arguments:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after taking action
            done: episode terminated?
        """

        current_q = self.q_table[state][action]

        if done:
            # No future rewards if episode ended
            target_q = reward
        else:
            # Bellman equation: current reward 0 discounted max future Q
            # Temporal difference learning
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
        # Q-learning update
        new_q = current_q + self.alpha * (target_q - current_q)
        self.q_table[state][action] = new_q

        self.total_steps += 1

    def decay_epsilon(self):
        """
        Decay epsilonm after each episode
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes_trained += 1