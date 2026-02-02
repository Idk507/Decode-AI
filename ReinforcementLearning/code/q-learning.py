import numpy as np
import random

class QLearningAgent:
    """
    Tabular Q-learning agent for discrete state and action spaces.
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.Q[state])

    def update(self, s, a, r, s_next):
        """
        Q-learning update rule.
        """
        best_next_action = np.argmax(self.Q[s_next])
        td_target = r + self.gamma * self.Q[s_next, best_next_action]
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
