import numpy as np

class LinearTDValueFunction:
    """
    Linear function approximator for V(s) using TD learning.
    """

    def __init__(self, n_features, alpha=0.01, gamma=0.99):
        self.theta = np.zeros(n_features)
        self.alpha = alpha
        self.gamma = gamma

    def features(self, state):
        """
        Example feature extractor.
        In practice, this should encode meaningful properties of the state.
        """
        return np.array(state)

    def predict(self, state):
        phi = self.features(state)
        return np.dot(self.theta, phi)

    def update(self, s, r, s_next):
        phi_s = self.features(s)
        phi_next = self.features(s_next)

        v_s = np.dot(self.theta, phi_s)
        v_next = np.dot(self.theta, phi_next)

        td_error = r + self.gamma * v_next - v_s
        self.theta += self.alpha * td_error * phi_s
