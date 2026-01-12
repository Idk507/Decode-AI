"""
1. Simple Environment (Toy MDP)

We’ll use a small episodic environment:

States: 0 → 1 → 2 → terminal

Actions: left (0), right (1)

Rewards:

Reaching terminal gives +1

Otherwise 0

This keeps the focus on Monte Carlo, not environment complexity.

"""
import random
from collections import defaultdict

class SimpleEnvironment:
    def __init__(self):
        self.terminal_state = 3
        self.reset()

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        """
        action: 0 = left, 1 = right
        """
        if action == 1:
            self.state += 1
        else:
            self.state = max(0, self.state - 1)

        if self.state == self.terminal_state:
            return self.state, 1.0, True
        else:
            return self.state, 0.0, False

# Policy (ε-Greedy)

def epsilon_greedy_policy(Q, state, n_actions, epsilon):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        return max(range(n_actions), key=lambda a: Q[state][a])

"""
Monte Carlo Policy Evaluation (State Value)
Theory reminder:
𝑉(s) = E[G_t | S_t = S ]
Implementation (First-Visit MC)
"""


def monte_carlo_policy_evaluation(env, policy, episodes=5000, gamma=0.9):
    returns = defaultdict(list)
    V = defaultdict(float)

    for _ in range(episodes):
        episode = []
        state = env.reset()
        done = False

        # Generate episode
        while not done:
            action = policy(state)
            next_state, reward, done = env.step(action)
            episode.append((state, reward))
            state = next_state

        # Compute returns
        G = 0
        visited_states = set()

        for state, reward in reversed(episode):
            G = reward + gamma * G
            if state not in visited_states:
                returns[state].append(G)
                V[state] = sum(returns[state]) / len(returns[state])
                visited_states.add(state)

    return V

"""

5. Monte Carlo Control (Learning Optimal Policy)

Now we learn Q(s,a) and improve the policy.

Algorithm

Generate episode

Compute returns

Update Q

Improve policy (ε-greedy)

"""
def monte_carlo_control(env, episodes=10000, gamma=0.9, epsilon=0.1):
    n_actions = 2
    Q = defaultdict(lambda: [0.0] * n_actions)
    returns = defaultdict(list)

    for _ in range(episodes):
        episode = []
        state = env.reset()
        done = False

        # Generate episode using ε-greedy
        while not done:
            action = epsilon_greedy_policy(Q, state, n_actions, epsilon)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        # Compute returns
        G = 0
        visited = set()

        for state, action, reward in reversed(episode):
            G = reward + gamma * G
            if (state, action) not in visited:
                returns[(state, action)].append(G)
                Q[state][action] = sum(returns[(state, action)]) / len(returns[(state, action)])
                visited.add((state, action))

    return Q

env = SimpleEnvironment()

Q = monte_carlo_control(env)

print("Learned Q-values:")
for state in sorted(Q):
    print(f"State {state}: Left={Q[state][0]:.2f}, Right={Q[state][1]:.2f}")


# . Extracting the Optimal Policy
def extract_policy(Q):
    policy = {}
    for state in Q:
        policy[state] = max(range(len(Q[state])), key=lambda a: Q[state][a])
    return policy

optimal_policy = extract_policy(Q)

print("\nOptimal Policy:")
for state in sorted(optimal_policy):
    print(f"State {state}: Action {optimal_policy[state]}")


