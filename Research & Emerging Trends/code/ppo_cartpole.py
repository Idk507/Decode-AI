import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)
    
    def forward(self, state):
        features = self.shared(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

# PPO Agent
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2, epochs=10):
        self.model = ActorCritic(state_dim, action_dim).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
    
    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        for r, v, nv, done in zip(reversed(rewards), reversed(values), reversed([next_value] + values[:-1]), reversed(dones)):
            delta = r + self.gamma * nv * (1 - done) - v
            gae = delta + self.gamma * self.lam * (1 - done) * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32).cuda()
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        for _ in range(self.epochs):
            action_logits, values = self.model(states)
            dist = torch.distributions.Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            loss = policy_loss + 0.5 * value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).cuda()
        action_logits, value = self.model(state)
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

# Training loop
def train_ppo(env_name='CartPole-v1', max_timesteps=100000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, action_dim)
    
    state = env.reset()
    episode_reward = 0
    states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
    timestep = 0
    
    while timestep < max_timesteps:
        action, log_prob, value = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        dones.append(done)
        
        state = next_state
        episode_reward += reward
        timestep += 1
        
        if done or len(states) >= 2048:
            if done:
                print(f"Episode Reward: {episode_reward}")
                episode_reward = 0
                state = env.reset()
            
            # Compute returns and advantages
            next_value = agent.model(torch.tensor(state, dtype=torch.float32).cuda())[1]
            returns = []
            R = next_value.item()
            for r, done in zip(reversed(rewards), reversed(dones)):
                R = r + agent.gamma * R * (1 - done)
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32).cuda()
            advantages = agent.compute_gae(rewards, [v.item() for v in values], next_value.item(), dones)
            
            # Update policy
            states_t = torch.tensor(states, dtype=torch.float32).cuda()
            actions_t = torch.tensor(actions, dtype=torch.long).cuda()
            log_probs_t = torch.tensor([lp.item() for lp in log_probs], dtype=torch.float32).cuda()
            agent.update(states_t, actions_t, log_probs_t, returns, advantages)
            
            states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
    
    env.close()

if __name__ == "__main__":
    train_ppo()
