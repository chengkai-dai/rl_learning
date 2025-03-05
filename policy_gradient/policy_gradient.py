import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Actor (Policy) Network: outputs action probabilities
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)  # Larger network
        self.fc2 = nn.Linear(256, 128)  # Added layer
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

# Critic (Value) Network: estimates the state value V(s)
class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)  # Larger network
        self.fc2 = nn.Linear(256, 128)  # Added layer
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Compute Generalized Advantage Estimation (GAE)
def compute_gae(rewards, values, gamma, lam):
    gae = 0
    advantages = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages

def main():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Policy(state_dim, action_dim)
    critic = ValueNet(state_dim)
    
    # Better learning rates
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    
    gamma = 0.99    # Discount factor
    lam = 0.95      # GAE parameter
    entropy_coef = 0.01  # Add entropy regularization
    num_episodes = 5000  # More episodes
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        log_probs = []
        rewards = []
        states = []
        values = []
        entropies = []
        done = False
        
        # Generate an episode trajectory
        while not done:
            state_tensor = torch.FloatTensor(state)
            states.append(state_tensor)
            
            # Get value estimate for current state
            value = critic(state_tensor)
            values.append(value.item())
            
            # Actor: sample an action
            probs = actor(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()  # Calculate entropy for exploration
            
            log_probs.append(log_prob)
            entropies.append(entropy)
            
            next_state, reward, terminated, truncated, info = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated
            state = next_state
        
        # For GAE, we need an extra value for the final state
        state_tensor = torch.FloatTensor(state)
        final_value = critic(state_tensor)
        values.append(final_value.item())
        
        # Compute GAE advantages
        advantages = compute_gae(rewards, values, gamma, lam)
        advantages = torch.tensor(advantages, dtype=torch.float)
        
        # Normalize advantages for better stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute returns (targets for critic update)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float)
        
        # Convert lists to tensors
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)
        
        # Actor loss with entropy regularization
        actor_loss = -(log_probs_tensor * advantages.detach()).sum() - entropy_coef * entropies_tensor.sum()
        
        # Critic loss: mean squared error
        states_tensor = torch.stack(states)
        value_preds = critic(states_tensor).squeeze()
        critic_loss = nn.MSELoss()(value_preds, returns)
        
        # Update the actor (policy network)
        actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_optimizer.step()
        
        # Update the critic (value network)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        
        episode_total_reward = sum(rewards)
        episode_rewards.append(episode_total_reward)
        
        # More frequent progress updates
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            print(f"Episode {episode}: Total Reward: {episode_total_reward:.2f}, Avg Reward (last 10): {avg_reward:.2f}")
            
            # Early stopping if we've reached good performance
            if avg_reward > 490 and episode > 100:
                print(f"Environment solved in {episode} episodes!")
                break
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Actor-Critic with GAE")
    plt.grid(True)
    
    # Add moving average to the plot
    window_size = 20
    if len(episode_rewards) > window_size:
        moving_avg = [np.mean(episode_rewards[i:i+window_size]) for i in range(len(episode_rewards)-window_size+1)]
        plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, 'r', linewidth=2)
        plt.legend(['Rewards', f'{window_size}-Episode Moving Avg'])
    
    plt.show()
    
    env.close()

if __name__ == "__main__":
    main()