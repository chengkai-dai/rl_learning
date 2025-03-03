import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the policy network
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # Softmax over the output for action probabilities
        return torch.softmax(self.fc2(x), dim=-1)

def compute_reward_to_go(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    return returns

def main():
    # Create the environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = Policy(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    
    gamma = 0.99
    num_episodes = 5000
    episode_rewards = []

    early_stop_window = 20
    reward_threshold = 495  # 可根据实际情况调整
    
    # Training loop
    for episode in range(num_episodes):
        state, info = env.reset()
        log_probs = []
        rewards = []
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state)
            probs = policy(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        
        episode_total_reward = sum(rewards)
        episode_rewards.append(episode_total_reward)
        
        # Compute discounted returns and update policy
        returns = compute_reward_to_go(rewards, gamma)
        loss = 0
        for log_prob, R in zip(log_probs, returns):
            loss += -log_prob * R
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Total Reward: {episode_total_reward}")
        if len(episode_rewards) >= early_stop_window:
            recent_avg = sum(episode_rewards[-early_stop_window:]) / early_stop_window
            if recent_avg >= reward_threshold:
                print(f"Convergence achieved at episode {episode} with average reward {recent_avg:.2f} over the last {early_stop_window} episodes.")
                break
    # Validation: run a few evaluation episodes without policy updates.
    eval_rewards = []
    num_eval_episodes = 10
    for _ in range(num_eval_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state)
            probs = policy(state_tensor)
            # Choose the action with highest probability for evaluation.
            action = torch.argmax(probs).item()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        eval_rewards.append(total_reward)
    
    print("Evaluation rewards:", eval_rewards)
    
    # Visualization: Plot the training reward progress.
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Progress")
    plt.grid(True)
    plt.show()
    
    env.close()

if __name__ == "__main__":
    main()