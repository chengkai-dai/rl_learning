# Policy Gradient

$\theta$ represents the **parameters** of your policy $\pi_{\theta}(a|s)$
$$
J(\theta)= \mathbb{E}_{\tau \sim \pi_{\theta}}(R(\tau))
$$
where $\tau$ is a trajectory represent a seqeunce of state-action.
$$
\tau=(s_0, a_0, s_1,a_1\dots,s_T,a_T)
$$
 and $R$ is the reward from $\tau$
$$
R(\tau)=\sum_{t=0}^{T}r(s_t,a_t)
$$
The probability of trajectory $\tau$ is 
$$
p_{\theta}(\tau)=p(s_0)\prod_{t=0}^{T}\pi_{\theta}(a_r|s_t)p(s_{t+1}|s_t,a_t)
$$
Specifically,
$$
p_{\theta}(\tau)=p(s_0)\cdot\pi_{\theta}(a_0|s_0)\cdot p(s_1|s_0,a_0)\cdot \pi_{\theta}(a_1|a_0)\cdot p(s_2|s_1,a_1)\dots\pi_{\theta}(a_T|s_{T})\cdot p(s_{T+1}|s_T,a_T)
$$
The goal is to get the gradient of $J_{\theta}$, which is $\nabla_{\theta} J_{\theta}$,
$$
\nabla_{\theta} J(\theta)=\nabla_{\theta}\mathbb{E}_{\tau \sim \pi_{\theta}}(R(\tau))
$$
First we formulate the problem as intergral form
$$
\nabla_{\theta}J(\theta)=\nabla_{\theta}\int_{\tau}p_{\theta}(\tau)R(\tau)d\tau
$$
Equiventally,
$$
\nabla_{\theta}J(\theta)=\int_{\tau}\nabla_{\theta}[p_{\theta}(\tau)]R(\tau)d\tau
$$


using the log-gradient trick,
$$
\nabla_{\theta}p_{\theta}(x)=p_{\theta}(x)\nabla_{\theta}\log p_{\theta}(x)
$$


we get,
$$
\nabla_{\theta}J(\theta)=\int_{\tau}p_\theta(\tau)\nabla_{\theta}\log p_\theta(\tau) R(\tau)d\tau
$$
change to the sum form,
$$
\nabla_{\theta}J(\theta)=\sum_{\tau}p_\theta(\tau)\nabla_{\theta}\log p_\theta(\tau) R(\tau)d\tau
$$
we get the expection form,
$$
\nabla_{\theta}J(\theta)=\mathbb{E}_{\tau \sim \pi_{\theta}}[\nabla_{\theta}\log p_\theta(\tau) R(\tau)]
$$
From the definition of $p_{\theta}(\tau)$, we get
$$
\log p_{\theta}(\tau)=\log p(s_0)+ \sum_{t=0}^T \log \pi_{\theta}(a_t|s_t)+\sum_{t=0}^T \log p(s_{t+1}|s_t,a_t)
$$

The state transition probabilities $p(s_{t+1}|s_t,a_t)$ are independent of the policy parameters $\theta$ .

Similarly, the initial state probability $p(s_0)$ also does not depend on $\theta$ .

Therefore, when taking the gradient with respect to $\theta$ , these terms vanish:
$$
\nabla_{\theta}\log p_\theta(\tau)=\sum_{t=0}^T\nabla_\theta\log \pi_{\theta}(a_t|s_t)
$$
Here,we derive the **Policy Gradient Theorem**:
$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\Bigl[\Bigl(\sum_{t=0}^{T}\nabla_\theta\log\pi_{\theta}(a_t|s_t)\Bigr)R(\tau)\Bigr]
$$
This is an expectation, which means that we can estimate it with a sample mean. If we collect a set of trajectories $\mathcal{D} = \{\tau_i\}_{i=1,...,N}$  where each trajectory is obtained by letting the agent act in the environment using the policy $\pi_{\theta}$, the policy gradient can be estimated with
$$
\hat g=\frac{1}{|\mathcal{D}|}\sum_{\tau\in \mathcal{D}}\sum_{t=0}^{T}\nabla_\theta\log\pi_{\theta}(a_t|s_t)R(\tau)
$$

## Reward-to-go policy gradient

Don’t Let the Past Distract You

Agents should really only reinforce actions on the basis of their *consequences*. Rewards obtained before taking an action have no bearing on how good that action was: only rewards that come *after*.
$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\Bigl[\sum_{t=0}^{T}\nabla_\theta\log\pi_{\theta}(a_t|s_t)\sum_{t^\prime=t}^TR(s_{t^\prime},a_{t^\prime},s_{t^\prime})\Bigr]
$$

## Implementation

```python
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
```

## Baseline

In policy gradient methods, a **baseline** is an auxiliary function used to reduce the variance of the gradient estimates without introducing bias.

For a typical policy gradient (like REINFORCE), the gradient update is given by:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \, R_t\right],
$$

where $R_t$ is the return (cumulative discounted reward) from time step $t$.

When you introduce a baseline $b(s_t)$ (which is a function of the state $s_t$ only), the update becomes:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \, \big(R_t - b(s_t)\big)\right]
$$

As long as the baseline $b(s_t)$ does not depend on the action $a_t$, subtracting it does not change the expected value of the gradient (i.e., the estimate remains unbiased) but can significantly reduce the variance.

> policy gradients use random returns based on reward samples, which are noisy and random. This causes fluctuations in the estimated return. For analogy: imagine measuring your performance by the total reward of one episode, which can differ randomly, not necessarily reflecting your action’s quality. This randomness adds noise to the gradient. Using a baseline (an average return) reduces this variance, allowing for a more stable gradient estimate.

## Implementation with State-Value Function $V(s_t)$

```python
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 定义策略网络
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # 输出动作概率分布
        return torch.softmax(self.fc2(x), dim=-1)

# 定义状态值函数网络
class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # 输出一个标量，代表状态的价值
        return self.fc2(x)

# 计算 reward-to-go: 对于每个时间步 t，计算 R_t = r_t + gamma * r_(t+1) + ...
def compute_reward_to_go(rewards, gamma):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns)

def main():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 创建策略网络和状态值函数网络
    policy = Policy(state_dim, action_dim)
    value_net = ValueNet(state_dim)
    
    # 定义两个优化器
    optimizer_policy = optim.Adam(policy.parameters(), lr=1e-2)
    optimizer_value = optim.Adam(value_net.parameters(), lr=1e-2)
    
    gamma = 0.99
    num_episodes = 500
    episode_rewards = []
    
    # 早停条件
    early_stop_window = 20
    reward_threshold = 195

    for episode in range(num_episodes):
        state, info = env.reset()
        log_probs = []
        rewards = []
        states = []  # 保存所有状态，用于价值网络的更新
        done = False
        
        # 收集整个轨迹数据
        while not done:
            state_tensor = torch.FloatTensor(state)
            states.append(state_tensor)  # 保存状态
            
            probs = policy(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            log_probs.append(log_prob)
            rewards.append(reward)
            
            state = next_state
        
        # 计算每个时间步的 reward-to-go
        returns = compute_reward_to_go(rewards, gamma)
        returns = returns.detach()
        
        # 将 states 列表转换为 tensor (batch_size, state_dim)
        states_tensor = torch.stack(states)
        
        # 使用状态值网络计算每个状态的 baseline 值 V(s_t)
        values = value_net(states_tensor).squeeze()  # shape: [num_steps]
        
        # 计算 advantage = returns - V(s_t)
        advantages = returns - values
        
        # 策略梯度损失：利用 advantage 来减少方差
        policy_loss = (-torch.stack(log_probs) * advantages).sum()
        
        # 状态值函数的均方误差损失
        value_loss = nn.MSELoss()(values, returns)
        
        # 更新策略网络
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()
        
        # 更新状态值网络
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()
        
        episode_total_reward = sum(rewards)
        episode_rewards.append(episode_total_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Total Reward: {episode_total_reward}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")
        
        # 早停条件检查
        if len(episode_rewards) >= early_stop_window:
            recent_avg = sum(episode_rewards[-early_stop_window:]) / early_stop_window
            if recent_avg >= reward_threshold:
                print(f"Convergence achieved at episode {episode} with average reward {recent_avg:.2f} over the last {early_stop_window} episodes.")
                break

    # 绘制训练奖励曲线
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Progress with Value Baseline")
    plt.grid(True)
    plt.show()
    
    env.close()

if __name__ == "__main__":
    main()
```

