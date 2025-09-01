# qlearning_frozenlake.py
# Q-learning on Gymnasium FrozenLake-v1 (tabular). MIT-like example.

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete


def return_decayed_value(starting_value: float, global_step: int, decay_step: int) -> float:
    """starting_value * 0.1 ** (global_step / decay_step)"""
    return starting_value * np.power(0.1, (global_step / decay_step))


def epsilon_greedy_action(Q: np.ndarray, s: int, epsilon: float, rng: np.random.Generator) -> int:
    nA = Q.shape[1]
    if rng.random() < epsilon:
        return int(rng.integers(nA))
    return int(np.argmax(Q[s]))


def print_policy_grid(policy: np.ndarray, nrow: int, ncol: int):
    """Pretty-print a discrete policy for FrozenLake.
    Actions: 0=Left, 1=Down, 2=Right, 3=Up (Gymnasium convention)"""
    sym = {0: "<", 1: "v", 2: ">", 3: "^"}
    lines = []
    for r in range(nrow):
        row_syms = []
        for c in range(ncol):
            a = int(policy[r * ncol + c])
            row_syms.append(sym.get(a, "?"))
        lines.append("  ".join(row_syms))
    print("\n".join(lines))


def evaluate_greedy(env_id: str, Q: np.ndarray, make_kwargs=None, episodes: int = 100, seed: int | None = 0):
    """Run greedy policy induced by Q and report average return & success rate."""
    make_kwargs = make_kwargs or {}
    env = gym.make(env_id, **make_kwargs)
    nS, nA = env.observation_space.n, env.action_space.n
    assert Q.shape == (nS, nA)

    successes = 0
    returns = []
    for ep in range(episodes):
        s, info = env.reset(seed=seed if ep == 0 else None)
        G = 0.0
        for t in range(200):
            a = int(np.argmax(Q[s]))
            s, r, terminated, truncated, info = env.step(a)
            G += r
            if terminated or truncated:
                if r > 0:
                    successes += 1
                break
        returns.append(G)
    env.close()
    return float(np.mean(returns)), successes / episodes


def q_learning(
    env_id: str = "FrozenLake-v1",
    make_kwargs: dict | None = None,
    total_episodes: int = 10000,
    max_steps_per_ep: int = 200,
    alpha: float = 0.8,          # 学习率
    gamma: float = 0.99,         # 折扣
    eps_start: float = 1.0,      # 探索率起始
    eps_end: float = 0.05,       # 探索率下界
    eps_decay_steps: int = 5000, # 指数衰减步长（越大衰减越慢）
    seed: int | None = 42,
):
    """
    Tabular Q-learning for Gymnasium environments (discrete state/action).
    Returns: Q-table, (avg_return, success_rate)
    """
    make_kwargs = make_kwargs or {}
    if env_id == "FrozenLake-v1" and "is_slippery" not in make_kwargs:
        make_kwargs["is_slippery"] = False

    env = gym.make(env_id, **make_kwargs)
    assert isinstance(env.observation_space, Discrete)
    assert isinstance(env.action_space, Discrete)

    nS = env.observation_space.n
    nA = env.action_space.n
    rng = np.random.default_rng(seed)

    Q = np.zeros((nS, nA), dtype=np.float64)

    # 主循环
    for ep in range(total_episodes):
        s, info = env.reset(seed=seed if ep == 0 else None)

        eps = max(eps_end, return_decayed_value(eps_start, ep, eps_decay_steps))

        for t in range(max_steps_per_ep):
            a = epsilon_greedy_action(Q, s, eps, rng)
            s_next, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            # Q-learning 目标：r + γ * max_a' Q[s', a']
            best_next = 0.0 if done else np.max(Q[s_next])
            td_target = r + gamma * best_next
            td_error = td_target - Q[s, a]
            Q[s, a] += alpha * td_error

            s = s_next
            if done:
                break

        # 可选：每隔一段打印一次
        if (ep + 1) % max(1, total_episodes // 10) == 0:
            avg_ret, succ = evaluate_greedy(env_id, Q, make_kwargs, episodes=50)
            print(f"[{ep+1}/{total_episodes}] eps≈{eps:.3f}  avg_return={avg_ret:.3f}  success={succ*100:.1f}%")

    env.close()

    avg_ret, succ = evaluate_greedy(env_id, Q, make_kwargs, episodes=200)
    print("\nFinal greedy evaluation:")
    print(f"Average return (200 eps) = {avg_ret:.3f},  Success rate = {succ*100:.1f}%")

    side = int(np.sqrt(nS))
    greedy_policy = np.argmax(Q, axis=1)
    print("\nGreedy policy grid:")
    print_policy_grid(greedy_policy, nrow=side, ncol=side)

    return Q, (avg_ret, succ)


if __name__ == "__main__":
    # 1) 确定性湖面：更容易学
    Q, metrics = q_learning(
        env_id="FrozenLake-v1",
        make_kwargs={"is_slippery": False},
        total_episodes=8000,
        max_steps_per_ep=200,
        alpha=0.8,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=5000,
        seed=0,
    )

    # 2) 想挑战随机滑冰，把 is_slippery=True；适当增大训练轮次、降低 alpha。
    # Q_rand, _ = q_learning(
    #     env_id="FrozenLake-v1",
    #     make_kwargs={"is_slippery": True},
    #     total_episodes=100000,
    #     max_steps_per_ep=10000,
    #     alpha=0.1,
    #     gamma=0.99,
    #     eps_start=1.0,
    #     eps_end=0.01,
    #     eps_decay_steps=100000,
    #     seed=0,
    # )