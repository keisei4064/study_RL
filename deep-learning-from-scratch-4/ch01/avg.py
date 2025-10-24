import numpy as np

# naive implementation
print("naive implementation - formula (1.1)")
np.random.seed(0)
rewards = []

for n in range(1, 11):
    reward = np.random.rand()
    rewards.append(reward)
    Q = sum(rewards) / n  # 式(1.1)
    print(f"iter: {n},\treward: {reward:.3f}, Q: {Q:.3f}")

print("---")

# incremental implementation
print("incremental implementation - formula (1.5)")
np.random.seed(0)
Q = 0

for n in range(1, 11):
    reward = np.random.rand()
    # 式(1.5)
    # Q = Q + (reward - Q) / n  # 同じ式
    Q += (reward - Q) / n
    print(f"iter: {n},\treward: {reward:.3f}, Q: {Q:.3f}")
