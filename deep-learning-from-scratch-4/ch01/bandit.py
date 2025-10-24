import numpy as np
import matplotlib.pyplot as plt


# ---

# バンディット
class Bandit:
    def __init__(self, arms=10):
        # 各アームの報酬率(未知)
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]

        # ベルヌーイ分布に従って報酬を返す
        if rate > np.random.rand():
            return 1
        else:
            return 0


# # 引いてみる ---
# bandit = Bandit()

# for i in range(3):
#     print(bandit.play(i))

# ---

# # 0番目の期待値を推定してみる ---
# bandit = Bandit()
# Q = 0

# for n in range(1, 101):  # 100回試行
#     reward = bandit.play(0)
#     Q += (reward - Q) / n
#     print("estimated rate:", Q)
    
# print(f"true rate: {bandit.rates[0]}")

# ---

# # 10台について推定
# bandit = Bandit()
# Qs = np.zeros(10)
# ns = np.zeros(10)  # プレイ回数

# for n in range(1, 101):  # 100回試行
#     action = np.random.randint(0, 10)  # ランダムにアームを選ぶ
#     reward = bandit.play(action)
#     ns[action] += 1
#     Qs[action] += (reward - Qs[action]) / ns[action]


# np.set_printoptions(precision=2, suppress=True)  # 小数点以下2桁、指数表記を抑制
# print(f"true rates:\t\t{bandit.rates}")
# print(f"estimated rates:\t{Qs}")

# ---

# Agent : ε-greedy法に従って行動
class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self) -> int:
        if np.random.rand() < self.epsilon:  # ε-greedy
            # 探索
            return np.random.randint(0, len(self.Qs))
        # 活用
        return int(np.argmax(self.Qs))

# ---

if __name__ == "__main__":
    steps = 1000
    epsilon = 0.1

    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward

        total_rewards.append(total_reward)  # 総報酬
        rates.append(total_reward / (step + 1))  # 勝率

    print(f"{total_reward=}")

    plt.ylabel("Total reward")
    plt.xlabel("Steps")
    plt.plot(total_rewards)
    plt.show()

    # 勝率
    plt.ylabel("Rates")
    plt.xlabel("Steps")
    plt.plot(rates)
    plt.show()
