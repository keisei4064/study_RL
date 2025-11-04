if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

if not hasattr(np, "int"):
    np.int = int  # pyright: ignore[reportAttributeAccessIssue]
import gymnasium as gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class Policy(Model):
    def __init__(self, action_size: int):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)  # 行動選択肢

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x)) # 方策（確率分布）
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.pi)

    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(state)
        probs = probs[0]
        # 方策から具体的な行動をサンプリング
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    # 1エピソードが終わったときに呼び出される
    def update(self):
        self.pi.cleargrads()

        G, loss = 0, 0
        
        # ゴールから遡って全ステップの報酬を計算
        for reward, prob in reversed(self.memory):
            # 累積報酬 G(τ)
            G = reward + self.gamma * G

        for reward, prob in self.memory:
            # 式(9.2)
            #   最大化問題を最小化問題に変換するため，-1 を掛ける
            loss += -F.log(prob) * G

        loss.backward()
        self.optimizer.update()
        self.memory = []


episodes = 3000
env = gym.make("CartPole-v0")
agent = Agent()
reward_history = []

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:  # ゴールするまで
        action, prob = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.add(reward, prob)
        state = next_state
        total_reward += reward

    # 方策更新
    agent.update()

    reward_history.append(total_reward)
    if episode % 100 == 0:
        print("episode :{}, total reward : {:.1f}".format(episode, total_reward))


# plot
from common.utils import plot_total_reward

plot_total_reward(reward_history)
