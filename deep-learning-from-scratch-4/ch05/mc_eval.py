import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..")
)  # for importing the parent dirs
from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld
from common.save_animation import save_animation_with_steps
from matplotlib.figure import Figure
import pathlib


class RandomAgent:
    def __init__(self):
        self.gamma = 0.9  # 割引率
        self.action_size = 4

        # ランダム方策
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)

        # 推定状態価値
        self.V = defaultdict(lambda: 0)
        self.cnts = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):  # s, a, r
        data = (state, action, reward)  # save as tuple
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def eval(self):
        G = 0

        # 逆順に割引率をかけながら評価
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward  # p.142: 収益の計算
            self.cnts[state] += 1  # s で得たデータ数
            # 状態価値 = E[G | s] をモンテカルロ法で計算
            self.V[state] += (G - self.V[state]) / self.cnts[state]


env = GridWorld()
agent = RandomAgent()
episodes = 1000

figs: list[Figure] = []
frame_names: list[str] = []
save_rate = 20

for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.add(state, action, reward)
        if done:  # ゴールしたら評価
            agent.eval()
            break

        state = next_state

    # 現在の状態価値を描画
    if episode % save_rate == 0:
        fig = env.render_v(agent.V, show_plt=False)
        figs.append(fig)
        frame_names.append(f"episode={episode}")


env.render_v(agent.V)

# 状態価値更新のアニメーション
save_animation_with_steps(
    "Monte Carlo V(s) Evaluation",
    figs,
    str(pathlib.Path(__file__).parent / "monte_carlo_v_evaluation.gif"),
    interval=100,
    show_anim=True,
    frame_names=frame_names,
)
