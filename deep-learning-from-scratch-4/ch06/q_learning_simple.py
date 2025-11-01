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


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.2
        # self.epsilon = 0.1
        self.action_size = 4
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)  # random (ε)
        else:
            qs = [self.Q[state, a] for a in range(self.action_size)]
            return np.argmax(qs)  # greedy

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0  # ゴール地点の価値は0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)  # 次の状態での最大価値

        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha


env = GridWorld()
agent = QLearningAgent()
episodes = 1000


figs: list[Figure] = []
frame_names: list[str] = []
save_rate = 50


for episode in range(episodes):
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state, done)
        if done:
            break
        state = next_state

    # 現在の行動価値を描画
    if episode % save_rate == 0:
        fig, _ = env.render_q(agent.Q, show_plt=False)
        figs.append(fig)
        frame_names.append(f"episode={episode}")

_, pi_fig = env.render_q(agent.Q)


# 行動価値関数Q(s,a)更新のアニメーション
save_animation_with_steps(
    "Q-learning",
    figs,
    str(pathlib.Path(__file__).parent / "q_learning.gif"),
    interval=100,
    show_anim=True,
    frame_names=frame_names,
)
