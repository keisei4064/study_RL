import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..")
)  # for importing the parent dirs
from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld
from common.utils import greedy_probs
from common.save_animation import save_animation_with_steps
from matplotlib.figure import Figure
import pathlib


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)  # ターゲット方策
        self.b = defaultdict(lambda: random_actions)  # 挙動方策
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)  # a ~ π(a|s)

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0  # ゴール地点の価値は0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)  # 次の状態での最大価値

        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        self.pi[state] = greedy_probs(
            self.Q, state, epsilon=0
        )  # greedyなターゲット方策πを更新
        self.b[state] = greedy_probs(
            self.Q, state, self.epsilon
        )  # ε-greedyな挙動方策bを更新


env = GridWorld()
agent = QLearningAgent()
episodes = 10000


figs: list[Figure] = []
frame_names: list[str] = []
save_rate = 500

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


env.render_q(agent.Q)


# 行動価値関数Q(s,a)更新のアニメーション
save_animation_with_steps(
    "Q-learning",
    figs,
    str(pathlib.Path(__file__).parent / "q_learning.gif"),
    interval=100,
    show_anim=True,
    frame_names=frame_names,
)
