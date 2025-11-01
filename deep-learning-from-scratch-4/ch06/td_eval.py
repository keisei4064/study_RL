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


class TdAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.01
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}  # ランダム方策
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def eval(self, state, reward, next_state, done):
        next_V = 0 if done else self.V[next_state]
        target = reward + self.gamma * next_V  # TD-target
        self.V[state] += (target - self.V[state]) * self.alpha


env = GridWorld()
agent = TdAgent()
episodes = 1000

figs: list[Figure] = []
frame_names: list[str] = []
save_rate = 20

for episode in range(episodes):
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.eval(state, reward, next_state, done)  # Rtが得られるたびに評価
        if done:
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
    "TD V(s) Evaluation",
    figs,
    str(pathlib.Path(__file__).parent / "td_v_evaluation.gif"),
    interval=100,
    show_anim=True,
    frame_names=frame_names,
)
