import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..")
)  # for importing the parent dirs
from collections import defaultdict, deque
import numpy as np
from common.gridworld import GridWorld
from common.utils import greedy_probs
from common.save_animation import save_animation_with_steps
from matplotlib.figure import Figure
import pathlib


class SarsaAgent:
    def __init__(self):
        self.gamma = 0.9  # 割引率
        self.alpha = 0.8  # 指数移動平均の更新係数
        self.epsilon = 0.1  # ε-greedyのε
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}  # ランダム方策
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)  # Q(s, a)
        self.memory = deque(maxlen=2)

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)  # a ~ π(a|s)

    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))  # (s, a, r, is_goal)
        if len(self.memory) < 2:  # 2ステップ分無いとSARSAは更新できない
            return

        state, action, reward, done = self.memory[0]  # (s, a, r, is_goal)
        next_state, next_action, _, _ = self.memory[1]  # (s', a', _, _)

        # Q(s', a')
        next_q = 0 if done else self.Q[next_state, next_action]  # ゴール時のQ値は0

        target = reward + self.gamma * next_q  # TD-target
        # 指数移動平均で更新
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # ε-greedy で方策π(a|s)を更新
        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)


env = GridWorld()
agent = SarsaAgent()
episodes = 10000

figs: list[Figure] = []
frame_names: list[str] = []
save_rate = 500


for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)  # a ~ π(a|s)
        next_state, reward, done = env.step(action)  # S', R, is_goal

        agent.update(state, action, reward, done)

        if done:
            # 最終ステップを更新するため
            agent.update(next_state, None, None, None)
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
    "SARSA",
    figs,
    str(
        pathlib.Path(__file__).parent
        / "sarsa(on_policy).gif"
    ),
    interval=100,
    show_anim=True,
    frame_names=frame_names,
)
