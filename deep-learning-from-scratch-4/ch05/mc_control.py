import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..")
)  # for importing the parent dirs
import numpy as np
from collections import defaultdict
from common.gridworld import GridWorld
from common.save_animation import save_animation_with_steps
from matplotlib.figure import Figure
import pathlib


def greedy_probs(Q, state, epsilon=0.0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = int(np.argmax(qs))

    # εの確率でランダム行動
    base_prob = epsilon / action_size
    action_probs: dict[int, float] = {
        action: base_prob for action in range(action_size)
    }  # {0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}

    # 1-εの確率で最適行動
    action_probs[max_action] += 1 - epsilon
    return action_probs


class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state, action)

            # 指数移動平均でQ(s,a)を更新
            self.Q[key] += (G - self.Q[key]) * self.alpha

            # 方策π(s)を更新(ε-greedy)
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)


env = GridWorld()
agent = McAgent()

episodes = 10000

figs: list[Figure] = []
frame_names: list[str] = []
save_rate = 500


for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.add(state, action, reward)
        if done:  # ゴールしたらQ(s,a)とπ(s)を更新
            agent.update()
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
    "Monte Carlo Q(s,a) Evaluation\n& Policy Control",
    figs,
    str(pathlib.Path(__file__).parent / "monte_carlo_q_evaluation_and_policy_control.gif"),
    interval=100,
    show_anim=True,
    frame_names=frame_names,
)
