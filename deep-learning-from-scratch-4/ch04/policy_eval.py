# 反復方策評価

if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
import pathlib
from common.gridworld import GridWorld
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import typing
from matplotlib.figure import Figure
from common.save_animation import save_animation_with_steps
import copy


# 式(4.3) p.106 を一回評価
def eval_onestep(
    pi: defaultdict[typing.Any, dict[typing.Any, float]],  # 方策
    V: defaultdict[typing.Any, float],  # 推定状態価値関数
    env: GridWorld,  # 環境（状態遷移関数と報酬関数）
    gamma=0.9,
):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_probs = pi[state]
        new_V = 0
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V
    return V


V_history = []


# 反復評価
def policy_eval(
    pi: defaultdict[typing.Any, dict[typing.Any, float]],  # 方策
    V: defaultdict[typing.Any, float],  # 推定状態価値関数
    env: GridWorld,  # 環境（状態遷移関数と報酬関数）
    gamma,
    threshold=0.001,
):
    V_history.append(V.copy())
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        V_history.append(V.copy())

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break
    return V


# V_history: list[dict]  ← 各ステップのV(s)を格納済み をプロット
def plot_value_history(
    V_history,
    env: GridWorld,
):
    states = env.states()
    n_iter = len(V_history)

    # 状態ごとに収束曲線を描く
    plt.figure(figsize=(8, 5))
    for state in states:
        if state == env.goal_state:
            continue
        values = [V[state] for V in V_history]
        plt.plot(range(n_iter), values, label=f"state {state}")

    plt.xlabel("Iteration")
    plt.ylabel("Value V(s)")
    plt.title("Convergence of Value Function")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# V_historyをヒートマップとしてアニメーション表示
def animate_value_history(V_history, env: GridWorld, interval=100):
    fig_list: list[Figure] = []
    for V in V_history:
        fig = env.render_v(V, pi, show_plt=False)
        fig_list.append(copy.deepcopy(fig))
        plt.close(fig)

    save_animation_with_steps(
        "Iterative Policy Evaluation",
        fig_list,
        str(pathlib.Path(__file__).parent / "iterative_policy_evaluation.gif"),
        interval=interval,
        show_anim=True,
    )


if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9

    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0.0)

    # 反復方策評価
    V = policy_eval(pi, V, env, gamma)

    # 可視化
    # env.render_v(V, pi)
    plot_value_history(V_history, env)
    ani = animate_value_history(V_history, env, interval=150)
