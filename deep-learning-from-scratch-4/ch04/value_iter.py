if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
from common.gridworld import GridWorld
from ch04.policy_iter import greedy_policy
from typing import Any
import pprint
import copy
import pathlib
from matplotlib.figure import Figure
from common.save_animation import save_animation_with_steps


# 式(4.13) p.123
# 価値関数の更新
def value_iter_onestep(
    V: dict[Any, float],  # 推定状態価値関数
    env: GridWorld,
    gamma: float,
) -> dict[Any, float]:
    for state in env.states():  # すべてのsについて
        if state == env.goal_state:  # ゴールの状態価値は0
            V[state] = 0
            continue

        action_values = []
        for action in env.actions():  # sにおける全ての行動aについて
            next_state = env.next_state(state, action)  # 決定論的状態遷移 s'=f(s,a)
            r = env.reward(state, action, next_state)  # 即時報酬
            value = r + gamma * V[next_state]  # ベルマン方程式
            action_values.append(value)

        # greedyな行動a*を選んだ前提で，状態価値関数を更新
        V[state] = max(action_values)
    return V


fig_history: list[Figure] = []


# 価値反復
def value_iter(
    V: dict[Any, float],  # 推定状態価値関数
    env: GridWorld,
    gamma: float,
    threshold=0.001,
    is_render=True,
):
    while True:
        if is_render:
            fig = env.render_v(V)
            fig_history.append(copy.deepcopy(fig))

        old_V = V.copy()

        V = value_iter_onestep(V, env, gamma)  # 更新

        # 収束判定
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break
    return V


if __name__ == "__main__":
    V = defaultdict(lambda: 0.0)  # 推定状態価値の初期値 = all 0
    env = GridWorld()
    gamma = 0.9

    V = value_iter(V, env, gamma)  # 価値反復

    print("推定状態価値関数が収束\n", "V*=")
    pprint.pprint(V)

    pi = greedy_policy(V, env, gamma)  # V*を用いて，最適方策π*を求める
    print("最適方策pi*=")
    pprint.pprint(pi)

    fig = env.render_v(V, pi)
    fig_history.append(copy.deepcopy(fig))

    # アニメーションを保存
    save_animation_with_steps(
        "Value Iteration",
        fig_history,
        str(pathlib.Path(__file__).parent / "value_iteration.gif"),
        interval=500,
    )
