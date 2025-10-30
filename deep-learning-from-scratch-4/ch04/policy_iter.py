if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
from common.gridworld import GridWorld
from ch04.policy_eval import policy_eval
import typing
import pprint


def argmax(d: dict[typing.Any, float]) -> int:
    """d (dict)"""
    # なんか非効率じゃね？
    max_value = max(d.values())
    max_key = -1
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


# greedy化
def greedy_policy(
    V: dict[typing.Any, float],  # 推定状態価値関数
    env: GridWorld,  # 環境（状態遷移関数と報酬関数）
    gamma: float,  # 割引率γ
) -> defaultdict[typing.Any, dict[typing.Any, float]]:
    pi: defaultdict[typing.Any, dict[typing.Any, float]] = {}  # type: ignore

    for state in env.states():  # すべてのsについて，μ'(s)を求める
        action_values = {}

        # 式(4.8) p.112
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values[action] = value

        max_action = argmax(action_values)

        # μ'(s) を π(a|s) の確率分布に変換
        action_probs: dict[int, float] = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi


# 方策反復法
def policy_iter(env: GridWorld, gamma, threshold=0.001, is_render=True):
    pi: defaultdict[typing.Any, dict[typing.Any, float]] = defaultdict(
        lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    )  # ランダム方策
    V = defaultdict(lambda: 0.0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)  # 方策評価
        new_pi = greedy_policy(V, env, gamma)  # greedy化
        
        # メモ: 割引率のおかげで，最短でゴール地点の報酬1を得ようとする方策を獲得できる

        if is_render:
            env.render_v(V, pi)

        if new_pi == pi:  # 方策が変化しなくなったら終了
            print("収束: pi=")
            pprint.pprint(pi)
            env.render_v(V, pi)
            break
        pi = new_pi

    return pi


if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    pi = policy_iter(env, gamma)
