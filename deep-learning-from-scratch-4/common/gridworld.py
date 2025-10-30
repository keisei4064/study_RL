if "__file__" in globals():  # パス登録
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import matplotlib.figure
import numpy as np
import common.gridworld_render as render_helper


class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        self.reward_map = np.array(
            [
                [0, 0, 0, 1.0],
                [0, None, 0, -1.0],
                [0, 0, 0, 0],
            ]
        )
        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        self.start_state = (2, 0)
        self.agent_state = self.start_state

    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

    def actions(self):
        return self.action_space

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    # 状態遷移関数（決定論的）
    def next_state(self, state, action) -> tuple[int, int]:
        # 行動に対応する移動量オフセット
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]

        # 移動
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        # 壁の当たり判定
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state

        # 次の状態を返す
        return next_state

    # 報酬関数 r(s, a, s')
    def reward(self, state, action, next_state) -> float:
        return float(self.reward_map[next_state])

    # ゲームを初期状態に戻す
    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    # 1ステップ分，エージェントに行動させる
    def step(self, action) -> tuple[tuple[int, int], float, bool]:
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = next_state == self.goal_state  # ゴール判定

        self.agent_state = next_state
        return next_state, reward, done  # (次の状態, 報酬, ゲーム終了フラグ)

    # 現在の推定状態価値関数:V(s) を描画
    def render_v(self, v=None, policy=None, print_value=True) -> matplotlib.figure.Figure:
        renderer = render_helper.Renderer(
            self.reward_map, self.goal_state, self.wall_state
        )
        renderer.render_v(v, policy, print_value)
        
        return renderer.fig   # type: ignore

    # 現在の推定行動価値関数:Q(s, a) を描画
    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(
            self.reward_map, self.goal_state, self.wall_state
        )
        renderer.render_q(q, print_value)


if __name__ == "__main__":
    env = GridWorld()

    print(f"{env.height=}")
    print(f"{env.width=}")
    print(f"{env.shape=}")

    print("---")

    print("actions:")
    for action in env.actions():
        print("\t", action)

    print("---")

    print("states:")
    for state in env.states():
        print("\t", state)
