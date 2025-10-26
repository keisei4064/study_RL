if "__file__" in globals():  # パス登録
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from common.gridworld import GridWorld

env = GridWorld()
V = {}
for state in env.states():
    V[state] = np.random.randn()  # 適当な状態価値を設定
env.render_v(V)
