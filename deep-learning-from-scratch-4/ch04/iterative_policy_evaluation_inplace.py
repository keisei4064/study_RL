# 反復方策評価

import matplotlib.pyplot as plt

V = {"L1": 0.0, "L2": 0.0}
new_V = V.copy()

V_history = []

cnt = 0
while True:
    # V_(k+1) = Σ{π(a|s) * 1 * (R + γ * V_k)}
    new_V["L1"] = 0.5 * (-1 + 0.9 * V["L1"]) + 0.5 * (1 + 0.9 * V["L2"])
    new_V["L2"] = 0.5 * (0 + 0.9 * V["L1"]) + 0.5 * (-1 + 0.9 * V["L2"])

    # 小数第4位までにフォーマット
    formatted_V = {k: f"{v:.4f}" for k, v in new_V.items()}
    print(f"cnt: {cnt} V: {formatted_V}")

    # 収束判定
    delta = max(abs(new_V["L1"] - V["L1"]), abs(new_V["L2"] - V["L2"]))  # 更新量最大値
    cnt += 1
    if delta < 0.0001:
        break

    V = new_V.copy()
    V_history.append(V.copy())


print(f"cnt: {cnt} V: {formatted_V}")

# 結果の可視化
plt.plot(range(len(V_history)), [v["L1"] for v in V_history], label="L1")
plt.plot(range(len(V_history)), [v["L2"] for v in V_history], label="L2")
plt.xlabel("cnt")
plt.ylabel("V")
plt.legend()
plt.grid()
plt.show()
