import numpy as np
import gymnasium as gym


env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()  # 新 API では reset() が二つ返す
done = False

while not done:
    env.render()
    action = np.random.choice([0, 1])  # ランダム方策
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    print(f"state: {state}, action: {action}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}")
    
    # state: [カートの位置, カートの速度, ポールの角度, ポールの角速度]
    # action: 0: 左へ移動, 1: 右へ移動
    # reward: バランスを保っている間は 1.0
    
    state = next_state

env.close()
