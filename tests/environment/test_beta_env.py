import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.environment.beta.ship_env import ShipEnv
import numpy as np

if __name__ == '__main__':
    # 环境测试代码
    env = ShipEnv()
    env.reset()

    # 示例：验证动作空间采样
    samples = np.array([env.action_space.sample() for _ in range(2000)])
    print(f"均值: {np.mean(samples)}")  # 应接近0.0
    print(f"标准差: {np.std(samples)}")  # 应接近0.577（均匀分布标准差）

    # 运行10步示例
    for i in range(10):
        print('--------------')
        action = env.action_space.sample()
        print(f'速度: {action}')
        state, reward, terminated, truncated, info = env.step(action)
        print(f'步数: {i}, 状态: {state}, 奖励: {reward}, 终止: {terminated}, 截断: {truncated}, 信息: {info}')
        if terminated or truncated:
            break