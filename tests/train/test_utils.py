import numpy as np
from stable_baselines3 import PPO
from pathlib import Path  
from loguru import logger 

logger.add('train.log', rotation='10 MB')

def test_model(eval_env, model_path, seed, verbose=0):
    # if not Path(f'{model_path}.zip').exists():
    #     logger.warning(f'{model_path} not exits')
    #     return
    model = PPO.load(model_path) 

    # 初始化环境
    obs, _ = eval_env.reset()

    # 运行测试
    x_optimal = []
    gas_consumption = 0.0

    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        if verbose >= 2:
            logger.debug(f'{obs=}') 
            logger.debug(f'{action=}') 
        # v_norm = action / 50.0  # 反归一化 
        v_real = action # (env.vmax - env.vmin) * v_norm + env.vmin 
        
        # 获取下一个状态
        obs, reward, terminated, truncated, info  = eval_env.step(action)
        if terminated or truncated: 
            if verbose >= 0:
                logger.info(f'{info=}')
            gas_consumption = info['total_fuel']
            break 
        
        # 记录数据（需要修改evaluate3返回燃料消耗）
        x_optimal.append(v_real) 
        # gas_consumption += abs(reward*env.dt)  # 假设reward为负的燃料消耗

    # 保存结果
    path = 'all.txt'
    with open(path, 'a') as f:
        logger.info(f'wriite: {path=}, {model_path=} {seed=}') 
        f.write(f'{model_path=} {seed=}\n') 
        f.write(f'{gas_consumption=}\n') 
        f.write(f'{x_optimal=}\n') 

    if verbose  >= 0:
        logger.info(f"Optimal Speed Sequence:\n{np.round(x_optimal, 2)}")
        logger.info(f"Total Fuel Consumption: {gas_consumption:.2f} kg")


def plot_learning_curves(rewards, tag, window=50):
    # 绘制奖励曲线 
    import os 
    import pandas as pd
    import matplotlib.pyplot as plt 

    avg_rewards = pd.Series(rewards).rolling(window).mean() 
    fig = plt.figure() 
    plt.plot(avg_rewards) 
    plt.title(f'avg Reward={window}') 
    os.makedirs('./images', exist_ok=True) 
    fig.savefig(os.path.join('./images', f"{tag}_reward_MA{window}.png"))

    fig = plt.figure()
    plt.plot(rewards)
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Reward Curve')
    fig.savefig(os.path.join('./images', f"{tag}_reward.png"))
