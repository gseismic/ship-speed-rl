

#% 训练脚本 train.py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.environment.gamma.ship_env import ShipEnv
from src.algorithm.utils.sb3 import EpisodeRewardCallback, SaveBestModelCallback 
from src.algorithm.utils.seed import seed_all 
from test_utils import test_model, plot_learning_curves 

def train_model(seed, 
                total_timesteps=3_000_000, 
                lr=3e-4, gamma=1.0, 
                reward_type='scaled', 
                engine_version='v1', 
                regularization_type='none'): 
    seed_all(seed) # 确定种子 
    import os 
    os.makedirs('rewards', exist_ok=True) 
    os.makedirs('models', exist_ok=True) 

    param_tag = f'{lr}_{gamma}_{reward_type}_{engine_version}_{regularization_type}'
    tag = f'gamma_{seed}_{param_tag}' 
    # tag = f'mp_env1_r_pos_time_soc_nm_t2_largenn_vsmall_{seed}_{param_tag}' 

    # 创建并行环境 
    data_file = 'data/Data_Input.xlsx' 
    env = make_vec_env(lambda: ShipEnv(reward_type=reward_type, engine_version=engine_version, regularization_type=regularization_type, data_file=data_file),
                       n_envs=4, seed=seed) # XXX 1 debug 
    # env = make_vec_env(lambda: ShipEnv(eval=True), n_envs=4, seed=seed) # XXX 1 直接训练貌似不能更好？ 
    # env.action_space.seed(seed) 

    eval_env = make_vec_env(lambda: ShipEnv(eval=True, reward_type=reward_type, engine_version=engine_version, data_file=data_file), n_envs=1, seed=seed)

    best_model_path = f"models/{tag}_best" 
    eval_callback = SaveBestModelCallback(eval_env, check_freq=10000, n_eval_episodes=1, model_save_path=best_model_path) 

    # 初始化PPO模型 
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        learning_rate=lr, 
        n_steps=36, # 2048, 这个和dt相关  72/2 = 36 
        batch_size=256, 
        n_epochs=10, 
        gamma=gamma, # 0.995, # 1.0, # 0.99, 
        gae_lambda=0.95, 
        clip_range=0.2, 
        seed=seed,
        policy_kwargs={
            "net_arch": [dict(pi=[512, 512, 256], vf=[512, 512, 256])],
            # "net_arch": [dict(pi=[256, 256, 256], vf=[256, 256, 256])],
            # "activation_fn": nn.ReLU, 
        },
        tensorboard_log="./ppo_ship_log_gamma/"
    )

    # 创建奖励回调 
    reward_callback = EpisodeRewardCallback() # RewardCallback() 
    # 训练模型 
    model.learn(total_timesteps=total_timesteps, 
                callback=[reward_callback, eval_callback],
                progress_bar=False) 

    final_model_path = f'models/{tag}' 
    model.save(final_model_path) # 保存最终模型 

    # 保存奖励曲线 
    rewards = reward_callback.get_rewards() 
    np.save(f'rewards/{tag}.npy', rewards) 

    # 保持奖励曲线的原始图和均值绘图
    plot_learning_curves(rewards, tag=tag)

    return model, reward_callback, final_model_path, best_model_path


import multiprocessing 
from functools import partial 
import traceback
import tqdm

def train_and_test(seed, lr=1e-4, gamma=1.0, reward_type='raw', engine_version='v1', regularization_type='none'):
    """单个种子的训练和测试任务"""
    try:
        # 每个进程独立初始化（避免环境/模型共享冲突）
        model, reward_callback, final_model_path, best_model_path = train_model(
            seed, lr=lr, gamma=gamma, reward_type=reward_type, engine_version=engine_version, 
            regularization_type=regularization_type,
        )
        
        test_env = ShipEnv(eval=True, reward_type=reward_type, engine_version=engine_version, regularization_type=regularization_type, data_file='data/Data_Input.xlsx')
        print(f'Seed {seed} final-model:')
        test_model(test_env, final_model_path, seed, verbose=1)
        print(f'Seed {seed} best-model:')
        test_model(test_env, best_model_path, seed, verbose=1)
        return True
    except Exception as e: 
        print(f"Seed {seed} 训练失败: {str(e)}\n{traceback.format_exc()}") 
        return False

if __name__ == "__main__":
    from joblib import Parallel, delayed 
    import tqdm 
    
    n_jobs = 6  # 并行进程数（建议设置为CPU核心数）
    backend = "loky"  # 多进程后端（可选 "threading" 但PyTorch多线程可能不稳定）
    
    # 创建任务列表 
    seeds = range(10) 
    # reward_types = ['per_nm', 'scaled', 'raw', 'distance_raw'] 
    reward_type = 'raw' 
    # engine_version = ['v1', 'v2'] 
    regularization_type = 'Np' 
    regularization_type = 'Np_Q_Nrpm' 
    regularization_type = 'Np_Nrpm' 
    regularization_type = 'Np' 
    regularization_type = 'Np_SOC' 
    engine_version = 'v1' 
    # engine_version = 'v2' 
    # all_config = list(product(seeds, reward_type, engine_version))
    # task_func = partial(train_and_test, lr=1e-4, gamma=0.1, reward_type='per_nm', engine_version='v1') 
    # task_func = partial(train_and_test, lr=1e-4, gamma=1.0, reward_type='per_nm', engine_version='v2') 
    task_func = partial(train_and_test, lr=3e-4, gamma=1.0, reward_type=reward_type, 
                        engine_version=engine_version, regularization_type=regularization_type) 
    
    # 并行执行（带进度条）
    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
        delayed(task_func)(seed) for seed in tqdm.tqdm(seeds, desc="Processing Seeds")
    ) 

    # task_func = partial(train_and_test, lr=1e-4, gamma=1.0, reward_type='raw', engine_version='v1') 
    # task_func = partial(train_and_test, lr=1e-4, gamma=1.0, reward_type='raw', engine_version='v2') 
    # task_func = partial(train_and_test, lr=1e-4, gamma=1.0, reward_type='scaled', engine_version='v1') 
    # task_func = partial(train_and_test, lr=1e-4, gamma=1.0, reward_type='scaled', engine_version='v2') 
    
    # 分析结果
    # failed_seeds = [seed for (seed, success) in results if not success]
    # print(f"失败种子列表: {failed_seeds}")
