
# 测试脚本 test.py
import numpy as np
from stable_baselines3 import PPO
import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.environment.beta3.ship_env import ShipEnv
from test_utils import test_model
# 加载训练好的模型
# model = PPO.load("ppo_ship") 

def eval(eval_env, model_path, seed, verbose=0): 
    model = PPO.load(model_path) 
    # 初始化环境 
    obs, _ = eval_env.reset()

    # 运行测试
    x_optimal = [] 
    gas_consumption = 0.0 

    for _ in range(1000): 
        action, _ = model.predict(obs, deterministic=True) 
        x_optimal.append(action) 
        if verbose >= 2: 
            print(f'{obs=}') 
            print(f'{action=}') 
        
        # 获取下一个状态 
        obs, reward, terminated, truncated, info  = eval_env.step(action) 
        if terminated or truncated: 
            if verbose >= 0:
                print(f'{reward=}, {info=}') 
            gas_consumption = info.get('total_fuel', 0.0) 
            break 
        
        # 记录数据（需要修改evaluate3返回燃料消耗）
        # gas_consumption += abs(reward*env.dt)  # 假设reward为负的燃料消耗

    if verbose  >= 0: 
        print(f"Optimal Speed Sequence:\n{np.round(x_optimal, 2)}") 
        print(f"Total Fuel Consumption: {gas_consumption:.2f} kg") 


seeds = range(5) # 20
for seed in seeds:
    print(seed, '-'*30)
    # reward_type = 'per_nm' 
    reward_type = 'raw' 
    engine_version = 'v2' 
    engine_version = 'v3' 
    regularization_type = 'Np' 
    regularization_type = 'Np_SOC' 
    dt = 0.1  
    dt = 0.5 
    dt = 2.0 
    
    test_env = ShipEnv(eval=True, 
                       reward_type=reward_type, 
                       engine_version=engine_version, 
                       regularization_type=regularization_type,
                       dt=dt, 
                       data_file='data/Data_Input.xlsx') 
    # model_path = f'models/mp_env1_r_pos_time_soc_nm_{seed}_0.000_1.0_per_nm_v2_best' 
    # model_path = f'models/mp_env1_r_pos_time_soc_nm_{seed}_0.0003_1.0_per_nm_v1_best' 
    # model_path = f'models/mp_env1_r_pos_time_soc_nm_{seed}_0.0003_1.0_per_nm_{engine_version}' 
    model_path = f'models/mp_env1_r_pos_time_soc_nm_{seed}_0.0003_1.0_per_nm_{engine_version}_Np_Q_Nrpm_best' 
    model_path = f'models/mp_env1_r_pos_time_soc_nm_{seed}_0.0003_1.0_per_nm_{engine_version}_Np_Nrpm_best' 
    model_path = f'models/mp_env1_r_pos_time_soc_nm_{seed}_0.0003_1.0_per_nm_{engine_version}_Np_dt{dt}_best' 
    model_path = f'models/beta2_{seed}_0.0003_1.0_{reward_type}_{engine_version}_Np_SOC_best' 
    # model_path = f'models/mp_env1_r_pos_time_soc_nm_t_{seed}_0.0003_1.0_per_nm_{engine_version}_Np' 

    eval(test_env, model_path, seed, verbose=1) 
