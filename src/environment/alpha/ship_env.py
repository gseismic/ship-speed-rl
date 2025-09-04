#%% 环境定义 env.py
import gymnasium as gym 
import gymnasium.spaces as spaces 
import numpy as np
from typing import Tuple, Dict
import copy
import pandas as pd
from ..common.ship_resistance3 import ship_resistance3
from ..common.gear_box import GearBox
from ..common.propeller import propeller
from ..common.pms import PMS 
from ..common.battery_power import battery_power 
from ..common.battery_soc import Battery_SOC
from ..common.gas_consumption import Gas_consumption 
from ..common.propeller_v2 import propeller as propeller_v2
from ..common.ship_resistance3_v2 import ship_resistance3 as ship_resistance3_v2
from ..common.gear_box_v2 import GearBox as GearBox_v2
from ..common.reward_utils import Np_bound_reward, Q_bound_reward, Nrpm_bound_reward

# 相对于ship_env: 添加剩余时间，用于评估不到达风险，使得reward相对于时间可导
# soc 范围扩大， 1.0
# [2.96 2.95 2.98 2.98 2.98 2.98 2.99 3.   3.14 3.08 3.21 3.15 2.94 2.92 
#  2.92 2.94 3.05 3.07 3.09 3.01 2.83 2.65 2.54 2.61 2.66 2.66 2.66 2.66 
#  2.75 2.77 2.81 3.18] 
# Total Fuel Consumption: 1631.30 kg 

# 4.28
#    # def __init__(self, dt=2, soc_min=0.2, soc_max=1.0, v_min=0.1, v_max=10.0, max_time=72, eval=False):
# 1677.68 
# Optimal Speed Sequence:
# [2.92 3.01 2.97 2.93 2.92 2.92 2.93 3.01 3.14 3.11 3.1  3.12 2.9  2.89
#  2.89 2.9  3.09 3.15 3.07 3.06 2.74 2.6  2.58 2.55 2.52 2.51 2.51 2.53
#  2.8  2.84 2.85 2.9  3.11]
# Total Fuel Consumption: 1674.44 kg

class ShipEnv(gym.Env):
    """船舶航行强化学习环境（PPO适配版）"""
    
    def __init__(self, dt=2, soc_min=0.2, 
                 soc_max=1.0, v_min=0.1, v_max=10.0, 
                 max_time=72, eval=False, 
                 data_file='../../data/Data_Input.xlsx', 
                 engine_version='v1',
                 reward_type='raw',
                 regularization_type=None,
                 ):
        """
        Args:
            dt 采样时间，用于计算下一个时刻的位置 
            soc_min, soc_max: SOC的最小值和最大值 
            v_min, v_max: 船舶速度的最小值和最大值 (m/s) 
            max_time: 最大航行时间 (h) 
        """ 
        super(ShipEnv, self).__init__() 
        self.data_file = data_file 
        self.engine_version = engine_version 
        self.reward_type = reward_type 
        self.regularization_type = regularization_type 
               
        # 环境参数
        self.dt = dt 
        self.vmin = v_min 
        self.vmax = v_max 
        self.SOC_low = soc_min 
        self.SOC_high = soc_max 
        self.max_time = max_time 
        self.eval = eval 

        self.dict_data = self._read_data() 
        self.S_nm = self.dict_data['V_S'].iloc[:,0] 
        self.V_alpha = self.dict_data['V_S'].iloc[:, 2] 
        self.V_wind = self.dict_data['V_wind'] 
        self.num_segments = len(self.dict_data['V_wind']) 
        self.max_S_nm = self.S_nm.iloc[-1]
        # 定义动作空间（离散51个动作）
        self.action_space = gym.spaces.Box(low=v_min, high=v_max, shape=(), dtype=np.float64)
        
        self.observation_space = gym.spaces.Dict({
            'soc': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),  # 若为标量可省略 shape 或设为 ()
                dtype=np.float32
            ),
            'position': spaces.Box(
                low=0,
                high=self.max_S_nm,
                shape=(1,),
                dtype=np.float32
            ),
            'time': spaces.Box(
                low=0.0,
                high=self.max_time,
                shape=(1,),
                dtype=np.float32
            ),
            'r_position': spaces.Box(
                low=0.0,
                high=self.max_S_nm,
                shape=(1,),
                dtype=np.float32
            ),
            'r_time': spaces.Box(
                low=0.0,
                high=self.max_time,
                shape=(1,),
                dtype=np.float32
            ),
            # 'V_wind': spaces.Box(
            #     low=0.0,
            #     high=20,
            #     shape=(1,),
            #     dtype=np.float32
            # ),
            # 'alpha_wind': spaces.Box(
            #     low=0.0,
            #     high=360,
            #     shape=(1,),
            #     dtype=np.float32
            # ),
        })
 
        self.state = None 
    
    def _read_data(self):
        input_data = {
            'V_S': pd.read_excel(self.data_file, sheet_name='V_S&alpha_S', engine='openpyxl'),
            'V_wind': pd.read_excel(self.data_file, sheet_name='V_Wind', engine='openpyxl'),
            'alpha_wind': pd.read_excel(self.data_file, sheet_name='alpha_wind', engine='openpyxl')
        } 
        return input_data 

    def reset(self, **kwargs) -> np.ndarray:
        """重置环境状态"""
        super().reset(**kwargs)
        import random
        # initial_V = random.uniform(self.vmin, self.vmax)
        # initial_SOC = 0.8 # random.uniform(self.SOC_low, self.SOC_high) 固定会导致回跳
        # initial_SOC = random.uniform(self.SOC_low, self.SOC_high)  
        # initial_SOC = self.np_random.uniform(self.SOC_low, self.SOC_high) 
        # initial_index = 0 # random.choice(range(self.num_segments)) 
        # 会导致价值模型预测不稳定 
        if not self.eval: 
            n = int(self.max_time/self.dt) 
            time_position_pairs = [(i*self.max_time/n, i*self.max_S_nm/n) for i in range(n)] 
            pair = self.np_random.choice(time_position_pairs) 
            initial_time = pair[0] # float(self.np_random.uniform(0.0, 72.0)) # 当前已经花了多少时间 
            initial_position = pair[1] # float(self.np_random.uniform(0, self.max_S_nm)) 
            initial_SOC = random.uniform(self.SOC_low, self.SOC_high) # 固定会导致回跳 
        else: 
            # initial_time = 0.0 # random.uniform(0.0, 72.0) # 当前已经花了多少时间 
            # initial_position = 0.0 # random.uniform(0, self.max_S_nm) 
            initial_time = 0.0*self.max_time # random.uniform(0.0, 72.0) # 当前已经花了多少时间 
            initial_position = 0.0*self.max_S_nm # random.uniform(0, self.max_S_nm) 
            initial_SOC = 0.8 # random.uniform(self.SOC_low, self.SOC_high) 固定会导致回跳 
        # self.use_remaing_time = False # True 
        self.use_remaing_time = True 
        
        self.initial_position = initial_position 
        self.initial_time = initial_time 
        # print(f'{type(initial_time)=}') 
        # print(f'{type(initial_position)=}') 
        info =  {
            'soc': initial_SOC,
            'position': initial_position, 
            'time': initial_time,
            # 'V': initial_V
        } 

        # self.state = self._build_state(info)
        self.state = {
                'soc': np.array([initial_SOC], dtype=np.float32), 
                'position': np.array([initial_position], dtype=np.float32), 
                'time': np.array([initial_time], dtype=np.float32), # 正order始终需要，因为要索引数据 
                'r_position': np.array([self.max_S_nm - initial_position], dtype=np.float32), 
                'r_time': np.array([self.max_time-initial_time], dtype=np.float32), 
                # 'V_wind': np.array([initial_V_wind], dtype=np.float32), 
                # 'alpha_wind': np.array([initial_alpha_wind], dtype=np.float32),
            }

        self._step = 0
        self._total_fuel = 0
        self._total_fuel_list = []
        self._total_time = 0
        return self.state, info

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        time = self.state['time'][0] 
        SOC = self.state['soc'][0] 
        position = self.state['position'][0] 

        time_index = int(time/2)

        V_ship = action 
        index = np.searchsorted(self.S_nm, position) 

        # V_ship (m/s) 
        del_S_nm = V_ship * self.dt * 3600/1852 # 单位标准化 
        # del_t = del_S * 1852 / V_ship / 3600 
        del_t = self.dt 
        newPosition = position + del_S_nm 
        newTime = time + self.dt 

        out_of_time = False 
        if newTime >= self.max_time: 
            out_of_time = True 

        reach_goal = False 
        if newPosition >= self.max_S_nm: 
            reach_goal = True 
        
        # print(f'{index=}, {time_index=}: {newPosition=}, {newTime=}')

        info = {}
        terminated, truncated = False, False
        if out_of_time and reach_goal:
            reward = 0 
            terminated = True 
        elif out_of_time and not reach_goal:
            reward = -2000 
            truncated = True 
        elif not out_of_time and reach_goal:
            reward = 2000 
            terminated = True
            info = {
                'reward': reward,
                'total_fuel': self._total_fuel,
            }
            # print(f'**{index=}, {action=}, {time_index=}: {newPosition=}, {newTime=}, {self._total_fuel=}, avg={self._total_fuel/(position-self.initial_position)}')
        else: 
            alpha_S = self.V_alpha[index] 
            V_wind = self.dict_data['V_wind'].iloc[index, time_index+1] 
            # print(f'{index=}, {time_index+1=}') 
            alpha_wind = self.dict_data['alpha_wind'].iloc[index, time_index+1] 
            # V_ship = 0.2 
            if self.engine_version == 'v1': 
                RN = ship_resistance3(V_ship, alpha_S, V_wind, alpha_wind) 
            elif self.engine_version == 'v2': 
                RN = ship_resistance3_v2(V_ship, alpha_S, V_wind, alpha_wind) 
            else:
                raise ValueError(f'Invalid engine version: {self.engine_version}') 

            if self.engine_version == 'v1':
                N_p, Q_p, Esignal1 = propeller(V_ship, RN)
            elif self.engine_version == 'v2':
                N_p, Q_p, Esignal1 = propeller_v2(V_ship, RN)
            else:
                raise ValueError(f'Invalid engine version: {self.engine_version}')
            
            if self.engine_version == 'v1':
                N_rpm, Q_req = GearBox(N_p, Q_p)
            elif self.engine_version == 'v2':
                N_rpm, Q_req = GearBox_v2(N_p, Q_p)
            else:
                raise ValueError(f'Invalid engine version: {self.engine_version}') 
            
            # NEW Q_mmax
            _, N_e, Q_e, N_m, Q_m, Q_emin, Q_emax, Q_mmax, Esignal2 = PMS(N_rpm, Q_req, SOC) 

            # print(f'{Q_emin=}, {Q_emax=}, {Q_mmax=}') 
            P_b = battery_power(N_m, Q_m) 
            newSOC, Battery_state, del_t_Out = Battery_SOC(P_b, SOC, del_t) 
            W_fuel = Gas_consumption(N_e, Q_e)

            # 燃料消耗计算 
            del_fuel_consumption = W_fuel * (N_e * Q_e / 9550) * del_t / 1000 
            self._total_fuel += del_fuel_consumption 
            self._total_fuel_list.append(del_fuel_consumption) 
            # print(f'{W_fuel=}, {N_e=}, {Q_e=}, {del_t=}, {del_fuel_consumption=}') 

            # print(f'{SOC=}, {newSOC=}, {newTime=}, {newPosition=}') 
            self.state = {
                'soc': np.array([newSOC], dtype=np.float32),
                'position': np.array([newPosition], dtype=np.float32),
                'time': np.array([newTime], dtype=np.float32),
                'r_position': np.array([self.max_S_nm - newPosition], dtype=np.float32),
                'r_time': np.array([self.max_time-newTime], dtype=np.float32), 
                # 'V_wind': np.array([V_wind], dtype=np.float32), 
                # 'alpha_wind': np.array([alpha_wind], dtype=np.float32), 
            }

            if Esignal1 == -1: 
                reward = -2000 # future: 考虑是直接结束还是，试错一段时间结束 
                terminated = True 
            elif Esignal1 == 1:
                reward = -2000 
                terminated = True
            elif Esignal2 == 1: 
                reward = -2000 
                terminated = True
            else: 
                # reward = - del_fuel_consumption 
                # 跑的公里数 
                distance = 0 # del_S_nm/self.max_S_nm * 1000 
                # 奖励计算 
                # reward = -del_fuel_consumption 
                if not self.eval: 
                    if self.reward_type == 'raw': 
                        reward = - del_fuel_consumption # 单位时间的耗油 
                    elif self.reward_type == 'per_nm': 
                        reward = - del_fuel_consumption/del_S_nm # 单位里程的耗油 
                    elif self.reward_type == 'scaled': 
                        scale = (self.max_S_nm - self.initial_position) / self.max_S_nm 
                        reward = - del_fuel_consumption / scale 
                    elif self.reward_type == 'distance': 
                        reward = distance 
                    
                    if self.regularization_type is not None and self.regularization_type.lower() != 'none': 
                        term = 0 
                        types_ = self.regularization_type.split('_') 
                        if 'Np' in types_: 
                            term += Np_bound_reward(N_p, version=self.engine_version) 
                        if 'Q' in types_: 
                            term += Q_bound_reward(Q_req, Q_emax, Q_mmax) 
                        if 'Nrpm' in types_: 
                            term += Nrpm_bound_reward(N_rpm) 
                        # else:
                        #     raise Exception(f'Unknown regularization_type {self.regularization_type}')
                    
                        # print(f'original: {reward}, {term=}, {types_=}') 
                        reward += term 
                else:
                    reward = -del_fuel_consumption # 单位时间的耗油
                # print(f'{del_fuel_consumption/self.dt=}')
                # if position >= self.max_S_nm: 
                #     terminated = True
                #     reward = 1000
            
            info = {
                'Esignal1': Esignal1, 
                'Esignal2': Esignal2, 
                'soc': newSOC,
                'position': newPosition,
                'time': newTime, 
                'reward': reward, 
                'RN': RN, 
                'delta_fuel': del_fuel_consumption, 
                'total_fuel': self._total_fuel, 
                'total_fuel_list': self._total_fuel_list, 
                'N_p': N_p, 
                'Q_p': Q_p, 
            }
        
        self._total_time += self.dt 
        info['reward'] = reward 
        info['total_time'] = self._total_time 
        
        # if not reach_goal and (terminated or truncated): 
        #     print(f'xx**{index=}, {time_index=}: {newPosition=}, {newTime=}, {self._total_fuel=}, {info=}') 

        self._step += 1 
        return self.state, reward, terminated, truncated, info


if __name__ == '__main__':
    env = ShipEnv()
    env.reset() 

    # 示例：定义对称的动作空间
    # action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    # 采样验证
    samples = np.array([env.action_space.sample() for _ in range(2000)])
    print(f"均值: {np.mean(samples)}")  # 应接近 0.0 
    print(f"标准差: {np.std(samples)}")  # 应接近 0.577（均匀分布标准差）

    for i in range(10):
        print('--------------')
        print(env.action_space)
        # env.reset() 
        action = env.action_space.sample()
        print(f'vel: {action}')
        state, reward, terminated, truncated, info = env.step(action)
        print(f'{i=}, {state=}, {reward=}, {terminated=}, {truncated=}, {info=}')
        # if terminated or truncated:
        #     break
        pass