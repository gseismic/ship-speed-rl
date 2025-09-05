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
from ..common.ecms import ECMS
from ..common.ecms_qm2 import ECMS_Qm2
from ..common.battery_power import battery_power
from ..common.battery_soc import Battery_SOC
from ..common.gas_consumption import Gas_consumption
from ..common.propeller_v2 import propeller as propeller_v2
from ..common.ship_resistance3_v2 import ship_resistance3 as ship_resistance3_v2
from ..common.gear_box_v2 import GearBox as GearBox_v2 
from ..common.reward_utils import Np_bound_reward, Q_bound_reward, Nrpm_bound_reward, SOC_bound_reward, make_bound_reward

# from beta/..t2m.py
class ShipEnv(gym.Env):
    """船舶航行强化学习环境（PPO适配版）
    
    ChangeLog:
        - 2025-09-02: 删除future_avg_V_wind和future_avg_alpha_wind
    """

    def __init__(
        self,
        dt=2,
        soc_min=0.2,
        soc_max=0.8,
        v_min=2.5, #0.1,
        v_max=3.5, #50.0,
        max_time=72,
        eval=False,
        data_file='data/Data_Input.xlsx',
        engine_version='v1',
        reward_type='raw',
        regularization_type=None,
        # avg_window_sizes=[3, 5, 10],
    ):
        """
        初始化环境参数和空间。

        Args:
            dt: 采样时间（秒），用于计算下一个时刻的位置。
            soc_min: SOC的最小值。
            soc_max: SOC的最大值。
            v_min: 船舶速度的最小值（m/s）。
            v_max: 船舶速度的最大值（m/s）。
            max_time: 最大航行时间（小时）。
            eval: 是否为评估模式（布尔值）。
            data_file: 数据文件路径。
            engine_version: 引擎版本（'v1'或'v2'）。
            reward_type: 奖励类型（'raw', 'per_nm', 'scaled'）。
            regularization_type: 正则化类型（字符串，例如'Np_Q_Nrpm'）。
            avg_window_sizes: 平均窗口大小列表，用于计算未来风速和风向。
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

        # 读取数据
        self.dict_data = self._read_data()
        self.S_nm = self.dict_data['V_S'].iloc[:, 0]
        self.V_alpha = self.dict_data['V_S'].iloc[:, 2]
        self.V_wind = self.dict_data['V_wind']
        self.alpha_wind = self.dict_data['alpha_wind']
        self.num_segments = len(self.dict_data['V_wind'])
        self.max_S_nm = self.S_nm.iloc[-1]
        # self.avg_window_sizes = avg_window_sizes
        
        # 定义动作空间（连续速度值）
        Q_m_ratio_min = 0
        Q_m_ratio_max = 1.0 
        self.action_space = gym.spaces.Box(
            low=np.array([v_min, Q_m_ratio_min]), 
            high=np.array([v_max, Q_m_ratio_max]), shape=(2,), dtype=np.float64
        )

        # 定义观察空间（字典形式）
        # num_windows = len(avg_window_sizes) * 3  # 未来3个航段的多个时间窗口
        self.observation_space = gym.spaces.Dict({
            'soc': spaces.Box(
                low=0.0,
                high=1.0,
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
            'V_wind': spaces.Box(
                low=0.0,
                high=20,
                shape=(1,),
                dtype=np.float32
            ),
            'alpha_wind': spaces.Box(
                low=0.0,
                high=360,
                shape=(1,),
                dtype=np.float32
            ),
            # 'future_avg_V_wind': spaces.Box(
            #     low=0.0,
            #     high=20,
            #     shape=(num_windows,),
            #     dtype=np.float32
            # ),
            # 'future_avg_alpha_wind': spaces.Box(
            #     low=0.0,
            #     high=360,
            #     shape=(num_windows,),
            #     dtype=np.float32
            # ),
        })

        self.state = None 
        self._step = 0 
        self._total_fuel = 0 
        self._total_fuel_list = [] 
        self._total_time = 0 
        self.use_remaing_time = True  # 使用剩余时间标志

    def _read_data(self):
        """从Excel文件读取数据。"""
        input_data = {
            'V_S': pd.read_excel(self.data_file, sheet_name='V_S&alpha_S', engine='openpyxl'),
            'V_wind': pd.read_excel(self.data_file, sheet_name='V_Wind', engine='openpyxl'),
            'alpha_wind': pd.read_excel(self.data_file, sheet_name='alpha_wind', engine='openpyxl')
        }
        return input_data

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """重置环境状态，返回初始观察和info。"""
        super().reset(**kwargs)
        import random

        if not self.eval:
            # 训练模式：随机初始化时间、位置和SOC
            n = int(self.max_time / self.dt) # 最多有多少个段 
            time_position_pairs = [ 
                (i * self.max_time / n, i * self.max_S_nm / n) for i in range(n)
            ] 
            pair = self.np_random.choice(time_position_pairs) 
            initial_time = pair[0]  # 初始时间（小时）
            initial_position = pair[1]  # 初始位置（海里）
            initial_SOC = random.uniform(self.SOC_low, self.SOC_high)  # 随机SOC

            # 计算数据索引 
            time_index = int(initial_time / self.dt) # FIXED 2 -> dt 
            index = np.searchsorted(self.S_nm, initial_position)
            initial_V_wind = self.V_wind.iloc[index, time_index + 1]
            initial_alpha_wind = self.alpha_wind.iloc[index, time_index + 1]
        else:
            # 评估模式：固定初始值
            time_index = 0
            index = 0
            initial_time = 0.0
            initial_position = 0.0
            initial_SOC = 0.8
            initial_V_wind = self.V_wind.iloc[index, time_index + 1]
            initial_alpha_wind = self.alpha_wind.iloc[index, time_index + 1]
            # print(f'初始风速: {initial_V_wind}')
            # print(f'初始风向: {initial_alpha_wind}')

        # 计算未来平均风速和风向
        # future_v_avgs, future_alpha_avgs = self._compute_avg_wind(
        #     initial_V_wind, initial_alpha_wind, time_index, index, self.avg_window_sizes
        # )

        # 保存初始状态
        self.initial_position = initial_position
        self.initial_time = initial_time
        self.time = initial_time
        self.position = initial_position

        # 构建状态字典
        self.state = {
            'soc': np.array([initial_SOC], dtype=np.float32),
            'position': np.array([initial_position], dtype=np.float32),
            'time': np.array([initial_time], dtype=np.float32),
            'r_position': np.array([self.max_S_nm - initial_position], dtype=np.float32),
            'r_time': np.array([self.max_time - initial_time], dtype=np.float32),
            'V_wind': np.array([initial_V_wind], dtype=np.float32),
            'alpha_wind': np.array([initial_alpha_wind], dtype=np.float32),
            # 'future_avg_V_wind': np.array(future_v_avgs, dtype=np.float32),
            # 'future_avg_alpha_wind': np.array(future_alpha_avgs, dtype=np.float32),
        }

        # 重置内部计数器
        self._step = 0
        self._total_fuel = 0
        self._total_fuel_list = []
        self._total_time = 0

        # 返回状态和info（深拷贝以避免修改）
        state = copy.deepcopy(self.state)
        del state['position']
        del state['time']
        info = {
            'soc': initial_SOC,
            'position': initial_position,
            'time': initial_time,
            'V_wind': initial_V_wind,
            'alpha_wind': initial_alpha_wind
        }
        return state, info

    def _compute_avg_wind(self, initial_V_wind, initial_alpha_wind, time_index, index, avg_window_sizes):
        """计算未来多个窗口的平均风速和风向。"""
        indices = [index, index + 1, index + 2]  # 未来3个航段
        future_v_avgs = []
        future_alpha_avgs = []
        for idx in indices:
            sub_v_avgs, sub_alpha_avgs = self._compute_avg_wind_sub(
                initial_V_wind, initial_alpha_wind, time_index, idx, avg_window_sizes
            )
            future_v_avgs.extend(list(sub_v_avgs.values()))
            future_alpha_avgs.extend(list(sub_alpha_avgs.values()))
        return future_v_avgs, future_alpha_avgs

    def _compute_avg_wind_sub(self, initial_V_wind, initial_alpha_wind, time_index, index, avg_window_sizes):
        """计算单个航段未来多个窗口的平均风速和风向。"""
        future_v_avgs = {}
        future_alpha_avgs = {}
        current_col = time_index + 1  # 当前数据列
        max_col = self.V_wind.shape[1] - 1  # 最大列索引

        for ws in avg_window_sizes:
            start_col = current_col + 1
            end_col = current_col + ws
            start = max(start_col, 0)
            end = min(end_col, max_col)

            if start > end:
                # 无未来数据，使用当前值
                avg_v = initial_V_wind
                avg_alpha = initial_alpha_wind
            else:
                # 收集有效数据
                valid_cols = range(start, end + 1)
                if index < self.V_wind.shape[0]:
                    v_values = [self.V_wind.iloc[index, c] for c in valid_cols]
                    alpha_values = [self.alpha_wind.iloc[index, c] for c in valid_cols]
                    avg_v = np.mean(v_values) if v_values else initial_V_wind
                    avg_alpha = np.mean(alpha_values) if alpha_values else initial_alpha_wind
                else:
                    avg_v = initial_V_wind
                    avg_alpha = initial_alpha_wind
            future_v_avgs[ws] = avg_v
            future_alpha_avgs[ws] = avg_alpha
        return future_v_avgs, future_alpha_avgs

    def step(self, actions) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行动作，返回新状态、奖励、终止标志、截断标志和info。"""
        assert self.engine_version == 'v3', 'Engine must be v3'
        action = actions[0] 
        Q_m_ratio = actions[1] 
        time = self.state['time'][0] 
        SOC = self.state['soc'][0] 
        position = self.state['position'][0] 
        time_index = int(time / 2) # FIXED 2 -> dt 
        V_ship = action  # 船舶速度（m/s）
        index = np.searchsorted(self.S_nm, position) 
        
        # print(f'{Q_m_ratio=}, {V_ship=}') 

        # 计算新位置和时间 
        del_S_nm = V_ship * self.dt * 3600 / 1852  # 转换为海里
        del_t = self.dt  # 时间步长（小时）
        newPosition = position + del_S_nm
        newTime = time + self.dt
        reason = 'none'

        # 初始化标志和奖励
        terminated = False
        truncated = False
        goal_reward = 0
        reach_goal = False
        out_of_time = False
        remaining_nm = 0

        # 检查是否到达目标或超时
        if newTime >= self.max_time:
            # 超时处理
            del_t = self.max_time - time
            del_S_nm = V_ship * del_t * 3600 / 1852
            newPosition = position + del_S_nm
            if newPosition >= self.max_S_nm:
                # 到达目标
                del_S_nm = self.max_S_nm - position
                del_t = del_S_nm * 1852 / V_ship / 3600
                reach_goal = True
            else:
                # 未到达目标
                out_of_time = True
                remaining_nm = self.max_S_nm - newPosition
                reason = 'out_of_time'
        else:
            # 未超时
            if newPosition >= self.max_S_nm:
                # 到达目标
                del_S_nm = self.max_S_nm - position
                del_t = del_S_nm * 1852 / V_ship / 3600
                reach_goal = True
            else:
                # 未到达目标，继续 
                remaining_nm = self.max_S_nm - newPosition
                # reason = 'not_reach_goal'

        # 根据是否到达目标设置奖励和终止标志
        if reach_goal:
            goal_reward = 2000
            terminated = True 
        elif out_of_time:
            # goal_reward = -2000 * (remaining_nm / self.max_S_nm) ** 2 - 2000
            goal_reward = 0 # -2000 * (remaining_nm / self.max_S_nm) + 2000
            terminated = True 

        # 获取当前风速和风向 
        alpha_S = self.V_alpha[index]  # 未使用
        V_wind = self.V_wind.iloc[index, time_index + 1]
        alpha_wind = self.alpha_wind.iloc[index, time_index + 1]

        # 计算未来平均风速和风向
        # future_v_avgs, future_alpha_avgs = self._compute_avg_wind(
        #     V_wind, alpha_wind, time_index, index, self.avg_window_sizes
        # )

        # 计算船舶阻力
        if self.engine_version == 'v1':
            RN = ship_resistance3(V_ship, alpha_S, V_wind, alpha_wind)
        elif self.engine_version in ['v2', 'v3']:
            RN = ship_resistance3_v2(V_ship, alpha_S, V_wind, alpha_wind)
        else:
            raise ValueError(f'无效引擎版本: {self.engine_version}')

        # 计算螺旋桨参数
        if self.engine_version == 'v1':
            N_p, Q_p, Esignal1 = propeller(V_ship, RN)
        elif self.engine_version in ['v2', 'v3']:
            N_p, Q_p, Esignal1 = propeller_v2(V_ship, RN)
        else:
            raise ValueError(f'无效引擎版本: {self.engine_version}')

        # 计算齿轮箱参数 
        if self.engine_version == 'v1': 
            N_rpm, Q_req = GearBox(N_p, Q_p) 
        elif self.engine_version in ['v2', 'v3']: 
            N_rpm, Q_req = GearBox_v2(N_p, Q_p) 
        else: 
            raise ValueError(f'无效引擎版本: {self.engine_version}')

        # 计算PMS参数 
        if self.engine_version == 'v1': 
            _, N_e, Q_e, N_m, Q_m, Q_emin, Q_emax, Q_mmax, Esignal2 = PMS(N_rpm, Q_req, SOC)
        elif self.engine_version == 'v2': 
            _, N_e, Q_e, N_m, Q_m, Q_emin, Q_emax, Q_mmax, Esignal2 = ECMS(N_rpm, Q_req, SOC)            
        elif self.engine_version == 'v3': 
            # Esignal2 = 0 # ratio下 始终合法
            N_e, Q_e, N_m, Q_m, Q_emin, Q_emax, Q_mmax, Esignal2 = ECMS_Qm2(
                N_rpm, Q_req, SOC, Q_m_ratio) 
        else:
            raise ValueError(f'无效引擎版本: {self.engine_version}')

        # 计算电池功率和SOC 
        P_b = battery_power(N_m, Q_m) 
        newSOC, Battery_state, del_t_Out = Battery_SOC(P_b, SOC, del_t)
        print(f'{Q_req=}, {Q_m_ratio=}, {SOC=}, {newSOC=}, {P_b=}, {N_m=}, {N_e=}, {Q_e=}, {Q_m=}')

        # 计算燃料消耗 
        W_fuel = Gas_consumption(N_e, Q_e) 
        del_fuel_consumption = W_fuel * (N_e * Q_e / 9550) * del_t / 1000
        # print(f'{W_fuel=}, {N_e=}, {Q_e=}, {del_t=}, {del_fuel_consumption=}')
        self._total_fuel += del_fuel_consumption 
        self._total_fuel_list.append(del_fuel_consumption) 

        # 初始化奖励
        reward = 0
        
        # 检查Q_e是否合理
        # Q_e__check = Q_req - Q_m
        # if Q_e__check < 0:
        #     print(f'Q_e__check < 0: {Q_e__check=}')
        # if Q_e__check < 0:
        #     reward += -5000
        #     terminated = True
        # elif Q_e__check > Q_emax:
        #     reward += -5000
        #     terminated = True
        # else:
        #     reward += -del_fuel_consumption
        
        
        Q_req_reward = make_bound_reward(Q_req, Q_emin, Q_emax + Q_mmax, 1e-2, thresh_ratio=0.05)
        print(f'**{Q_req=}, {Q_emin=}, {Q_emax + Q_mmax=}, {Q_req_reward=}')
        reward += Q_req_reward 

        # 添加正则化奖励（如果启用）
        if self.regularization_type is not None and self.regularization_type.lower() != 'none':
            term = 0 
            types_ = self.regularization_type.split('_') 
            # print(types_) 
            # raise 
            soc_regl = 0
            if 'Np' in types_: 
                term += Np_bound_reward(N_p, version=self.engine_version)
            if 'Q' in types_: 
                term += Q_bound_reward(Q_req, Q_emax, Q_mmax) 
            if 'Nrpm' in types_: 
                term += Nrpm_bound_reward(N_rpm) 
            if 'SOC' in types_: 
                # term += SOC_bound_reward(SOC, self.SOC_low, self.SOC_high)
                soc_regl = SOC_bound_reward(newSOC, self.SOC_low, self.SOC_high)
            if not self.eval:
                reward += term
            
            reward += soc_regl
            print(f'{soc_regl=}')

        # 处理信号错误
        BIG_PENALTY = -5000
        if Esignal1 == -1 or Esignal1 == 1 or Battery_state != 0:
            # NEW: Battery_state 大惩罚
            reward += BIG_PENALTY
            terminated = True
        elif Esignal2 == 1:
            reward += BIG_PENALTY
            terminated = True
        else:
            # 根据奖励类型计算奖励
            if not self.eval:
                if self.reward_type == 'raw':
                    reward += -del_fuel_consumption
                elif self.reward_type == 'per_nm':
                    reward += -del_fuel_consumption / del_S_nm
                elif self.reward_type == 'scaled':
                    scale = (self.max_S_nm - self.initial_position) / self.max_S_nm
                    reward += -del_fuel_consumption / scale
            else:
                reward += -del_fuel_consumption

        # 更新状态
        self.state = {
            'soc': np.array([newSOC], dtype=np.float32),
            'position': np.array([newPosition], dtype=np.float32),
            'time': np.array([newTime], dtype=np.float32),
            'r_position': np.array([self.max_S_nm - newPosition], dtype=np.float32),
            'r_time': np.array([self.max_time - newTime], dtype=np.float32),
            'V_wind': np.array([V_wind], dtype=np.float32),
            'alpha_wind': np.array([alpha_wind], dtype=np.float32),
            # 'future_avg_V_wind': np.array(future_v_avgs, dtype=np.float32),
            # 'future_avg_alpha_wind': np.array(future_alpha_avgs, dtype=np.float32),
        }

        # 添加目标奖励
        reward += goal_reward

        # 更新总时间
        self._total_time += del_t # FIXED self.dt

        # 构建info字典
        info = {
            'succ': reach_goal,
            'reason': reason,
            'Esignal1': Esignal1,
            'Esignal2': Esignal2,
            'soc': newSOC,
            'position': newPosition,
            'time': newTime,
            'reward': reward,
            'RN': RN,
            'Battery_state': Battery_state,
            'delta_fuel': del_fuel_consumption,
            'total_fuel': self._total_fuel,
            'total_fuel_list': self._total_fuel_list,
            'N_p': N_p,
            'Q_p': Q_p,
            'total_time': self._total_time,
        }

        # 准备返回状态（移除位置和时间）
        state = copy.deepcopy(self.state)
        del state['position']
        del state['time']

        self._step += 1
        return state, reward, terminated, truncated, info
