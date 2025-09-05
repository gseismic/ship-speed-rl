import numpy as np
from .engine import Engine
from .motor import Motor
from .battery_power import battery_power
from .gas_consumption import Gas_consumption

def ECMS_Qm2(N, Q_req, SOC, Q_m_ratio):
    """
    Q_m作为额外的控制变量 
    """
    Esignal2 = 0
    model = 0
    E_batt = 205.6
    
    # 发动机和电机约束
    Q_emin, Q_emax = Engine(N) 
    Q_mmax = Motor(N) 
    time_interval = 2  # 环境数据变化的时间间隔 
    
    # 边界超限报警
    if Q_req > Q_emax + Q_mmax: 
        Esignal2 = 1 
        print('需求扭矩超限：Esignal2 = 1') 
        # return model, N, 0, N, 0, Q_emin, Q_emax, Q_mmax, Esignal2 
        return N, 0, N, 0, Q_emin, Q_emax, Q_mmax, Esignal2
    else: 
        # 惩罚函数参数 
        # SOC_min = 0.2 
        # SOC_max = 0.8 
        # SOC_target = 0.6 
        # a = 3  # 惩罚函数比例系数
        # Delta_SOC = (SOC_max - SOC_min) / 2
        # p_soc = 1 - ((SOC - SOC_target) / Delta_SOC) ** a
        # p_soc = max(0.01, min(5, p_soc)) 
        
        # 遍历所有可能的发动机扭矩分配 
        # for Q_e_temp in Q_e_list:
        assert Q_req >= 0, f'Q_req must be greater than 0, but got {Q_req}'
        Q_m = Q_m_ratio * Q_req
        Q_e = Q_req - Q_m 
        # Q_m_temp = Q_req - Q_e_temp
        
        # 输出结果 
        N_e = N 
        N_m = N 
        
        # 这里不再检查 SOC标记，env中检查，防止检查方式不一致 
        # 二次SOC检查
        # P_b_1 = battery_power(N_m, Q_m) 
        # SOC_pred = (E_batt * SOC - P_b_1 * time_interval) / E_batt
        # if SOC_pred < 0.2: 
        #     Esignal2 = 1 
        # elif SOC_pred > 0.8: 
        #     Esignal2 = -1 
        
        # 判断运行模式
        assert Q_e >= 0 and Q_m >= 0
        # if Q_e > 0 and Q_m > 0: 
        #     model = 1 
        # elif Q_e > 0 and Q_m == 0: 
        #     model = 2 
        # elif Q_e == 0 and Q_m > 0: 
        #     model = 4 
        # elif Q_e > 0 and Q_m < 0: 
        #     model = 3 
        # else: 
        #     model = 8 
        
        # return model, N_e, Q_e, N_m, Q_m, Q_emin, Q_emax, Q_mmax, Esignal2
        return N_e, Q_e, N_m, Q_m, Q_emin, Q_emax, Q_mmax, Esignal2
