import numpy as np
from .engine import Engine
from .motor import Motor
from .battery_power import Battery_power
from .gas_consumption import Gas_consumption

def ECMS(N, Q_req, SOC):
    # 初始化输出
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
        return model, N, 0, N, 0, Q_emin, Q_emax, Esignal2 
    else: 
        # 惩罚函数参数 
        SOC_min = 0.2 
        SOC_max = 0.8 
        SOC_target = 0.6 
        a = 3  # 惩罚函数比例系数
        Delta_SOC = (SOC_max - SOC_min) / 2
        p_soc = 1 - ((SOC - SOC_target) / Delta_SOC) ** a
        p_soc = max(0.01, min(5, p_soc))
        
        # 等效因子计算 
        LHV = 50  # LNG低热值 MJ/kg 
        s_charge = 2.14 * p_soc * 3.6 / LHV * 1000 
        s_discharge = 3.21 * p_soc * 3.6 / LHV * 1000 
        
        # 发动机搜索范围
        start = max(0, Q_req - Q_mmax)
        stop = min(Q_emax, Q_req + Q_mmax)
        Q_e_list = np.linspace(start, stop, 100).tolist()
        
        # 添加边界值
        if Q_req >= Q_emax:
            Q_e_list.append(Q_emax)
        else:
            Q_e_list.append(Q_req)
        
        best_cost = float('inf')
        Q_e_opt = 0
        Q_m_opt = 0
        
        # 遍历所有可能的发动机扭矩分配
        for Q_e_temp in Q_e_list:
            Q_m_temp = Q_req - Q_e_temp
            
            # 电池功率计算
            P_b = Battery_power(N, Q_m_temp)
            
            # 发动机功率和油耗计算
            W_fuel = Gas_consumption(N, Q_e_temp)
            P_eng = 2 * np.pi * N * Q_e_temp / 60 / 1000  # 转换为kW
            
            # 计算等效消耗
            if P_b >= 0:
                cost = W_fuel * P_eng + s_discharge * P_b
            else:
                cost = W_fuel * P_eng + s_charge * P_b
            
            # SOC预测惩罚
            SOC_pred = SOC - (P_b * time_interval) / E_batt
            if SOC_pred < 0.2 or SOC_pred > 0.8:
                cost += 1e9
            
            # 更新最优解
            if cost < best_cost:
                best_cost = cost
                Q_e_opt = Q_e_temp
                Q_m_opt = Q_m_temp
        
        # 输出结果
        Q_e = Q_e_opt
        Q_m = Q_m_opt
        N_e = N
        N_m = N
        
        # 二次SOC检查
        P_b_1 = Battery_power(N_m, Q_m)
        SOC_pred = (E_batt * SOC - P_b_1 * time_interval) / E_batt
        if SOC_pred < 0.2:
            Esignal2 = 1
        elif SOC_pred > 0.8:
            Esignal2 = -1
        
        # 判断运行模式
        if Q_e > 0 and Q_m > 0:
            model = 1
        elif Q_e > 0 and Q_m == 0:
            model = 2
        elif Q_e == 0 and Q_m > 0:
            model = 4
        elif Q_e > 0 and Q_m < 0:
            model = 3
        else:
            model = 8
        
        return model, N_e, Q_e, N_m, Q_m, Q_emin, Q_emax, Esignal2
