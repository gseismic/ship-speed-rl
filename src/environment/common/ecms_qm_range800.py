import numpy as np
from .engine import Engine 
from .motor import Motor 
from .battery_power import battery_power 
from .gas_consumption import Gas_consumption 

def ECMS_Qm_range800(N, Q_req, SOC, Q_m):
    """
    Q_m作为额外的控制变量 
    Q_req 可以是负的吗? 螺旋桨转速限制了，不能为负
    
    Note: Q_m 这里直接为绝对值，而不是相对值 
    
      - 这里可能更适合把Q_e作为变量，不够的扭矩从电机补？
    """
    Esignal2 = 0 
    
    # 发动机和电机约束 
    Q_emin, Q_emax = Engine(N) # 0, Q_emax 
    # Q_mmax = Motor(N) # -Q_mmax, Q_mmax 
    Q_emin = 0 # XXX NOTE 这里直接指定为0N·m
    Q_mmax = 800 # XXX NOTE 这里直接指定为800N·m
    time_interval = 2  # 环境数据变化的时间间隔 
    
    Q_e = Q_req - Q_m 
    assert Q_req >= 0, f'Q_req must be greater than 0, but got {Q_req}'
    # 边界超限报警 
    if Q_req > Q_emax + Q_mmax or Q_req < 0 or Q_e < 0: 
        Esignal2 = 1 
        print('需求扭矩超限：Esignal2 = 1') 
        # return model, N, 0, N, 0, Q_emin, Q_emax, Q_mmax, Esignal2 
        return N, 0, N, 0, Q_emin, Q_emax, Q_mmax, Esignal2 
    else: 
        # 遍历所有可能的发动机扭矩分配 
        # for Q_e_temp in Q_e_list: 
        # assert Q_req >= 0, f'Q_req must be greater than 0, but got {Q_req}' 
        # Q_m = Q_m_ratio * Q_req 
        # Q_e = Q_req - Q_m 
        
        # 输出结果 
        N_e = N 
        N_m = N 
        
        # 判断运行模式
        # assert Q_e >= 0 and Q_m >= 0
        
        # return model, N_e, Q_e, N_m, Q_m, Q_emin, Q_emax, Q_mmax, Esignal2
        return N_e, Q_e, N_m, Q_m, Q_emin, Q_emax, Q_mmax, Esignal2
