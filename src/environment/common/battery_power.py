import numpy as np
from .battery_power_const import COM_DICT, CON_DICT

def battery_power(N_m: float, Q_m: float) -> float:
    """
    计算电池充放电功率
    
    参数：
        N_m (float): 电机转速 (rpm)
        Q_m (float): 电机转矩 (N·m)
    
    返回：
        float: 电池充/放电功率 (kW)
    """
    if N_m == 0:
        return 0.0

    p_b = 0 # XXX 
    # 系数矩阵和常数项
    COE = np.array([
        [0.0165, 0.0914],
        [-0.0367, 0.1317],
        [0.0740, 0.1048],
        [-0.0422, 0.0091],
        [-0.0680, 0.0693],
        [-0.0156, 0.0728],
        [0.0528, 0.0169],
        [0.0372, 0.1855]
    ])
    COE_C = np.array([-14.0053, 49.1345, -71.4792, 4.3289, 48.6462, 11.6567, -8.0373, -61.7449])

    input_vec = np.array([[N_m], [Q_m]])
    p_b_all = (COE @ input_vec).flatten() + COE_C
    
    # 遍历所有条件寻找满足的插值结果
    for i in range(1, 9):
        con = CON_DICT[i] 
        com = COM_DICT[i] 
        result = (con @ input_vec).flatten() 
        
        if result.shape == com.shape and np.all(result <= com):
            p_b = p_b_all[i-1]
            break
    
    # print(f'{p_b=}')
    # 符号校正逻辑
    if (Q_m > 0 and p_b < 0) or (Q_m < 0 and p_b > 0):
        p_b *= -1
        
    return float(p_b)