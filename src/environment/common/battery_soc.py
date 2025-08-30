def Battery_SOC(P_b: float, SOC_In: float, del_t: float) -> tuple: 
    """
    计算电池的下一时刻SOC值并判断电池状态 
    
    该函数根据当前SOC和充放电功率，计算指定时间后的SOC值，
    并根据SOC边界条件调整实际可充放电时间。

    Parameters 
    ---------- 
    P_b : float 
        电池充/放电功率（kW），正值为放电，负值为充电 
    SOC_In : float 
        当前时刻的电池SOC值（0.0~1.0）
    del_t : float 
        充/放电时间间隔（秒）

    Returns
    -------
    tuple (float, int, float)
        (SOC_Out, Battery_state, del_t_Out)
        SOC_Out : float 
            下一时刻的SOC值（0.0~1.0）
        Battery_state : int
            电池状态：-1=放电保护，0=正常，1=充电保护
        del_t_Out : float
            实际可充/放电时间（秒）

    Notes
    ----- 
    1. 当SOC低于20%继续放电时，触发放电保护状态
    2. 当SOC高于80%继续充电时，触发充电保护状态
    3. 实际可充放电时间根据SOC边界条件动态调整
    4. 电池总容量固定为205.6kWh
    """
    E_batt = 205.6  # 电池总容量(kW·h) TODO: 写在const中

    # 计算理论SOC变化
    SOC_Out = (E_batt * SOC_In - P_b * del_t) / E_batt 
    # print(f'{SOC_Out=}, {P_b=}, {del_t=}, {E_batt=}, {del_t*P_b/E_batt=}')

    # 边界条件判断及状态调整
    if SOC_Out < 0.2 and P_b > 0:  # 放电保护
        Battery_state = -1
        del_t_Out = E_batt * (SOC_In - 0.2) / P_b
    elif SOC_Out > 0.8 and P_b < 0:  # 充电保护
        Battery_state = 1
        del_t_Out = E_batt * (SOC_In - 0.8) / P_b
    else:  # 正常工作状态
        Battery_state = 0
        del_t_Out = del_t

    return SOC_Out, Battery_state, del_t_Out