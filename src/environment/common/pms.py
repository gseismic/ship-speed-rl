from .engine import Engine
from .motor import Motor
from .const import SOC_BOUNDS

def PMS(N: float, Q_req: float, SOC: float) -> tuple[int, float, float, float, float, float, float, int]:
    """
    动力管理系统模式选择与扭矩分配
    
    根据动力系统状态和需求扭矩，选择最优工作模式并分配发动机/电机扭矩。

    Parameters 
    ---------- 
    N : float 
        系统转速，单位：转/分钟 (rpm) 
    Q_req : float 
        总需求扭矩，单位：牛·米 (N·m) 
    SOC : float 
        电池荷电状态，范围：0.0~1.0 

    Returns
    -------
    tuple
        (model, N_e, Q_e, N_m, Q_m, Q_emin, Q_emax, Esignal2) 
        model : 工作模式编号 (1-8) 
        N_e : 发动机实际转速 (rpm)
        Q_e : 发动机输出扭矩 (N·m)
        N_m : 电机实际转速 (rpm)
        Q_m : 电机输出扭矩 (N·m)
        Q_emin : 发动机最小扭矩 (N·m)
        Q_emax : 发动机最大扭矩 (N·m)
        Esignal2 : 异常状态标志 (0/1)

    Notes
    -----
    工作模式说明：
    1: 发动机最大扭矩+电机补充
    2: 纯发动机工作
    3: 高SOC时发动机最大+电机补充
    4: 纯电机工作
    5: 低SOC时发动机最大
    6: 超扭矩需求+低SOC
    7: 超扭矩需求+高SOC
    8: 低SOC时强制发动机工作
    """
    # 系统参数配置
    Q_emin, Q_emax = Engine(N)
    Q_mmax = Motor(N)
    
    # 模式选择逻辑
    if Q_req < Q_emin:
        model = 8 if SOC <= SOC_BOUNDS[0] else (4 if Q_req <= Q_mmax else 8) 
    elif Q_emin <= Q_req <= Q_emax: 
        model = 3 if SOC < SOC_BOUNDS[1] else 2 
    elif Q_emax < Q_req <= (Q_emax + Q_mmax): 
        model = 5 if SOC <= SOC_BOUNDS[0] else 1 
    else:  # 超最大能力范围
        model = 6 if SOC <= SOC_BOUNDS[0] else 7 

    # 扭矩分配逻辑 
    Esignal2 = 0 
    N_e, Q_e, N_m, Q_m = 0.0, 0.0, 0.0, 0.0 
    # print(f'{SOC=}, {model=}, {Q_emin=}, {Q_emax=}, {Q_mmax=}, {Q_req=}, {SOC=}')
    # print(f'{SOC=}, {model=}') 
    if model == 1 or model == 3:  # 发动机最大+电机补充 
        Q_e, N_e = min(Q_emax, Q_req), N # FIXED 
        # Q_m, N_m = Q_req - Q_e, N 
        Q_m, N_m = Q_req - Q_e, N 
    elif model == 2 or model == 8:  # 纯发动机工作 
        Q_e, N_e = (Q_req, N) if model == 8 else (Q_req, N)
        N_m, Q_m = 0.0, 0.0
    elif model == 4:  # 纯电机工作
        Q_m, N_m = Q_req, N
    elif model == 5 or model == 6:  # 发动机最大+异常标志
        Q_e, N_e = Q_emax, N
        Esignal2 = 1
    elif model == 7:  # 超扭矩复合模式
        Q_e, N_e = Q_emax, N
        Q_m, N_m = Q_mmax, N
        Esignal2 = 1

    print(f'{model=}, {Q_m=}, {Q_req=}')
    return (model, N_e, Q_e, N_m, Q_m, Q_emin, Q_emax, Q_mmax, Esignal2)