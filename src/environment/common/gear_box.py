def GearBox(N_p: float, Q_p: float) -> tuple[float, float]:
    """
    齿轮箱转速转矩转换计算
    
    根据螺旋桨端参数计算动力源端转速和需求转矩，实现传动系统的参数转换。

    核心计算步骤
        1. 计算推力 T = R/(1-t)   # 考虑推力减额
        2. 计算进速 V_a = (1-w)V_S # 考虑伴流
        3. 通过KT多项式求解进速比J
        4. 由J计算转速 N_p = 60*V_a/(J*D)
        5. 通过KQ多项式计算扭矩 Q_p = KQ*ρ*V_a²D⁵/(2πN_p)

    Parameters
    ----------
    N_p : float
        螺旋桨转速，单位：转/分钟 (rpm)
    Q_p : float
        螺旋桨转矩，单位：牛·米 (N·m)

    Returns
    -------
    tuple[float, float]
        (N, Q_req)
        N : 动力源输出转速，单位：转/分钟 (rpm)
        Q_req : 动力源需求转矩，单位：牛·米 (N·m)

    Notes
    -----
    1. 使用固定减速比i_c=5进行转速放大/转矩缩小
    2. 默认齿轮箱效率为100%（理想状态）
    3. 符合机械传动系统能量守恒原则
    """
    # 齿轮箱特性参数
    gear_ratio = 5.0    # 减速比（输入轴:输出轴 = 1:5）
    efficiency = 1.0    # 传动效率
    
    # 转速转矩转换
    N = N_p * gear_ratio
    Q_req = Q_p / (gear_ratio * efficiency)

    return N, Q_req