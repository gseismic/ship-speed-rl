import numpy as np
from .const import KQ_COEFFS, KT_COEFFS

def propeller(V_S: float, R: float) -> tuple[float, float, int]:
    """
    螺旋桨水动力性能计算器
    
    参数：
    V_S : 船舶航速 (m/s)，需 > 0
    R : 船舶阻力 (N)，需 > 0
    
    返回：
    (N_p, Q_p, Esignal1)
    N_p : 螺旋桨转速 (rpm)
    Q_p : 螺旋桨转矩 (N·m)
    Esignal1 : 状态标志 [-1:转速过低, 0:正常, 1:转速过高]
    """
    # 参数校验
    if V_S <= 0 or R <= 0:
        raise ValueError("输入参数必须大于零") 

    # print(f'{V_S=}, {R=}') 
    # 船舶参数配置 
    D_p = 2.06    # 螺旋桨直径 (m) 
    pd = 0.7      # 螺距比 
    AeAo = 0.55   # 盘面比 
    z = 4         # 叶片数 
    t = 0.305     # 推力减额系数 
    w = 0.22      # 伴流分数 
    rho = 1027    # 海水密度 (kg/m³) 

    # 计算推力和进速
    T = R / (1 - t)
    V_a = (1 - w) * V_S  # 进速 (m/s)

    # 计算KT多项式系数
    c0, c1, c2, c3 = 0.0, 0.0, 0.0, 0.0
    for coeff, j_exp, pd_exp, aeao_exp, z_exp in KT_COEFFS:
        term = coeff * (pd**pd_exp) * (AeAo**aeao_exp) * (z**z_exp)
        if j_exp == 0:   c0 += term
        elif j_exp == 1: c1 += term
        elif j_exp == 2: c2 += term
        elif j_exp == 3: c3 += term

    # 构建三次方程系数
    k = T / (rho * V_a**2 * D_p**2)
    coeffs = [c3, (c2 - k), c1, c0]  # 方程形式: c3*J^3 + (c2-k)J^2 + c1*J + c0 = 0 

    # 数值求解三次方程
    roots = np.roots(coeffs) 
    
    # 筛选有效实根 (0 <= J <= 1.5)
    valid_roots = []
    for root in roots:
        if np.isreal(root): 
            real_root = root.real 
            if 0 <= real_root <= 1.5: 
                valid_roots.append(real_root) 
    
    if not valid_roots:
        raise ValueError("无有效进速系数解，请检查输入参数")

    J = max(valid_roots)  # 选择最大实根

    # 计算螺旋桨转速
    N_ps = V_a / (J * D_p)          # 转/秒
    N_p = 60 * N_ps                  # 转/分钟
    Esignal1 = 0

    # 转速状态检查 
    if N_p < 140:    Esignal1 = -1
    elif N_p > 300:  Esignal1 = 1

    # 计算KQ值 
    KQ = 0.0 
    for coeff, j_exp, pd_exp, aeao_exp, z_exp in KQ_COEFFS:
        term = coeff * (J**j_exp) * (pd**pd_exp) * (AeAo**aeao_exp) * (z**z_exp)
        KQ += term

    # 计算螺旋桨转矩
    Q_p = KQ * rho * (N_ps**2) * (D_p**5)

    # 结果舍入处理
    return round(N_p, 2), round(Q_p, 2), Esignal1


if __name__ == "__main__":
    test_cases = [
        (5.0, 150900.820556087225),  
        (5.0, 18093.820556087225),  
        (10.0, 18093.82055608722),  
        (0.5, 18093.82055608722) 
    ]
    
    for V_S, R in test_cases:
        try:
            N_p, Q_p, flag = propeller(V_S, R)
            print(f"V_S={V_S}m/s, R={R}N => N_p={N_p}rpm, Q_p={Q_p}N·m, Flag={flag}")
        except Exception as e:
            print(f"Error: {str(e)}")