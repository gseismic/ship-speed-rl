from ship_resistance3 import ship_resistance3
from propeller import propeller
from gear_box import GearBox
from pms import PMS

def V_limit(alpha_S, V_wind, alpha_wind, SOC, V_S_Min, V_S_Max):
    """
    航速范围计算（MATLAB代码转换）
    输入：
        alpha_S - 航向角(°)（未实际使用）
        V_wind - 相对风速(m/s)
        alpha_wind - 相对风向角(°)
        SOC - 电池状态
        V_S_Min - 初始最小航速
        V_S_Max - 初始最大航速
    输出：
        V_S_Min, V_S_Max - 计算后的航速范围
    """
    del_V_S = 0.05  # 航速调整步长

    # 计算最小航速边界
    a = 1
    while a == 1:
        R = ship_resistance3(V_S_Min, alpha_S, V_wind, alpha_wind)
        N_p, Q_p, Esignal1 = propeller(V_S_Min, R)
        
        if Esignal1 == -1:
            V_S_Min += del_V_S
            continue
        elif Esignal1 == 0:
            a = 0

    # 计算最大航速边界
    a = 1
    while a == 1:
        R = ship_resistance3(V_S_Max, alpha_S, V_wind, alpha_wind)
        N_p, Q_p, Esignal1 = propeller(V_S_Max, R)
        
        if Esignal1 == 1:
            V_S_Max -= del_V_S
            continue
        
        N, Q_req = GearBox(N_p, Q_p)
        (model, N_e, Q_e, N_m, Q_m, 
         Q_emin, Q_emax, Esignal2) = PMS(N, Q_req, SOC)
        
        if Esignal2 == 1:
            V_S_Max -= del_V_S
            continue
        
        if Esignal1 == 0 and Esignal2 == 0:
            a = 0

    return V_S_Min, V_S_Max

# 测试案例（对应MATLAB中的示例）
if __name__ == "__main__":
    # [V_S_Min, V_S_Max] = V_limit(216.9, 2.87893908, 70.24121203, 0.8)
    test1 = V_limit(216.9, 2.87893908, 70.24121203, 0.8, 2.2, 5)
    print(f"Test Case 1: V_S_Min={test1[0]:.2f}, V_S_Max={test1[1]:.2f}")
    
    # [V_S_Min, V_S_Max] = V_limit(216.9, 2.87893908, 70.24121203, 0.2)
    test2 = V_limit(216.9, 2.87893908, 70.24121203, 0.2, 2.2, 5)
    print(f"Test Case 2: V_S_Min={test2[0]:.2f}, V_S_Max={test2[1]:.2f}")