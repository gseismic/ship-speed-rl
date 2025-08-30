import math

def ship_resistance3(V_S: float, alpha_S: float, V_wind: float, alpha_wind: float, use_alpha_correction=False) -> float:
    """
    船舶阻力计算器（基于物理公式的阻力模型）
    
    参数：
    V_S : 船舶航速 (m/s)
    alpha_S : 航向角 (度)（本实现中未实际使用）
    V_wind : 真实风速 (m/s) 
    alpha_wind : 真实风向角 (度) 
    
    返回：
    R : 总航行阻力 (牛顿)
    
    计算组成：
    1. 静水阻力
    2. 波浪增阻
    3. 风阻
    """
    
    # 船舶几何参数 
    L = 64.8          # 船长 (m) 
    B = 12.5          # 船宽 (m) 
    L_Bwl = 64.8      # 95%型宽处至船艏距离 (m) 
    A_S = 1164        # 上层建筑侧投影面积 (m²) 
    A_f = 18 * 12     # 上层建筑正投影面积 (m²) 
    
    # 环境参数
    rho_water = 1000  # 水密度 (kg/m³)
    g = 9.8           # 重力加速度 (m/s²)
    rho_air = 1.293   # 空气密度 (kg/m³)
    
    # 1. 静水阻力计算 (经验公式)
    R_calm = 1072 * (V_S**2)
    
    # 2. 波浪增阻计算
    # 有义波高估计公式
    H = 0.015 * (V_wind**2) + 1.5  

    # H_s = 0.22 * (V_wind **2) / g 
    # print(f'{H=}, {H_s=}') 
    # 波浪增阻公式 
    R_wave = 0.0625 * rho_water * g * (H**2) * B * math.sqrt(B / L_Bwl) 
    
    # 3. 风阻计算 
    # 计算相对风速 
    if not use_alpha_correction: 
        alpha = alpha_wind 
    else: 
        alpha = alpha_wind - alpha_S 
    wind_angle_rad = math.radians(alpha)
    V_a = V_S + V_wind * math.cos(wind_angle_rad)
    
    # 风阻系数计算
    C_wind = (2.152 + 
             2 * (-5) * (A_S / (L**2)) + 
             2 * 0.243 * (A_f / (B**2)) + 
             (-0.164) * (L / B))
    
    # 风阻计算公式
    R_wind = 0.5 * C_wind * rho_air * A_f * (V_a**2)
    
    # 风向修正（当风来自船首时增加阻力）
    if math.cos(wind_angle_rad) > 0:
        R_wind = -R_wind
    
    # 总阻力合成
    total_resistance = R_calm + R_wave + R_wind
    # print(f'{R_calm=}, {R_wave=}, {R_wind=}')
    
    return total_resistance