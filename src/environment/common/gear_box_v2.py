
def GearBox(N_p, Q_p):
    """
    输入参数​：
    •
    N_p：螺旋桨转速（单位：RPM）
    •
    Q_p：螺旋桨扭矩（单位：kN·m）
    ​输出结果​：
    •
    N：主机（发动机）输出转速（单位：RPM）
    •
    Q_req：主机所需输出扭矩（单位：kN·m）
    """
    i_c = 5 # 传动比 i_c=5​
    # e_gear = 0.97
    e_gear = 1.0 # 0.97 -> 1.0 e_gear=1.0：理想齿轮箱（实际工程中通常取0.97-0.99）
    e_z = 0.9 # e_z=0.9：包含轴承摩擦/轴系振动的损耗（实际值依系统设计而定） 
    N = N_p * i_c
    Q_req = Q_p / (i_c * e_gear * e_z)
    return N, Q_req 
    