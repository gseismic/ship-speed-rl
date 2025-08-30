def Motor(N):
    """
    电机最大扭矩模型（基于分段线性插值）
    输入：电机转速 N (rpm)
    输出：Q_mmax 对应最大输出扭矩 (N·m)
    """
    if 0 <= N < 866:
        Q_mmax = 849.9900
    elif 866 <= N < 926:
        Q_mmax = -0.1840 * N + 1009.3000
    elif 926 <= N < 1310:
        Q_mmax = -0.4377 * N + 1244.3001
    elif 1310 <= N < 1518:
        Q_mmax = -0.3361 * N + 1111.2000
    elif 1518 <= N < 1800:
        Q_mmax = -0.2486 * N + 978.3002
    else:
        Q_mmax = 0  # 超出范围时默认返回0
    return Q_mmax