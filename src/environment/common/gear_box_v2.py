
def GearBox(N_p, Q_p):
    i_c = 5
    # e_gear = 0.97
    e_gear = 1.0 # 0.97 -> 1.0 
    e_z = 0.9
    N = N_p * i_c
    Q_req = Q_p / (i_c * e_gear * e_z)
    return N, Q_req 
    