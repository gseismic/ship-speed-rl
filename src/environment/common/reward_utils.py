


def N_p_bound(version='v1'):
    if version == 'v1':
        return [140, 300]
    else:
        return [140, 260]

# def N_emin_bound():
#     return [700, 1500] 

# def N_emax_bound():
#     return [700, 1500] 


def N_rpm_bound():
    return [700, 1500] 


def make_bound_reward(v, v_min, v_max, beta, thresh_ratio=0.05):
    v_range = v_max - v_min 
    safe_dist = v_range * thresh_ratio
    if v < v_min + safe_dist:
        dist = v_min + safe_dist - v
    elif v > v_max - safe_dist:
        dist = v - (v_max - safe_dist)
    else: 
        dist = 0
    return - beta * dist**2 


def make_bound_reward_type2(v, v_min, v_max, beta):
    # r = make_bound_reward(v, v_min, v_max, beta, thresh_ratio=0.01) 
    # v_range = v_max - v_min 
    thresh_ratio=0.02 
    r2_beta = 1e-3 
    r = make_bound_reward(v, v_min, v_max, r2_beta, thresh_ratio) 
    if v_min <= v <= v_max:
        return r + 0.0
    elif v < v_min: 
        return r - beta * (v_min - v)**2
    else:
        return r - beta * (v - v_max) ** 2


def Np_bound_reward(N_p, beta1=1e-2, version='v1', barrier_type='barrier'):
    v_min, v_max = N_p_bound(version) 
    if barrier_type == 'barrier':
        return make_bound_reward(N_p, v_min, v_max, beta1)
    else:
        return make_bound_reward_type2(N_p, v_min, v_max, beta1)

def Q_bound_reward(Q_req, Q_emax, Q_mmax, beta2=1e-2, thresh_ratio=0.05, barrier_type='barrier'):
    # v_max = Q_emax + Q_mmax 
    # 快越界时惩罚 
    v_min = Q_emax 
    v_max = Q_emax + Q_mmax
    if barrier_type == 'barrier':
        return make_bound_reward(Q_req, v_min, v_max, beta2, thresh_ratio)
    else:
        return make_bound_reward_type2(Q_req, v_min, v_max, beta2)

def Nrpm_bound_reward(N_rpm, beta=1e-2, thresh_ratio=0.05, barrier_type='barrier'):
    v_min, v_max = N_rpm_bound() 
    if barrier_type == 'barrier':
        return make_bound_reward(N_rpm, v_min, v_max, beta, thresh_ratio)
    else:
        return make_bound_reward_type2(N_rpm, v_min, v_max, beta)
