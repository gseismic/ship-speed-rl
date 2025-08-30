
import math
from scipy.optimize import root_scalar
from .const import KQ_COEFFS, KT_COEFFS

def propeller(V_S, R):
    Esignal1 = 0

    D_p = 1.835
    pd = 0.7
    AeAo = 0.55
    z = 4
    D_sw = 1027

    t = 0.305
    T = R / (1 - t)

    w = 0.22
    V_a = (1 - w) * V_S

    At = KT_COEFFS
    c0, c1, c2, c3 = 0, 0, 0, 0
    for row in At:
        a, j_order, pd_order, aeao_order, z_order = row
        term = a * (pd ** pd_order) * (AeAo ** aeao_order) * (z ** z_order)
        if j_order == 0:
            c0 += term
        elif j_order == 1:
            c1 += term
        elif j_order == 2:
            c2 += term
        elif j_order == 3:
            c3 += term

    def equation(jcoef):
        return c0 + c1*jcoef + c2*jcoef**2 + c3*jcoef**3 - (T / (D_sw * V_a**2 * D_p**2)) * jcoef**2

    sol = root_scalar(equation, bracket=[0, 1.5], method='brentq')
    jcoef_val = sol.root

    N_ps = V_a / (jcoef_val * D_p)
    N_p = 60 * N_ps

    if N_p < 140:
        Esignal1 = -1
    elif N_p > 260:
        Esignal1 = 1

    Aq = KQ_COEFFS
    k_q = 0.0
    for row in Aq:
        a, j_exp, pd_exp, aeao_exp, z_exp = row
        term = a * (jcoef_val ** j_exp) * (pd ** pd_exp) * (AeAo ** aeao_exp) * (z ** z_exp)
        k_q += term

    Q_p = (k_q * 1.025) * D_sw * (N_ps ** 2) * (D_p ** 5)
    return N_p, Q_p, Esignal1
