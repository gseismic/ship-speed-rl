import math

def ship_resistance3(V_S, alpha_S, V_wind, alpha_wind):
    L = 64.8
    B = 12.5
    L_Bwl = 64.8
    A_S = 1164
    A_f = 18 * 12
    Rho_w = 1000
    g = 9.8
    Rho_a = 1.293

    R_calm = 1072 * V_S ** 2

    H = 0.015 * V_wind ** 2 + 1.5
    R_wave = 0.0625 * Rho_w * g * H ** 2 * B * math.sqrt(B / L_Bwl)

    V_a = V_S + V_wind * math.cos(math.radians(alpha_wind))
    A0 = 2.152
    A1 = -5
    A2 = 0.243
    A3 = -0.164
    C_wind = A0 + 2 * A1 * A_S / L**2 + 2 * A2 * A_f / B**2 + A3 * L / B
    R_wind = 0.5 * C_wind * Rho_a * A_f * V_a ** 2

    if math.cos(math.radians(alpha_wind)) > 0:
        R_wind = -R_wind

    R = 1.4 * R_calm + 0.74 * R_wave + 1.4 * R_wind
    return R