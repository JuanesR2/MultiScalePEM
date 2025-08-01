import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('TkAgg')  # o 'Qt5Agg'
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from dataclasses import dataclass

# ------------------------------------------------------------------
# Parámetros generales de simulación
# ------------------------------------------------------------------
SIM_SCALE = 3600.0 / 100.0   # 1 s sim = 36 s real (100 s sim = 1 h real)

# 0) Perfil solar 24 h → 2400 s sim
hours     = np.arange(24)
irr_Wm2   = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   42.10, 118.35, 205.80, 295.40, 352.25, 381.60,
   372.45, 340.10, 258.75, 176.30, 88.40, 15.20,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
], dtype=float)
irr_norm  = irr_Wm2 / 1000.0
P_day     = 10000.0 * irr_norm             # W pico 10 kW
T_END     = 2400.0                         # s sim (24 h)
Ts_slow   = 0.1                            # s sim

t_slow    = np.arange(0, T_END+Ts_slow, Ts_slow)
P_FV      = np.interp(t_slow/100.0, hours, P_day)
Irr_s     = np.interp(t_slow/100.0, hours, irr_norm)

# 1) Nivel 1 – PI corriente (100 kHz → Ts_fast=100µs)
L, VDC, Ts_fast = 200e-6, 150.0, 100e-6
N, V_CELL       = 60, 1.90

@dataclass
class PI1:
    Kp: float; Ki: float; Ts: float; Vdc: float

class FastPI:
    def __init__(self, p): 
        self.p, self.i = p, 0.0
    def update(self, ref, meas):
        e = ref - meas
        self.i += self.p.Ki * self.p.Ts * e
        u = self.p.Kp*e + self.i
        d = np.clip(u/self.p.Vdc, 0.0, 1.0)
        if u != self.p.Vdc*d:  # anti-windup
            self.i -= self.p.Ki * self.p.Ts * e
        return d

pi_fast = FastPI(PI1(0.63, 990.0, Ts_fast, VDC))

# 2) Nivel 2 – Térmico lumped + PI temperatura
C_th, R_th_pass, R_th_cool = 1500.0, 0.50, 0.12
T_amb, T_set, T_crit       = 298.15, 328.15, 338.15
Kp_T, Ki_T                 = 1.2, 0.003
I_int_T, I_int_clamp       = 0.0, 40.0
I_min_hw, I_max_hw, dI_max = 0.0, 95.0, 25.0

# 3) Nivel 3 – MPC supervisor (4 s sim)
Ts_sup = 4.0
H      = 10
# Parámetros dinámicos térmicos
R_th, C = R_th_pass, C_th
a      = 1 - Ts_sup/(R_th*C)
b      = Ts_sup*(0.17*N*V_CELL)/C

# Parámetros H2
i_F = 96485.0
R_Nm3   = 22.414e-3
KG_Nm3  = 0.08988
alpha   = (R_Nm3*KG_Nm3)/(2*i_F)  # kg/(A·s)

# CVXPY setup
I_var    = cp.Variable(H)
T_var    = cp.Variable(H+1)
P_par    = cp.Parameter(H, nonneg=True)
T0_par   = cp.Parameter(nonneg=True)
constraints = [T_var[0] == T0_par]
for k in range(H):
    constraints += [
        I_var[k] >= 0.0, I_var[k] <= I_max_hw,
        N*V_CELL*I_var[k] <= P_par[k],
        T_var[k+1] == a*T_var[k] + b*I_var[k],
        T_var[k+1] <= T_crit
    ]
objective = cp.Maximize(cp.sum(alpha*I_var*Ts_sup))
prob = cp.Problem(objective, constraints)

def solve_mpc(P_slice, T_now):
    P_par.value  = P_slice
    T0_par.value = T_now
    for solver in ('ECOS','OSQP','SCS'):
        try:
            prob.solve(solver=solver, warm_start=True, verbose=False)
            if I_var.value is not None:
                return float(I_var.value[0])
        except:
            pass
    return 0.0

# 4) Simulación offline y logs
t_points  = len(t_slow)
sub_steps = int(Ts_slow / Ts_fast)
steps_sup = int(Ts_sup / Ts_slow)

I           = 0.0
I_ref_fast  = P_FV[0]/(N*V_CELL)
T_stack     = T_amb
H2_tot_kg   = 0.0

I_log       = np.zeros(t_points)
Ir_log      = np.zeros(t_points)
T_log       = np.zeros(t_points)
H2i_log_g   = np.zeros(t_points)
H2tot_log_g = np.zeros(t_points)

for k in range(t_points):
    # MPC supervisor cada Ts_sup
    if k % steps_sup == 0:
        P_slice = P_FV[k:k+H]
        if len(P_slice) < H:
            P_slice = np.pad(P_slice, (0, H-len(P_slice)), 'edge')
        I_max_sup = solve_mpc(P_slice, T_stack)

    # Sub-pasos de corriente
    for _ in range(sub_steps):
        duty = pi_fast.update(I_ref_fast, I)
        V_stk= N*V_CELL
        dI   = (VDC*duty - V_stk)/L
        if I <= 0 and dI < 0:
            dI = 0.0
        I += dI * Ts_fast
        # Acumulación real de H2
        H2_tot_kg += I * alpha * (Ts_fast * SIM_SCALE)

    # PI térmico
    eT = T_set - T_stack
    if abs(eT) < 1.0:
        I_int_T = np.clip(I_int_T + Ki_T*Ts_slow*eT, -I_int_clamp, I_int_clamp)
    dI_T      = Kp_T*eT + I_int_T
    I_target  = np.clip(I_max_sup + dI_T, I_min_hw, I_max_sup)
    delta     = np.clip(I_target - I_ref_fast, -dI_max*Ts_slow, dI_max*Ts_slow)
    I_ref_fast+= delta
    if T_stack < T_set - 1.0:
        I_int_T = 0.0

    # Modelo térmico RC
    P_loss = 0.0 if I < 1e-3 else 0.17 * V_stk * I
    R_eff  = R_th_cool if T_stack > T_set + 1.0 else R_th_pass
    dT     = (P_loss - (T_stack - T_amb)/R_eff) / C_th
    T_stack+= dT * Ts_slow

    # Logging
    I_log[k]    = I
    Ir_log[k]   = I_ref_fast
    T_log[k]    = T_stack - 273.15
    H2i_rate_kgph = I * alpha * 3600.0
    H2i_log_g[k]   = H2i_rate_kgph * 1000.0
    H2tot_log_g[k] = H2_tot_kg * 1000.0

# 5) Gráficos estáticos
fig, (axI, axT, axH) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Corriente, referencia, potencia e irradiancia
axI.plot(t_slow, I_log, 'b', label='I [A]')
axI.plot(t_slow, Ir_log, '--', label='I_ref [A]')
axI.set_ylabel('Corriente [A]')
axI.grid(True)
axI2 = axI.twinx()
axI2.plot(t_slow, P_FV/1000.0, 'k-', alpha=0.6, label='P_FV [kW]')
axI2.plot(t_slow,
          Irr_s, 'k:', alpha=0.8, label='Irradiancia [p.u.]')
axI2.set_ylabel('P_FV [kW] / Irrad.')

# Temperatura
axT.plot(t_slow, T_log, 'r', label='T_stack [°C]')
axT.axhline(T_set-273.15, ls='--', label='T_set')
axT.axhline(T_crit-273.15, ls=':', label='T_crit')
axT.set_ylabel('Temperatura [°C]')
axT.grid(True)
axT.legend()

# H2 en gramos
axHl = axH
axHr = axH.twinx()
axHl.plot(t_slow, H2i_log_g, 'g', label='H2 inst [g/h]')
axHr.plot(t_slow, H2tot_log_g, 'olive', label='H2 acum [g]')
axHl.set_ylabel('Producción [g/h]')
axHr.set_ylabel('Acumulado [g]')
for ax in (axHl, axHr):
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
axHl.grid(True)
axHl.legend(loc='upper left')

axH.set_xlabel('Tiempo [s sim]')
plt.tight_layout()
plt.show()
_