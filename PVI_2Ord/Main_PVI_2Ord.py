'''
Código construido para solucionar un PVI conformado por una EDO de segundo orden de la forma y'' = fld(t,y) y condiciones iniciales y(t_0)=y_0; y'(t_0)=y'_0.
'''

import met_solpvi_2ord as pvim2
import plot_pvi2 as p

# Solución PVIs de segundo orden
# =================================
# Método de Euler
teu, yeu = pvim2.euler_methord2()
# Método de Heun
the, yhe = pvim2.heun_methord2()
# Método de Punto Medio
tptom, yptom = pvim2.ptomed_methord2()
# Método de Runge-Kutta de 4to orden
trk4, yrk4 = pvim2.rk4_methord2()

# Gráficas
p.plot_graph(teu, yeu, the, yhe, tptom, yptom, trk4, yrk4)
