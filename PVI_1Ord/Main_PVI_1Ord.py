'''
Código construido para solucionar un PVI conformado
por una EDO de primer orden de la forma y'=f(t,y) y
condiciones iniciales y(t_0)=y_0. Este código implementa
los métodos de Euler, Heun, Punto Medio y Runge-Kutta
de 4to orden. Los métodos de Heun y Punto Medio surgen
de procesos de mejoramiento en el método de Euler.
'''

import met_solpvi_1ord as pvim
import plot_pvi1 as p

# Solución con el método de Euler
# params t0,y0,n,h
teu, yeu = pvim.euler_meth()
the, yhe = pvim.heun_meth()
tptom, yptom = pvim.ptomed_meth()
trk4, yrk4 = pvim.rk4_meth()

# Gráficas
p.plot_graph(teu, yeu, the, yhe, tptom, yptom, trk4, yrk4)
