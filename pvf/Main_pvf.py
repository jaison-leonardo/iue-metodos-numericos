'''
Código construido para solucionar un PVF conformado
por una EDO de segundo orden de la forma:
A(x)y'' + B(x)y' + C(x)y = f(x) y
condiciones de frontera y(t_0)=y_0 ; y(t_n)=y_n.
El PVF se soluciona sobre el intervalo [t_0, t_n] utilizando
el método de diferencias finitas con un tamaño de paso h.
'''

import metdf_solpvf as pvfm
import plot_pvf as p

# Solución con el método de diferencias finitas
# params t_0,t_n,y_0,y_n,h
t, y = pvfm.diffincentr()

# Gráficas
p.plot_graph(t, y)
