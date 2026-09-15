'''
Código ecrito para aproximar una raíz, r, de una función f.
El intervalo inicial para los métodos cerrados es [a_0, b_0].
a_0 es la aproximación inicial para los métodos abiertos que 
utilizan una única aproximación, aquellos que requieren dos
aproximaciones inciales utilizan adicionalmente r_.
Este código implementa los siguientes métodos:
1. Bisección
2. Falsa posición
3. Newton
4. Secante
'''

import numpy as np
import pandas as pd
import params
import plot as p
import deffunciones as deff
import Met_busqraices as metb
import matplotlib.pyplot as plt

# Utilización de los métodos de búsqueda de raíces
# params a_0,b_0,tol,maxit
solb, nitb, erb, iter_rb = metb.biseccion()
# params a_0,b_0,tol,maxit
solfp, nitfp, erfp, iter_rfp = metb.falsapos()
# params a_0,tol,maxit
soln, nitn, ern, iter_rn = metb.newton()
# params r_,a_0,tol,maxit
solsec, nitsec, ersec, iter_rsec = metb.secante() 

resultados = pd.DataFrame({
    'Método': ['Bisección', 'Falsa Posición', 'Newton', 'Secante'],
    'Raíz': [solb, solfp, soln, solsec],
    'Iteraciones': [nitb, nitfp, nitn, nitsec],
    'Error': [erb, erfp, ern, ersec]
}).sort_values(by='Error', ascending=True).reset_index(drop=True)
print(resultados)

# Gráficas
# Generación de puntos en el intervalo [a,b]
xg = np.linspace(params.A0,params.B0,params.CPG)
# extensión de intervalo [a,b] para graficar eje x
ext = np.abs(params.B0 - params.A0) * params.EXT_FACTOR
xeje = np.linspace(params.A0 - ext,params.B0 + ext,params.CPG)
ejex = xeje * 0
p.plot_graph(resultados, xg, xeje, ejex)

