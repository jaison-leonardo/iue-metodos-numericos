import numpy as np
import matplotlib.pyplot as plt
import params_pvi2 as params
import deffuncionespvi2 as dfp2


def plot_graph(teu, yeu, the, yhe, tptom, yptom, trk4, yrk4):
    # Generación de puntos en el intervalo [t_0,T]
    t_v = np.linspace(params.T0, params.T_END, params.CPG)

    plt.figure()
    ref = dfp2.funcsolref(params.T0, params.T_END, params.Y0_INIT)
    plt.plot(ref.t, ref.y[0], 'r', label='Solución de referencia (solve_ivp)')
    plt.plot(teu, yeu[0],'--o',label='Método de Euler')
    plt.plot(the, yhe[0],'--*',label='Método de Heun')
    plt.plot(tptom, yptom[0],'--s',label='Método de Punto Medio')
    plt.plot(trk4, yrk4[0],'--d',label='Método de Runge-Kutta 4to orden')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.title('Soluciones numéricas del PVI planteado')
    plt.show()
