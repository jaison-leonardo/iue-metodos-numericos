import numpy as np
import matplotlib.pyplot as plt
import params_pvf as params
import deffuncionespvf as dfp


def plot_graph(t, y):
    # Generación de puntos en el intervalo [t_0, t_n]
    t_v = np.linspace(params.T0, params.TN, params.CPG)
    ysr = np.ones((2, params.CPG))

    plt.figure()
    plt.plot(t_v, dfp.funcsolex(t_v), 'k', label='Solución exacta')
    ref = dfp.funcsolref(t_v, ysr, params.Y0, params.YN)
    plt.plot(ref.x, ref.y[0], 'r--', label='Solución de referencia (solve_bvp)')
    plt.plot(t, y, 'b--s', label='Diferencias finitas centradas')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.title('Soluciones numéricas del PVI planteado')
    plt.show()
