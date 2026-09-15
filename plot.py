import matplotlib.pyplot as plt
import deffunciones as deff


def plot_graph(resultados, xg, xeje, ejex):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    colores = ['r', 'g', 'm', 'c']
    marcadores = ['o', 's', '^', 'D']
    for i, (_, fila) in enumerate(resultados.iterrows()):
        fila_ax = i // 2
        col_ax = i % 2
        ax = axs[fila_ax, col_ax]
        ax.plot(xg, deff.funcraiz(xg))
        ax.plot(xeje, ejex, '--k')
        ax.plot(
            fila["Raíz"],
            0,
            color=colores[i],
            marker=marcadores[i],
            markersize=8
        )
        ax.set_title(fila["Método"])
        ax.grid(True)
    plt.tight_layout()
    plt.show()