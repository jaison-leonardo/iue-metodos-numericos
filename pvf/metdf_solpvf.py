import numpy as np
import params_pvf as params
import deffuncionespvf as dfp

# Método de diferencias finitas centradas (segundo orden de precisión)
def diffincentr(
    t_0=params.T0,
    t_n=params.TN,
    y_0=params.Y0,
    y_n=params.YN,
    h=params.HP
):

    n = int((t_n - t_0) / h)
    t = np.linspace(t_0, t_n, n + 1)

    y = np.zeros((n+1))     # Vector de solución
    y[0] = y_0              # Condición de frontera inicial
    y[-1] = y_n             # Condición de frontera final

    mat = np.zeros((n-1,n-1))   # Matriz del sistema
    vec = np.zeros((n-1))   # Vector del sistema
    
    # Loop que construye el sistema de ecuaciones Ax = b
    for i in range(n-1):
        if i == 0:  # Primera fila
            # Primera posición en la primera fila de la matriz A
            mat[i][i] = -2*dfp.A(t[i+1])/h**2 + dfp.C(t[i+1])
            # Segunda posición en la primera fila de la matriz A
            mat[i][i+1] = dfp.A(t[i+1])/h**2 + dfp.B(t[i+1])/(2*h)
            # Primera posición en el vector b
            vec[i] = dfp.f(t[i+1]) - (dfp.A(t[i+1])/h**2 - dfp.B(t[i+1])/(2*h))*y_0
        if i == n-2:    # Última fila
            # Penúltima posición en la última fila de la matriz A
            mat[i][i-1] = dfp.A(t[i+1])/h**2 - dfp.B(t[i+1])/(2*h)
            # Última posición en la última fila de la matriz A
            mat[i][i] = -2*dfp.A(t[i+1])/h**2 + dfp.C(t[i+1])
            # Última posición en el vector b
            vec[i] = dfp.f(t[i+1]) - (dfp.A(t[i+1])/h**2 + dfp.B(t[i+1])/(2*h))*y_n
        if 0 < i < n-2: # Filas intermedias
            # Posición i-1 en la fila i de la matriz A (izquierda de la diagonal principal)
            mat[i][i-1] = dfp.A(t[i+1])/h**2 - dfp.B(t[i+1])/(2*h)
            # Posición i en la fila i de la matriz A (diagonal principal)
            mat[i][i] = -2*dfp.A(t[i+1])/h**2 + dfp.C(t[i+1])
            # Posición i+1 en la fila i de la matriz A (derecha de la diagonal principal)
            mat[i][i+1] = dfp.A(t[i+1])/h**2 + dfp.B(t[i+1])/(2*h)
            # Posición i en el vector b
            vec[i] = dfp.f(t[i+1])
    
    y[1:-1] = np.linalg.solve(mat, vec) # Solución del sistema de ecuaciones

    return t, y