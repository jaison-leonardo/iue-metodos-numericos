# Parámetro del ejercicio
NN = 32

# Función a analizar
FX = f"x**3 + 3*x - {NN}"

# Intervalo inicial (métodos cerrados)
A0 = NN / 100
B0 = 8 + A0

# Aproximación inicial adicional (secante)
R0 = -0.8

# Tolerancia
TOL = 1e-6

# Máximo número de iteraciones
MAXIT = 50

# Configuración de gráficas
CPG = 200
EXT_FACTOR = 0.2

# rn
RN = [5,9]