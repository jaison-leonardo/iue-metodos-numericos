# Parámetros del PVI de segundo orden: y'' = f(t, y, y'), y(t_0)=y_0, y'(t_0)=y'_0

# Función f de la EDO (string simbólico en t, y0, y1)
FUNCED = "4*y0 + 12*t"

# Solución exacta (string simbólico en t)
FUNCSOLEX = "3*exp(2*t) + exp(-2*t) - 3*t"

# Condiciones iniciales para la variable independiente
T0 = 0.

# Condiciones iniciales para la variable dependiente [y(t_0), y'(t_0)]
Y0_INIT = [4., 1.]

# Punto final del intervalo de solución
T_END = 3.

# Tamaño de paso
HP = 0.5

# Número de pasos (número de iteraciones)
NIT = round((T_END - T0) / HP)

# Configuración de gráficas
CPG = 200
