# Parámetros del PVI de primer orden: y' = f(t, y), y(t_0) = y_0

# Función f de la EDO (string simbólico en t, y)
FUNCED = "y*t**3 - 1.1*y"

# Solución exacta (string simbólico en t)
FUNCSOLEX = "31*exp(-4*t)/32 + t**2/4 - t/8 + 1/32"

# Condiciones iniciales
T0 = 0.
Y0 = 1.

# Punto final del intervalo de solución
T_END = 2.

# Tamaño de paso
HP = 0.5

# Número de pasos (número de iteraciones)
NIT = int((T_END - T0) / HP)

# Configuración de gráficas
CPG = 200
