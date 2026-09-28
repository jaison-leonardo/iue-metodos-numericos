# Parámetros del PVF: A(x)y'' + B(x)y' + C(x)y = f(x)

# Coeficiente de la segunda derivada
AX = "7"

# Coeficiente de la primera derivada
BX = "-2"

# Coeficiente de la función y
CX = "-1"

# Función de términos independientes
FX = "-x"

# Solución exacta (string simbólico en x)
FUNCSOLEX = (
    "-(10*exp(-20/7) + 7*exp(-40*sqrt(2)/7)) / (2*sinh(40*sqrt(2)/7)) "
    "* exp((1 + 2*sqrt(2))/7*x) "
    "+ (10*exp(-20/7) + 7*exp(40*sqrt(2)/7)) / (2*sinh(40*sqrt(2)/7)) "
    "* exp((1 - 2*sqrt(2))/7*x) + x - 2"
)

# Condiciones de frontera
T0 = 0.
Y0 = 5.
TN = 20.
YN = 8.

# Tamaño de paso
HP = 2.

# Configuración de gráficas
CPG = 200
