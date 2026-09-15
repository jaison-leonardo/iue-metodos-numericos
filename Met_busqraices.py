"""
Módulo que contiene las funciones que implementan los métodos
de búsqueda de raíces.
"""
import numpy as np
import params
import deffunciones as deff

"""
Met_busqraices.py
Implementación de métodos numéricos para la búsqueda de raíces
de ecuaciones no lineales de la forma:
    f(x) = 0
El módulo incluye dos categorías de métodos:
Métodos cerrados:
    - Bisección
    - Falsa Posición
Estos métodos requieren un intervalo inicial [a, b]
que contenga una raíz y donde exista cambio de signo.
Métodos abiertos:
    - Newton-Raphson
    - Secante
Estos métodos utilizan una o varias aproximaciones iniciales
de la raíz y suelen converger más rápido, aunque no siempre
garantizan convergencia.
Todos los métodos retornan:
    - Aproximación de la raíz.
    - Número de iteraciones realizadas.
    - Error final.
    - Aproximaciones intermedias seleccionadas para análisis
      y visualización.
"""

# Método de la bisección
def biseccion(
    ei=params.A0,
    ed=params.B0,
    tol=params.TOL,
    nmi=params.MAXIT
):
    """
    MÉTODO DE LA BISECCIÓN
    Método cerrado para encontrar una raíz de una función continua.
    Funcionamiento:
    1. Se parte de un intervalo [ei, ed] donde la función cambia
       de signo, es decir:
           f(ei) * f(ed) < 0
       Esto garantiza la existencia de al menos una raíz dentro
       del intervalo según el Teorema del Valor Intermedio.
    2. En cada iteración se calcula el punto medio:
           r = (ei + ed) / 2
    3. Se evalúa el signo de la función en r para determinar
       en cuál mitad del intervalo se encuentra la raíz.
    4. El intervalo se reduce sucesivamente hasta alcanzar
       la tolerancia especificada.
    Características:
    - Siempre converge si se cumplen las hipótesis.
    - Es muy robusto.
    - Su convergencia es relativamente lenta.
    Parámetros:
    ----------
    ei : float
        Extremo izquierdo del intervalo inicial.
    ed : float
        Extremo derecho del intervalo inicial.
    tol : float
        Tolerancia deseada para el error.
    nmi : int
        Número máximo de iteraciones.
    Retorna:
    --------
    r : float
        Aproximación de la raíz.
    iterac : int
        Cantidad de iteraciones realizadas.
    Error : float
        Error final calculado.
    iterac_list : list
        Lista de aproximaciones almacenadas según RN.
    """
    
    Error = 1
    iterac = 0
    iterac_list = list()
    r = None

    while Error > tol:
        if deff.funcraiz(ei) * deff.funcraiz(ed) < 0:
            r = (ei + ed) / 2
            if iterac + 1 in params.RN:
                iterac_list.append(r)
            if deff.funcraiz(r) * deff.funcraiz(ei) < 0:
                ed = r
            elif deff.funcraiz(r) == 0:
                Error = 0
                break
            else:
                ei = r
        elif deff.funcraiz(ei) * deff.funcraiz(ed) == 0:
            if deff.funcraiz(ei) == 0:
                r = ei
            else:
                r = ed
            Error = 0
            break
        else:
            raise ValueError(
                "No es posible asegurar la existencia de una raíz en el intervalo ingresado."
            )
        Error = abs(deff.funcraiz(r))
        if iterac >= nmi:
            print("Se ha alcanzado el máximo de iteraciones")
            break
        iterac += 1
    return r, iterac, Error, iterac_list


# Método de falsa posición
def falsapos(ei = params.A0,
                ed = params.B0,
                tol = params.TOL,
                nmi = params.MAXIT):
    """
    MÉTODO DE FALSA POSICIÓN (REGULA FALSI)
    Método cerrado para encontrar raíces utilizando una recta
    secante entre los extremos del intervalo.
    Funcionamiento:
    1. Se inicia con un intervalo [ei, ed] donde existe un cambio
       de signo.
    2. Se construye una recta que une los puntos:
           (ei, f(ei))
           (ed, f(ed))
    3. Se calcula el punto donde dicha recta corta el eje x.
    4. Se evalúa el signo de la función en el nuevo punto
       y se actualiza el intervalo conservando el cambio de signo.
    Ventajas:
    - Generalmente converge más rápido que la bisección.
    - Conserva la seguridad de los métodos cerrados.
    Desventajas:
    - Puede estancarse cuando uno de los extremos permanece fijo
      durante muchas iteraciones.
    Parámetros:
    ----------
    ei : float
        Extremo izquierdo del intervalo.
    ed : float
        Extremo derecho del intervalo.
    tol : float
        Tolerancia deseada.
    nmi : int
        Número máximo de iteraciones.
    Retorna:
    --------
    r : float
        Aproximación de la raíz.
    iterac : int
        Número de iteraciones.
    Error : float
        Error final.
    iterac_list : list
        Aproximaciones guardadas.
    """

    Error = 1
    iterac = 0
    iterac_list = list()
    r = None

    while Error > tol:
        if deff.funcraiz(ei) * deff.funcraiz(ed) < 0:
            f_ei = deff.funcraiz(ei)
            f_ed = deff.funcraiz(ed)
            r = (ei * f_ed - ed * f_ei) / (f_ed - f_ei)
            if iterac + 1 in params.RN:
                iterac_list.append(r)
            if deff.funcraiz(r) * deff.funcraiz(ei) < 0:
                ed = r
            elif deff.funcraiz(r) == 0:
                Error = 0
                break
            else:
                ei = r
        elif deff.funcraiz(ei) * deff.funcraiz(ed) == 0:
            if deff.funcraiz(ei) == 0:
                r = ei
            else:
                r = ed
            Error = 0
            break
        else:
            raise ValueError(
                "No es posible asegurar la existencia de una raíz en el intervalo ingresado."
            )
        Error = abs(deff.funcraiz(r))
        if iterac >= nmi:
            print("Se ha alcanzado el máximo de iteraciones")
            break
        iterac += 1
    return r, iterac, Error, iterac_list


# Método de Newton
def newton(r = params.A0,
            tol = params.TOL,
            nmi = params.MAXIT):
    """
    MÉTODO DE NEWTON-RAPHSON
    Método abierto para aproximar raíces utilizando la recta
    tangente a la curva en cada iteración.
    Funcionamiento:
    1. Se parte de una aproximación inicial r.
    2. Se calcula la pendiente de la función en dicho punto:
           f'(r)
    3. Se obtiene el punto donde la recta tangente corta al eje x:
           r_(n+1) = r_n - f(r_n)/f'(r_n)
    4. El proceso se repite hasta alcanzar la tolerancia deseada.
    Ventajas:
    - Convergencia muy rápida cerca de la raíz.
    - Requiere pocas iteraciones.
    Desventajas:
    - Necesita conocer la derivada.
    - Puede divergir si el punto inicial no es adecuado.
    - Falla cuando f'(x) = 0.
    Parámetros:
    ----------
    r : float
        Aproximación inicial.
    tol : float
        Tolerancia del método.
    nmi : int
        Número máximo de iteraciones.
    Retorna:
    --------
    r : float
        Aproximación final de la raíz.
    iterac : int
        Número de iteraciones realizadas.
    Error : float
        Error final.
    iterac_list : list
        Aproximaciones guardadas.
    """

    Error = 1
    iterac = 0
    iterac_list = list()
    while Error > tol:
        derivada = deff.funcraizder(r)
        if derivada == 0:
            raise ZeroDivisionError(
                "La derivada es cero. No se puede continuar con Newton."
            )
        r = r - deff.funcraiz(r) / derivada
        if iterac + 1 in params.RN:
            iterac_list.append(r)
        Error = abs(deff.funcraiz(r))
        iterac += 1
        if iterac >= nmi:
            print("Se ha alcanzado el máximo de iteraciones")
            break
    return r, iterac, Error, iterac_list

# Método de la secante
def secante(r_anterior = params.R0,
                r_actual = params.A0,
                tol = params.TOL,
                nmi = params.MAXIT):
    """
    MÉTODO DE LA SECANTE
    Método abierto que aproxima la derivada mediante una recta
    secante construida con dos aproximaciones consecutivas.
    Funcionamiento:
    1. Se parte de dos aproximaciones iniciales:
           r_(n-1), r_n
    2. Se construye la recta secante que pasa por:
           (r_(n-1), f(r_(n-1)))
           (r_n,     f(r_n))
    3. Se calcula el punto donde dicha recta corta al eje x.
    4. El nuevo punto reemplaza a la aproximación más antigua.
    Ventajas:
    - No requiere calcular derivadas.
    - Generalmente converge más rápido que bisección
      y falsa posición.
    Desventajas:
    - Necesita dos aproximaciones iniciales.
    - Puede divergir para ciertas funciones.
    - Puede presentar divisiones por cero.
    Parámetros:
    ----------
    r_anterior : float
        Primera aproximación inicial.
    r_actual : float
        Segunda aproximación inicial.
    tol : float
        Tolerancia del método.
    nmi : int
        Número máximo de iteraciones.
    Retorna:
    --------
    r : float
        Aproximación final de la raíz.
    iterac : int
        Número de iteraciones realizadas.
    Error : float
        Error final.
    iterac_list : list
        Aproximaciones guardadas.
    """

    Error = 1
    iterac = 0
    iterac_list = list()
    while Error > tol:
        f_anterior = deff.funcraiz(r_anterior)
        f_actual = deff.funcraiz(r_actual)
        denominador = f_anterior - f_actual
        if denominador == 0:
            raise ZeroDivisionError(
                "División por cero en el método de la secante."
            )
        r = r_actual - (f_actual * (r_anterior - r_actual)) / denominador
        if iterac + 1 in params.RN:
            iterac_list.append(r)
        r_anterior = r_actual
        r_actual = r
        Error = abs(deff.funcraiz(r))
        iterac += 1
        if iterac >= nmi:
            print("Se ha alcanzado el máximo de iteraciones")
            break
    return r, iterac, Error, iterac_list
