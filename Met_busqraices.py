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
El módulo incluye tres categorías de métodos:
Métodos cerrados:
    - Bisección
    - Falsa Posición
Estos métodos requieren un intervalo inicial [a, b]
que contenga una raíz y donde exista cambio de signo.
Métodos abiertos:
    - Newton-Raphson
    - Secante
    - Punto Fijo
Estos métodos utilizan una o varias aproximaciones iniciales
de la raíz y suelen converger más rápido, aunque no siempre
garantizan convergencia.
Todos los métodos retornan:
    - Aproximación de la raíz.
    - Número de iteraciones realizadas.
    - Error final.
    - Aproximaciones intermedias seleccionadas para análisis
      y visualización.

Parámetro adicional opcional ``trace``
---------------------------------------
Todas las funciones aceptan un parámetro opcional ``trace``.
Si se pasa una lista (por ejemplo ``trace=[]``), cada función
agrega en esa lista, EN CADA ITERACIÓN (no solo las de RN), un
diccionario con las cantidades relevantes de ese paso. Esto permite
que una capa externa (reporter) reconstruya la traza pedagógica
completa sin alterar el cálculo ni la firma de retorno original.

Cada registro de ``trace`` (a partir de n=1) incluye también la
clave ``error_aprox``: el error aproximado porcentual entre la
iteración actual y la anterior,
    error_aprox = |r_n - r_(n-1)| / |r_n| * 100
tal como lo define Chapra (no confundir con ``Error``, el residuo
|f(r)| que ya usaba el código para decidir cuándo detenerse: ese
criterio de parada NO se modifica, ``error_aprox`` es puramente
informativo/aditivo). En n=0 (estado inicial) ``error_aprox`` es
``None`` porque todavía no hay una iteración previa con la cual
compararse.

Convención de numeración pública de iteraciones:
    n = 0  -> estado inicial (intervalo o aproximación inicial,
              antes de aplicar la recurrencia).
    n = k  (k >= 1) -> k-ésima aproximación calculada por el método.
Esta es la misma convención que ya usaba internamente ``params.RN``
(``if iterac + 1 in params.RN``), simplemente se documenta y se
extiende para registrar también la fila n = 0.
"""


def _error_aprox(nuevo, anterior):
    """error aproximado porcentual |nuevo - anterior| / |nuevo| * 100.
    Devuelve None si no hay `anterior` o si `nuevo` es 0 (division
    por cero); es una cantidad puramente informativa para el reporte,
    nunca se usa como criterio de parada."""
    if anterior is None or nuevo == 0:
        return None
    return abs(nuevo - anterior) / abs(nuevo) * 100


# Método de la bisección
def biseccion(
    ei=params.A0,
    ed=params.B0,
    tol=params.TOL,
    nmi=params.MAXIT,
    trace=None
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
    trace : list, opcional
        Si se proporciona, se le agregan registros por iteración
        con las claves: n, a, b, r, f_r, error_aprox.
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
    r_anterior = None

    if trace is not None:
        trace.append({'n': 0, 'a': ei, 'b': ed, 'r': None, 'f_r': None, 'error_aprox': None})

    while Error > tol:
        if deff.funcraiz(ei) * deff.funcraiz(ed) < 0:
            r = (ei + ed) / 2
            if iterac + 1 in params.RN:
                iterac_list.append(r)
            if trace is not None:
                trace.append({
                    'n': iterac + 1,
                    'a': ei,
                    'b': ed,
                    'r': r,
                    'f_r': deff.funcraiz(r),
                    'error_aprox': _error_aprox(r, r_anterior),
                })
            r_anterior = r
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
                nmi = params.MAXIT,
                trace = None):
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
    trace : list, opcional
        Si se proporciona, se le agregan registros por iteración
        con las claves: n, a, b, r, f_r, error_aprox.
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
    r_anterior = None

    if trace is not None:
        trace.append({'n': 0, 'a': ei, 'b': ed, 'r': None, 'f_r': None, 'error_aprox': None})

    while Error > tol:
        if deff.funcraiz(ei) * deff.funcraiz(ed) < 0:
            f_ei = deff.funcraiz(ei)
            f_ed = deff.funcraiz(ed)
            r = (ei * f_ed - ed * f_ei) / (f_ed - f_ei)
            if iterac + 1 in params.RN:
                iterac_list.append(r)
            if trace is not None:
                trace.append({
                    'n': iterac + 1,
                    'a': ei,
                    'b': ed,
                    'r': r,
                    'f_r': deff.funcraiz(r),
                    'error_aprox': _error_aprox(r, r_anterior),
                })
            r_anterior = r
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
            nmi = params.MAXIT,
            trace = None):
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
    trace : list, opcional
        Si se proporciona, se le agregan registros por iteración
        con las claves: n, r, f_r, fp_r, error_aprox.
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

    if trace is not None:
        trace.append({
            'n': 0,
            'r': r,
            'f_r': deff.funcraiz(r),
            'fp_r': deff.funcraizder(r),
            'error_aprox': None,
        })

    while Error > tol:
        derivada = deff.funcraizder(r)
        if derivada == 0:
            raise ZeroDivisionError(
                "La derivada es cero. No se puede continuar con Newton."
            )
        r_anterior = r
        r = r - deff.funcraiz(r) / derivada
        if iterac + 1 in params.RN:
            iterac_list.append(r)
        if trace is not None:
            trace.append({
                'n': iterac + 1,
                'r': r,
                'f_r': deff.funcraiz(r),
                'fp_r': deff.funcraizder(r),
                'error_aprox': _error_aprox(r, r_anterior),
            })
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
                nmi = params.MAXIT,
                trace = None):
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
    trace : list, opcional
        Si se proporciona, se le agregan registros por iteración
        con las claves: n, r_prev, r_curr, r_next, f_r_next, error_aprox.
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

    if trace is not None:
        trace.append({
            'n': 0,
            'r_prev': r_anterior,
            'r_curr': r_actual,
            'r_next': None,
            'f_r_next': None,
            'error_aprox': None,
        })

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
        if trace is not None:
            trace.append({
                'n': iterac + 1,
                'r_prev': r_anterior,
                'r_curr': r_actual,
                'r_next': r,
                'f_r_next': deff.funcraiz(r),
                'error_aprox': _error_aprox(r, r_actual),
            })
        r_anterior = r_actual
        r_actual = r
        Error = abs(deff.funcraiz(r))
        iterac += 1
        if iterac >= nmi:
            print("Se ha alcanzado el máximo de iteraciones")
            break
    return r, iterac, Error, iterac_list


# Método de Punto Fijo
def puntofijo(x0 = params.A0,
              tol = params.TOL,
              nmi = params.MAXIT,
              trace = None):
    """
    MÉTODO DE PUNTO FIJO (ITERACIÓN DE PUNTO FIJO)
    Método abierto que reescribe f(x) = 0 como x = g(x) y itera
    r_(n+1) = g(r_n) hasta que dos aproximaciones sucesivas estén
    suficientemente cerca (según el residuo |f(r)|, igual que los
    demás métodos de este módulo).

    IMPORTANTE: a diferencia de bisección/falsa posición/Newton/
    secante (que solo necesitan f(x)), este método necesita la
    función de iteración g(x) tal que x = g(x). g(x) NO se deriva
    automáticamente de f(x) (elegir un buen g(x) es parte del
    análisis del ejercicio, y una elección mala puede diverger);
    se lee de ``params.GX`` (mismo patrón que ``params.FX`` para
    f(x)), y assistant/executor.py hace monkeypatch de esa variable
    igual que ya hace con params.FX.

    Funcionamiento:
    1. Se parte de una aproximación inicial x0.
    2. Se calcula r_(n+1) = g(r_n).
    3. Se evalúa el residuo |f(r_(n+1))| (con la f(x) original,
       leída de params.FX vía deff.funcraiz) para decidir si se
       alcanzó la tolerancia.
    4. Se repite hasta la tolerancia o el máximo de iteraciones.
    Ventajas:
    - Muy simple de implementar; no requiere derivadas.
    Desventajas:
    - Solo converge si |g'(x)| < 1 cerca de la raíz; una elección
      de g(x) inadecuada puede diverger o converger muy lento.
    Parámetros:
    ----------
    x0 : float
        Aproximación inicial.
    tol : float
        Tolerancia deseada (sobre el residuo |f(r)|, igual que los
        demás métodos de este módulo).
    nmi : int
        Número máximo de iteraciones.
    trace : list, opcional
        Si se proporciona, se le agregan registros por iteración
        con las claves: n, r, g_r, f_r, error_aprox.
    Retorna:
    --------
    r : float
        Aproximación final de la raíz.
    iterac : int
        Número de iteraciones realizadas.
    Error : float
        Error final (residuo |f(r)|).
    iterac_list : list
        Aproximaciones guardadas según RN.
    """

    Error = 1
    iterac = 0
    iterac_list = list()
    r = x0

    if trace is not None:
        trace.append({
            'n': 0,
            'r': r,
            'g_r': deff.funcgx(r),
            'f_r': deff.funcraiz(r),
            'error_aprox': None,
        })

    while Error > tol:
        r_anterior = r
        r = deff.funcgx(r_anterior)
        if iterac + 1 in params.RN:
            iterac_list.append(r)
        if trace is not None:
            trace.append({
                'n': iterac + 1,
                'r': r,
                'g_r': deff.funcgx(r),
                'f_r': deff.funcraiz(r),
                'error_aprox': _error_aprox(r, r_anterior),
            })
        Error = abs(deff.funcraiz(r))
        iterac += 1
        if iterac >= nmi:
            print("Se ha alcanzado el máximo de iteraciones")
            break
    return r, iterac, Error, iterac_list
