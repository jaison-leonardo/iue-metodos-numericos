'''
Módulo que contiene las funciones que implementan los métodos de solución de problemas de valores iniciales (PVI) para ecuaciones diferenciales ordinarias (EDO) de primer orden.
'''

import numpy as np
import params_pvi1 as params
import deffuncionespvi as dfp


def _detectar_evento(t, y, i, h, evento_y, pendiente_i):
    """
    Revisa si la variable `y` cruzó el valor objetivo `evento_y` entre
    el paso i y el paso i+1 (ambos ya calculados), y si es así calcula
    el instante de cruce.

    Se usa para "parada por evento" (ej. "¿en qué tiempo se vacía el
    tanque?", donde el evento es y == 0), en vez de integrar un número
    fijo de pasos hasta t_end.

    Devuelve (alcanzado, t_evento, indice_final, aproximado, punto_sintetico):
      - Caso normal: cruce por interpolación lineal entre (t[i],y[i]) y
        (t[i+1],y[i+1]) (ambos valores finitos). aproximado=False,
        punto_sintetico=None, indice_final=i+1 (se conserva y[i+1] tal
        cual, ya calculado).
      - Caso borde de dominio: el paso completo hacia i+1 dio un valor
        no finito (ej. sqrt(y) con y<0 por sobrepaso numérico muy cerca
        de evento_y — típico en ecuaciones de vaciado de tanque tipo
        Torricelli, dy/dt = -k*sqrt(y)). En vez de propagar ese NaN, se
        aproxima el instante del cruce por extrapolación lineal usando
        la pendiente conocida en el último punto válido (t[i], y[i]), y
        se agrega un punto sintético (t_evento, evento_y) como resultado
        final. aproximado=True, indice_final=i (se descarta el punto
        inválido i+1).
      - Si no hay cruce: (False, None, None, False, None).
    """
    diff_i = y[i] - evento_y
    diff_ip1 = y[i + 1] - evento_y

    if diff_i == 0:
        return True, t[i], i, False, None

    if not np.isfinite(diff_ip1):
        if pendiente_i is None or not np.isfinite(pendiente_i) or pendiente_i == 0:
            # No hay forma confiable de estimar el cruce: se trata como
            # "evento no alcanzado" en este paso (el llamador decide qué
            # hacer con el resto, ej. seguir integrando no es seguro, así
            # que se detiene igual y se reporta como no alcanzado).
            return False, None, None, False, None
        t_evento = t[i] - diff_i / pendiente_i
        return True, t_evento, i, True, (t_evento, evento_y)

    if diff_ip1 == 0 or (diff_i > 0) != (diff_ip1 > 0):
        t_evento = t[i] + h * diff_i / (diff_i - diff_ip1)
        return True, t_evento, i + 1, False, None

    return False, None, None, False, None


def _reportar_evento(evento_info, alcanzado, t_evento, paso_evento, aproximado=False):
    if evento_info is not None:
        evento_info["alcanzado"] = alcanzado
        evento_info["t_evento"] = t_evento
        evento_info["paso_evento"] = paso_evento
        evento_info["aproximado"] = aproximado


def _aplicar_evento(t, y, i, h, evento_y, pendiente_i, evento_info):
    """
    Helper compartido por los 4 métodos: revisa el evento tras calcular
    y[i+1]/t[i+1] y, si se alcanzó (exacto o vía el fallback de dominio),
    trunca los arreglos y devuelve (t, y) listos para retornar. Si no se
    alcanzó, devuelve None (el llamador sigue el bucle normalmente).
    """
    alcanzado, t_evento, idx_final, aproximado, punto_sint = _detectar_evento(
        t, y, i, h, evento_y, pendiente_i
    )
    if not alcanzado:
        return None
    t_out, y_out = t[:idx_final + 1], y[:idx_final + 1]
    if punto_sint is not None:
        t_out = np.append(t_out, punto_sint[0])
        y_out = np.append(y_out, punto_sint[1])
    _reportar_evento(evento_info, True, t_evento, idx_final, aproximado)
    return t_out, y_out


# Método de Euler
def euler_meth(t0=params.T0, y0=params.Y0, n=params.NIT, h=params.HP, evento_y=None, evento_info=None):
    # t0: Valor inicial del intervalo (float)
    # y0: Valor inicial de la solución (float).
    # n: Número de pasos (int).
    # h: Tamaño del paso (float).
    # evento_y: (opcional) valor de y en el que detener la integración
    #     antes de completar los n pasos (ej. y=0 para "tiempo hasta
    #     vaciarse el tanque"). Si es None (default), el comportamiento
    #     es IDÉNTICO al de siempre: se integran los n pasos completos.
    # evento_info: (opcional) dict de salida; si se pasa, se llena con
    #     {'alcanzado', 't_evento', 'paso_evento', 'aproximado'}.
    # t: Arreglo con los valores de t (numpy array).
    # y: Arreglo con los valores aproximados de la solución (numpy array).

    t = np.zeros(n+1)
    y = np.zeros(n+1)
    t[0] = t0
    y[0] = y0
    for i in range(n):
        k1 = dfp.funced(t[i], y[i])
        y[i+1] = y[i] + h*k1
        t[i+1] = t[i] + h
        if evento_y is not None:
            resultado = _aplicar_evento(t, y, i, h, evento_y, k1, evento_info)
            if resultado is not None:
                return resultado
    _reportar_evento(evento_info, False, None, None)
    return t,y

# Método de Heun
def heun_meth(t0=params.T0, y0=params.Y0, n=params.NIT, h=params.HP, evento_y=None, evento_info=None):
    # t0: Valor inicial del intervalo (float)
    # y0: Valor inicial de la solución (float).
    # n: Número de pasos (int).
    # h: Tamaño del paso (float).
    # evento_y / evento_info: ver euler_meth.
    # t: Arreglo con los valores de t (numpy array).
    # y: Arreglo con los valores aproximados de la solución (numpy array).

    t = np.zeros(n+1)
    y = np.zeros(n+1)
    t[0] = t0
    y[0] = y0
    for i in range(n):
        k1 = dfp.funced(t[i], y[i])
        yaux = y[i] + h*k1
        t[i+1] = t[i] + h
        pend = (k1 + dfp.funced(t[i+1], yaux))/2.0
        y[i+1] = y[i] + h*pend
        if evento_y is not None:
            resultado = _aplicar_evento(t, y, i, h, evento_y, k1, evento_info)
            if resultado is not None:
                return resultado
    _reportar_evento(evento_info, False, None, None)
    return t,y

# Método del Punto Medio
def ptomed_meth(t0=params.T0, y0=params.Y0, n=params.NIT, h=params.HP, evento_y=None, evento_info=None):
    # t0: Valor inicial del intervalo (float)
    # y0: Valor inicial de la solución (float).
    # n: Número de pasos (int).
    # h: Tamaño del paso (float).
    # evento_y / evento_info: ver euler_meth.
    # t: Arreglo con los valores de t (numpy array).
    # y: Arreglo con los valores aproximados de la solución (numpy array).

    t = np.zeros(n+1)
    y = np.zeros(n+1)
    t[0] = t0
    y[0] = y0
    for i in range(n):
        k1 = dfp.funced(t[i], y[i])
        yaux = y[i] + h*k1/2.0
        taux = t[i] + 0.5*h
        y[i+1] = y[i] + h*dfp.funced(taux, yaux)
        t[i+1] = t[i] + h
        if evento_y is not None:
            resultado = _aplicar_evento(t, y, i, h, evento_y, k1, evento_info)
            if resultado is not None:
                return resultado
    _reportar_evento(evento_info, False, None, None)
    return t,y

# Método de Runge-Kutta de 4to orden
def rk4_meth(t0=params.T0, y0=params.Y0, n=params.NIT, h=params.HP, evento_y=None, evento_info=None):
    # t0: Valor inicial del intervalo (float)
    # y0: Valor inicial de la solución (float).
    # n: Número de pasos (int).
    # h: Tamaño del paso (float).
    # evento_y / evento_info: ver euler_meth.
    # t: Arreglo con los valores de t (numpy array).
    # y: Arreglo con los valores aproximados de la solución (numpy array).
    t = np.zeros(n+1)
    y = np.zeros(n+1)
    t[0] = t0
    y[0] = y0
    for i in range(n):
        k1 = dfp.funced(t[i],y[i])
        k2 = dfp.funced(t[i] + h/2, y[i] + k1*h/2)
        k3 = dfp.funced(t[i] + h/2, y[i] + k2*h/2)
        k4 = dfp.funced(t[i] + h,y[i] + k3*h)
        m = (k1 + 2*k2 + 2*k3 + k4)/6
        y[i+1] = y[i] + m*h
        t[i+1] = t[i] + h
        if evento_y is not None:
            resultado = _aplicar_evento(t, y, i, h, evento_y, k1, evento_info)
            if resultado is not None:
                return resultado
    _reportar_evento(evento_info, False, None, None)
    return t,y
