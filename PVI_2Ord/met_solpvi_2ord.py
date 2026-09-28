'''
Módulo que contiene las funciones que implementan los métodos de solución de problemas con valores iniciales (PVI) para ecuaciones diferenciales ordinarias (EDO) de segundo orden.
'''

import numpy as np
import params_pvi2 as params
import deffuncionespvi2 as dfp2

# Métodos para PVIs de segundo orden
# ===================================

def euler_methord2(t0=params.T0, yo2=params.Y0_INIT, n=params.NIT, h=params.HP):
    # t0: Valor inicial del intervalo (float)
    # y02: Condiciones iniciales [y(t0), y'(t0)] (array-like de tamaño 2).
    # n: Número de pasos (int).
    # h: Tamaño del paso (float).
    # t: Arreglo con los valores de t (numpy array).
    # y: Arreglo con los valores aproximados de la solución del sistema de PVIs (numpy array).
    
    t = np.zeros(n+1)
    y = np.zeros((2,n+1))
    t[0] = t0
    y[0,0] = yo2[0]
    y[1,0] = yo2[1]
    for i in range(n):
        y[0,i+1] = y[0,i] + h*y[1,i]
        y[1,i+1] = y[1,i] + h*dfp2.funced(t[i], y[:,i])
        t[i+1] = t[i] + h
    return t,y

def heun_methord2(t0=params.T0, yo2=params.Y0_INIT, n=params.NIT, h=params.HP):
    # t0: Valor inicial del intervalo (float)
    # yo2: Condiciones iniciales [y(t0), y'(t0)] (array-like de tamaño 2)
    # n: Número de pasos (int)
    # h: Tamaño del paso (float)
    # t: Arreglo con los valores de t (numpy array).
    # y: Arreglo con los valores aproximados de la solución del sistema de PVIs (numpy array).

    t = np.zeros(n+1)
    y = np.zeros((2,n+1))
    t[0] = t0
    y[0,0] = yo2[0]
    y[1,0] = yo2[1]

    for i in range(n):
        # Predictor de Euler explícito para el sistema de primer orden equivalente
        y_p = np.zeros(2)
        y_p[0] = y[0,i] + h*y[1,i]
        y_p[1] = y[1,i] + h*dfp2.funced(t[i], y[:,i])

        # Corrector de Heun (promedio de pendientes)
        f_i = dfp2.funced(t[i], y[:,i])
        f_p = dfp2.funced(t[i] + h, y_p)

        y[0,i+1] = y[0,i] + h*(y[1,i] + y_p[1]) / 2.0
        y[1,i+1] = y[1,i] + h*(f_i + f_p) / 2.0
        t[i+1] = t[i] + h

    return t,y

def ptomed_methord2(t0=params.T0, yo2=params.Y0_INIT, n=params.NIT, h=params.HP):
    # t0: Valor inicial del intervalo (float)
    # yo2: Condiciones iniciales [y(t0), y'(t0)] (array-like de tamaño 2)
    # n: Número de pasos (int)
    # h: Tamaño del paso (float)
    # t: Arreglo con los valores de t (numpy array).
    # y: Arreglo con los valores aproximados de la solución del sistema de PVIs (numpy array).

    t = np.zeros(n+1)
    y = np.zeros((2,n+1))
    t[0] = t0
    y[0,0] = yo2[0]
    y[1,0] = yo2[1]

    for i in range(n):
        f_i = dfp2.funced(t[i], y[:,i])

        # Punto medio: avance semipaso para y y y'
        y_mid = np.zeros(2)
        y_mid[0] = y[0,i] + h*y[1,i]/2.0
        y_mid[1] = y[1,i] + h*f_i/2.0
        t_mid = t[i] + h/2.0

        # Usar la pendiente en el punto medio para corregir el paso completo
        f_mid = dfp2.funced(t_mid, y_mid)
        y[0,i+1] = y[0,i] + h*y_mid[1]
        y[1,i+1] = y[1,i] + h*f_mid
        t[i+1] = t[i] + h

    return t,y

def rk4_methord2(t0=params.T0, yo2=params.Y0_INIT, n=params.NIT, h=params.HP):
    # t0: Valor inicial del intervalo (float)
    # yo2: Condiciones iniciales [y(t0), y'(t0)] (array-like de tamaño 2)
    # n: Número de pasos (int)
    # h: Tamaño del paso (float)
    # t: Arreglo con los valores de t (numpy array).
    # y: Arreglo con los valores aproximados de la solución del sistema de PVIs (numpy array).

    t = np.zeros(n+1)
    y = np.zeros((2,n+1))
    t[0] = t0
    y[0,0] = yo2[0]
    y[1,0] = yo2[1]

    for i in range(n):
        # k1
        k1 = np.zeros(2)
        k1[0] = y[1,i]
        k1[1] = dfp2.funced(t[i], y[:,i])

        # k2
        y_k2 = y[:,i] + h*k1/2.0
        k2 = np.zeros(2)
        k2[0] = y_k2[1]
        k2[1] = dfp2.funced(t[i] + h/2.0, y_k2)

        # k3
        y_k3 = y[:,i] + h*k2/2.0
        k3 = np.zeros(2)
        k3[0] = y_k3[1]
        k3[1] = dfp2.funced(t[i] + h/2.0, y_k3)

        # k4
        y_k4 = y[:,i] + h*k3
        k4 = np.zeros(2)
        k4[0] = y_k4[1]
        k4[1] = dfp2.funced(t[i] + h, y_k4)

        # Actualizar solución
        y[:,i+1] = y[:,i] + h*(k1 + 2*k2 + 2*k3 + k4)/6.0
        t[i+1] = t[i] + h

    return t,y