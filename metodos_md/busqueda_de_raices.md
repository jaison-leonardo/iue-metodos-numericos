# Métodos Numéricos: Búsqueda de Raíces

En este documento se resuelve paso a paso una iteración detallada para cada uno de los algoritmos mostrados en las diapositivas. Se utilizarán 4 cifras decimales.

---

## 1. Método de Bisección

**Ejercicio a):** $f(x) = f_1(x) = 6x^2 - 7x + 2$
*Nota: Se elige el intervalo inicial* $[0.0, 0.6]$ *porque encierra la raíz exacta de* $x = 0.5$.

**Algoritmo:**
1. Seleccionar $a_0 = 0.0, b_0 = 0.6$.
   Verificar: $f(0.0) = 2$ y $f(0.6) = 6(0.6)^2 - 7(0.6) + 2 = 2.16 - 4.2 + 2 = -0.04$.
   $f(a_0)f(b_0) = (2)(-0.04) = -0.08 < 0$. (Cumple).

**Iteración 1:**
2. Obtener $r_0 = \frac{a_0 + b_0}{2} = \frac{0.0 + 0.6}{2} = 0.3000$
3. Evaluar $f(r_0) = 6(0.3)^2 - 7(0.3) + 2 = 0.54 - 2.1 + 2 = 0.4400$
4. Como $f(a_0)f(r_0) = (2)(0.44) > 0$, entonces la raíz no está en la primera mitad.
   Se asigna: $a_1 = r_0 = 0.3$ y $b_1 = b_0 = 0.6$.

**Iteración 2:**
2. Obtener $r_1 = \frac{a_1 + b_1}{2} = \frac{0.3 + 0.6}{2} = 0.4500$
3. Evaluar $f(r_1) = 6(0.45)^2 - 7(0.45) + 2 = 1.215 - 3.15 + 2 = 0.0650$
4. Como $f(a_1)f(r_1) = (0.44)(0.065) > 0$.
   Se asigna: $a_2 = r_1 = 0.45$ y $b_2 = b_1 = 0.6$.

**Iteración 3:**
2. Obtener $r_2 = \frac{a_2 + b_2}{2} = \frac{0.45 + 0.6}{2} = 0.5250$
3. Evaluar $f(r_2) = 6(0.525)^2 - 7(0.525) + 2 = 1.6538 - 3.675 + 2 = -0.0212$

| $n$ | $a_n$ | $b_n$ | $r_n$ | $f(r_n)$ |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0.0000 | 0.6000 | 0.3000 | 0.4400 |
| 1 | 0.3000 | 0.6000 | 0.4500 | 0.0650 |
| 2 | 0.4500 | 0.6000 | 0.5250 | -0.0212 |

---

## 2. Método de Falsa Posición

**Ejercicio b):** $f(x) = f_2(x) = 2 \sin(5x) \sin(2x)$
*Nota: Los cálculos trigonométricos están en radianes. Intervalo inicial:* $[0.5, 0.7]$.

**Algoritmo:**
1. Seleccionar $a_0 = 0.5, b_0 = 0.7$.
   $f(0.5) = 2 \sin(2.5) \sin(1.0) = 2(0.5985)(0.8415) = 1.0072$
   $f(0.7) = 2 \sin(3.5) \sin(1.4) = 2(-0.3508)(0.9854) = -0.6914$
   $f(0.5)f(0.7) = (1.0072)(-0.6914) < 0$. (Cumple).

**Iteración 1:**
2. Obtener: 
   $r_0 = \frac{a_0 f(b_0) - b_0 f(a_0)}{f(b_0) - f(a_0)} = \frac{0.5(-0.6914) - 0.7(1.0072)}{-0.6914 - 1.0072} = \frac{-0.3457 - 0.7050}{-1.6986} = 0.6186$
3. Evaluar $f(r_0) = 2 \sin(5(0.6186)) \sin(2(0.6186)) = 0.0918$
4. Como $f(a_0)f(r_0) = (1.0072)(0.0918) > 0$.
   Se asigna: $a_1 = r_0 = 0.6186$ y $b_1 = b_0 = 0.7$.

**Iteración 2:**
2. Obtener:
   $r_1 = \frac{a_1 f(b_1) - b_1 f(a_1)}{f(b_1) - f(a_1)} = \frac{0.6186(-0.6914) - 0.7(0.0918)}{-0.6914 - 0.0918} = \frac{-0.4277 - 0.0643}{-0.7832} = 0.6282$
3. Evaluar $f(r_1) = 2 \sin(3.1410) \sin(1.2564) = 0.0011$

| $n$ | $a_n$ | $b_n$ | $r_n$ | $f(r_n)$ |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0.5000 | 0.7000 | 0.6186 | 0.0918 |
| 1 | 0.6186 | 0.7000 | 0.6282 | 0.0011 |

---

## 3. Método de Newton

**Para mostrar su poder, volveremos a la función a):** $f(x) = 6x^2 - 7x + 2$
*Necesitamos la derivada:* $f'(x) = 12x - 7$. *Iniciaremos en* $r_0 = 1.0$.

**Algoritmo:**
1. Seleccionar $r_0 = 1.0$.
2. Calcule $f(1.0) = 6(1)^2 - 7(1) + 2 = 1.0000$
   Calcule $f'(1.0) = 12(1) - 7 = 5.0000$

**Iteración 1:**
2. Obtenga $r_1 = r_0 - \frac{f(r_0)}{f'(r_0)} = 1.0 - \frac{1.0000}{5.0000} = 1.0 - 0.2000 = 0.8000$
3. Calcule $f(r_1) = 6(0.8)^2 - 7(0.8) + 2 = 3.84 - 5.6 + 2 = 0.2400$
   Calcule $f'(r_1) = 12(0.8) - 7 = 9.6 - 7 = 2.6000$

**Iteración 2:**
4. Obtenga $r_2 = r_1 - \frac{f(r_1)}{f'(r_1)} = 0.8000 - \frac{0.2400}{2.6000} = 0.8000 - 0.0923 = 0.7077$

| $n$ | $r_n$ | $f(r_n)$ | $f'(r_n)$ |
| :---: | :---: | :---: | :---: |
| 0 | 1.0000 | 1.0000 | 5.0000 |
| 1 | 0.8000 | 0.2400 | 2.6000 |
| 2 | 0.7077 | 0.0509 | 1.4924 |

---

## 4. Método de la Secante

**Volveremos a la función b):** $f(x) = 2 \sin(5x) \sin(2x)$
*Este método requiere dos valores iniciales, usaremos los mismos de falsa posición:* $r_{-1} = 0.5$ y $r_0 = 0.7$.

**Algoritmo:**
1. Seleccionar $r_{-1} = 0.5$ y $r_0 = 0.7$.
   Calcule $f(r_{-1}) = f(0.5) = 1.0072$
   Calcule $f(r_0) = f(0.7) = -0.6914$

**Iteración 1:**
2. Obtenga $r_1 = r_0 - \frac{f(r_0)(r_{-1} - r_0)}{f(r_{-1}) - f(r_0)}$
   $r_1 = 0.7 - \frac{-0.6914(0.5 - 0.7)}{1.0072 - (-0.6914)} = 0.7 - \frac{-0.6914(-0.2)}{1.6986} = 0.7 - \frac{0.1383}{1.6986} = 0.7 - 0.0814 = 0.6186$
3. Calcule $f(r_1) = f(0.6186) = 0.0918$

**Iteración 2:**
*(Ahora los puntos anteriores son $r_0 = 0.7$ y $r_1 = 0.6186$)*
2. Obtenga $r_2 = r_1 - \frac{f(r_1)(r_0 - r_1)}{f(r_0) - f(r_1)}$
   $r_2 = 0.6186 - \frac{0.0918(0.7 - 0.6186)}{-0.6914 - 0.0918} = 0.6186 - \frac{0.0918(0.0814)}{-0.7832} = 0.6186 - (-0.0095) = 0.6281$
   $f(r_2) = f(0.6281) = 0.0016$

| $n$ | $r_{n-1}$ | $r_n$ | $r_{n+1}$ | $f(r_{n+1})$ |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0.5000 | 0.7000 | 0.6186 | 0.0918 |
| 1 | 0.7000 | 0.6186 | 0.6281 | 0.0016 |
