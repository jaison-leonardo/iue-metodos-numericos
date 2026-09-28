# Resolución de PVI usando el Método de Euler

**Ecuación Diferencial:** $y'(t) = y t^3 - 1.1y \Rightarrow f(t,y) = y(t^3 - 1.1)$
**Condición Inicial:** $y(0) = 1$ (es decir, $t_0 = 0, y_0 = 1$)
**Intervalo:** $[0, 2]$
**Tamaño de paso:** $h = 0.5$
**Fórmula de Euler:** $y_{i+1} = y_i + f(t_i, y_i) h$
**Solución Analítica:** $y = e^{\frac{t^4}{4} - 1.1t}$

---

## Iteración 1 ($n=1, t_1=0.5$)

Partimos de $t_0 = 0, y_0 = 1$.

1. **Pendiente inicial:** $m_0 = f(0, 1) = 1(0^3 - 1.1) = -1.1$
2. **Siguiente valor:** $y_1 = y_0 + m_0 h = 1 + (-1.1)(0.5) = 1 - 0.5500 = 0.4500$

* **Analítico:** $y_{analítico}(0.5) = 0.5860$
* **Error:** $\epsilon_t = \left\vert{} \frac{0.5860 - 0.4500}{0.5860} \right\vert{} \times 100 = 23.21\%$

---

## Iteración 2 ($n=2, t_2=1.0$)

Partimos de $t_1 = 0.5, y_1 = 0.4500$.

1. **Pendiente inicial:** $m_1 = f(0.5, 0.4500) = 0.4500(0.5^3 - 1.1) = 0.4500(-0.975) = -0.4388$
2. **Siguiente valor:** $y_2 = y_1 + m_1 h = 0.4500 + (-0.4388)(0.5) = 0.4500 - 0.2194 = 0.2306$

* **Analítico:** $y_{analítico}(1.0) = 0.4274$
* **Error:** $\epsilon_t = \left\vert{} \frac{0.4274 - 0.2306}{0.4274} \right\vert{} \times 100 = 46.05\%$

---

## Iteración 3 ($n=3, t_3=1.5$)

Partimos de $t_2 = 1.0, y_2 = 0.2306$.

1. **Pendiente inicial:** $m_2 = f(1.0, 0.2306) = 0.2306(1.0^3 - 1.1) = 0.2306(-0.1) = -0.0231$
2. **Siguiente valor:** $y_3 = y_2 + m_2 h = 0.2306 + (-0.0231)(0.5) = 0.2306 - 0.0116 = 0.2190$

* **Analítico:** $y_{analítico}(1.5) = 0.6809$
* **Error:** $\epsilon_t = \left\vert{} \frac{0.6809 - 0.2190}{0.6809} \right\vert{} \times 100 = 67.84\%$

---

## Iteración 4 ($n=4, t_4=2.0$)

Partimos de $t_3 = 1.5, y_3 = 0.2190$.

1. **Pendiente inicial:** $m_3 = f(1.5, 0.2190) = 0.2190(1.5^3 - 1.1) = 0.2190(2.275) = 0.4982$
2. **Siguiente valor:** $y_4 = y_3 + m_3 h = 0.2190 + (0.4982)(0.5) = 0.2190 + 0.2491 = 0.4681$

* **Analítico:** $y_{analítico}(2.0) = 6.0496$
* **Error:** $\epsilon_t = \left\vert{} \frac{6.0496 - 0.4681}{6.0496} \right\vert{} \times 100 = 92.26\%$

---

## Tabla de Resultados Final - Método de Euler

| $n$ | $t$ | $y_{num}$ (Euler) | $y_{analítico}$ | $\epsilon_t$ (%) |
| ----- | ----- | ----- | ----- | ----- |
| 0 | 0.0 | 1.0000 | 1.0000 | 0.00% |
| 1 | 0.5 | 0.4500 | 0.5860 | 23.21% |
| 2 | 1.0 | 0.2306 | 0.4274 | 46.05% |
| 3 | 1.5 | 0.2190 | 0.6809 | 67.84% |
| 4 | 2.0 | 0.4681 | 6.0496 | 92.26% |
