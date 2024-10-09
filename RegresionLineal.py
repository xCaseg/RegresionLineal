
"""
La fórmula de regresión lineal es la siguiente

Y =B0 + B1x + e

donde:
Y es la predicción de datos
B1 es la inclianción de la recta
B0 es la intersección en el eje y
e es un factor de error

Obtener la gráfica.
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.array([3,7,11,15,18,27,29,30,30,31,31,32,33,33,34,36,36,36,37,38,39,39,39,40,41,42,42,43,44,45,46,47,50]) # Reducción de solidos %
y = np.array([5,11,21,16,16,28,27,25,35,30,40,32,34,32,34,37,38,34,36,38,37,36,45,39,41,40,44,37,44,46,46,49,51]) # Reducción de la demanda de Oxígeno

# Sumatoria de X
def SumatoriaX(x):
    sx = 0
    for dato_solido in x:
        sx += dato_solido
    return sx

# Sumatoria de Y
def SumatoriaY(y):
    sy = 0
    for dato_oxigeno in y:
        sy += dato_oxigeno
    return sy

# Sumatoria de XY
def SumatoriaXY(x,y):
    sxy = 0
    for i in range(len(x)):
        sxy += x[i] * y[i]
    return sxy

# Sumatoria de X^2
def SumatoriaX2(x):
    sx2 = 0
    for dato_sum in x:
        sx2 += dato_sum * dato_sum
    return sx2

# Calcular sumatorias
sx = SumatoriaX(x) # ΣX
sy = SumatoriaY(y) # ΣY
sxy = SumatoriaXY(x, y) #Σ X*Y
sx2 = SumatoriaX2(x) # X^2

# Calcular b1 y b0
n = len(x)
b1 = (n * sxy - sx * sy) / (n * sx2 - sx ** 2) # b1 = n * Σxy - (Σx) * (Σy) / n (Σx^2) - (Σx)^2
b0 = (sy - b1 * sx) / n # b0 = Σy -(b1 * Σx)

# Calcular error
#e = np.arange(len(x))  # Crear un array de 0 a len(x)-1

# Calcular b0 + b1x
m = b0 + (b1 * x) # μ y|x =

# Datos
print(f"\nArray x: {x}")
print(f"Array y: {y}")
print(f"Número de elementos (n): {n}")
print(f"Sumatoria de X: {sx}")
print(f"Sumatoria de Y: {sy}")
print(f"Sumatoria de XY: {sxy}")
print(f"Sumatoria de X^2: {sx2}")
print(f"Beta 1 (B1): {b1}")
print(f"Beta 0 (B0): {b0}\n")

# Graficar
plt.scatter(x, y, label='Datos', color='green')
plt.plot(x, m, color='blue', label='Regresión lineal')
plt.ylabel("Reducción de sólidos")
plt.xlabel("Reducción demanda oxígeno")
plt.legend()
plt.show()