import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definición de la función f(x, y)
def f(x, y):
    return 10 - np.exp(-((x**2) - (3*y**2)))

# Gradiente de la función f(x, y)
def gradient(x, y):
    df_dx = 2 * x * np.exp(-((x**2) - (3*y**2)))
    df_dy = -6 * y * np.exp(-((x**2) - (3*y**2)))
    return df_dx, df_dy

# Algoritmo de descenso del gradiente
def gradient_descent(learning_rate, num_iterations):
    x = np.random.uniform(-3, 3)  # Valor inicial de x en un rango manejable
    y = np.random.uniform(-3, 3)  # Valor inicial de y en un rango manejable
    history = []  # Almacenar los valores de (x, y) en cada iteración
    
    for i in range(num_iterations):
        df_dx, df_dy = gradient(x, y)
        x -= learning_rate * df_dx
        y -= learning_rate * df_dy
        
        # Limitar los valores de x e y para evitar errores de overflow
        x = np.clip(x, -3, 3)
        y = np.clip(y, -3, 3)
        
        history.append((x, y))
    
    return x, y, history

# Parámetros del descenso del gradiente
learning_rate = 0.0001
num_iterations = 500

# Ejecutar el descenso del gradiente
x_optimal, y_optimal, history = gradient_descent(learning_rate, num_iterations)

# Crear un grid de puntos para graficar la función
x_range = np.linspace(-3, 3, 400)
y_range = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)

# Crear una vista 3D de la función
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y)')
ax1.set_title('Vista 3D de la Función')

# Crear una vista 2D de la función con colores más profundos
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X, Y, Z, levels=100, cmap='viridis')
plt.colorbar(contour, ax=ax2, label='Profundidad')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.plot(*zip(*history), marker='o', linestyle='-', color='black', markersize=4)
ax2.scatter(x_optimal, y_optimal, color='red', marker='x', label='Mínimo', s=100)
ax2.legend()
ax2.set_title('Vista 2D Ampliada de la Función')

# Mostrar las gráficas
plt.tight_layout()
plt.show()


