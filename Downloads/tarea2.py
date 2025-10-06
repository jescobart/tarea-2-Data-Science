import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Datos 
datos = {
    'Superficie_m2': [50, 70, 65, 90, 45, 85, 60, 75, 55],
    'Num_Habitaciones': [1, 2, 2, 3, 1, 3, 2, 2, 1],
    'Distancia_Metro_km': [0.5, 1.2, 0.8, 0.2, 2.0, 0.6, 1.0, 0.9, 1.5],
    'Precio_UF': [2500, 3800, 3500, 5200, 2100, 4800, 3300, 4000, 2600]
}
df = pd.DataFrame(datos)

# Variables independientes (X) y dependiente (y)
X = df[['Superficie_m2', 'Num_Habitaciones', 'Distancia_Metro_km']]
y = df['Precio_UF']

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Evaluar el modelo con los datos de prueba
y_pred = modelo.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Evaluación del modelo de regresión lineal")
print("-----------------------------------------------")
print(f"RMSE: {round(rmse, 2)} → En promedio, las predicciones se desvían en {round(rmse, 2)} UF del precio real.")
print(f"R²: {round(r2, 3)} → El modelo explica el {round(r2*100, 1)}% de la variación en los precios.")
print("-----------------------------------------------")

# Función para predecir precio UF
def predecir_precio():
    print("\nPREDICCIÓN DE NUEVO DEPARTAMENTO")
    superficie = float(input("Ingrese superficie en m²: "))
    habitaciones = int(input("Ingrese número de habitaciones: "))
    distancia = float(input("Ingrese distancia al metro en km: "))

    nuevo_departamento = pd.DataFrame({
        'Superficie_m2': [superficie],
        'Num_Habitaciones': [habitaciones],
        'Distancia_Metro_km': [distancia]
    })

    precio_estimado = modelo.predict(nuevo_departamento)
    print(f"\n Predicción de precio UF: {round(precio_estimado[0], 2)}")
    print("→ Este es el valor estimado según las características ingresadas.")

# Ejecutar función interactiva
predecir_precio()
