import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, PoissonRegressor
#Importo gradient boosting para comparar
from sklearn.ensemble import GradientBoostingRegressor

#Importo una red neuronal para comparar
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import random

estaciones = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150]

proms_mae = []
proms_rmse = []
proms_mse = []
proms_r2 = []

for estacion in range(1, 380):
    if estacion == 359 or estacion == 255 or estacion == 29 or estacion == 206: #La 29 es rara
        continue
    # Cargar el dataset
    dataset_train_path = f"data/processed/features1/train/dataset_train_{estacion}.csv"
    dataset_val_path = f"data/processed/features1/validation/dataset_val_{estacion}.csv"
    
    if not os.path.exists(dataset_train_path) or not os.path.exists(dataset_val_path):
        print(f"Dataset no encontrado para la estación {estacion}.")
        continue
    
    dataset_train = pd.read_csv(dataset_train_path)
    dataset_val = pd.read_csv(dataset_val_path)
    
    # Diagnóstico inicial
    print(f"\n=== ESTACIÓN {estacion} ===")
    print(f"Train shape: {dataset_train.shape}")
    print(f"Val shape: {dataset_val.shape}")
    
    # Borrar las columnas fecha_hora, fecha, año
    cols_to_drop = ['fecha_hora', 'fecha', 'año', 'id_estacion']
    existing_cols = [col for col in cols_to_drop if col in dataset_train.columns]
    
    if existing_cols:
        dataset_train.drop(columns=existing_cols, inplace=True)
        dataset_val.drop(columns=existing_cols, inplace=True)
    
    # Verificar que existe la columna target
    if 'target' not in dataset_train.columns:
        print(f"ERROR: No se encuentra la columna 'target' en estación {estacion}")
        print(f"Columnas disponibles: {list(dataset_train.columns)}")
        continue
    
    # Separar características y objetivo
    X_train = dataset_train.drop(columns=['target'])
    y_train = dataset_train['target']
    X_val = dataset_val.drop(columns=['target'])
    y_val = dataset_val['target']
    
    # # DIAGNÓSTICOS IMPORTANTES
    # print(f"Target train - Min: {y_train.min():.2f}, Max: {y_train.max():.2f}, Mean: {y_train.mean():.2f}")
    # print(f"Target val - Min: {y_val.min():.2f}, Max: {y_val.max():.2f}, Mean: {y_val.mean():.2f}")
    
    # Verificar NaN
    if X_train.isnull().sum().sum() > 0 or y_train.isnull().sum() > 0:
        print(f"WARNING: Hay valores NaN en train")
        X_train = X_train.fillna(X_train.mean())
        y_train = y_train.fillna(y_train.mean())
    
    if X_val.isnull().sum().sum() > 0 or y_val.isnull().sum() > 0:
        print(f"WARNING: Hay valores NaN en validation")
        X_val = X_val.fillna(X_train.mean()) #Para no hacer leaking
        y_val = y_val.fillna(y_train.mean())
    
    # Verificar valores infinitos
    if np.isinf(X_train).any().any() or np.isinf(y_train).any():
        print(f"WARNING: Hay valores infinitos en train")

    #Desordeno el dataset
    X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
    y_train = y_train.sample(frac=1, random_state=42).reset_index(drop=True)
    X_val = X_val.sample(frac=1, random_state=42).reset_index(drop=True)
    y_val = y_val.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Entrenar la regresión lineal
    model = LinearRegression()
    #model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=13) #el de 3 dio mejor
    #model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=100, random_state=13)  # Red neuronal simple
    #model = PoissonRegressor(alpha=1.0, max_iter=1000, tol=1e-4)
    model.fit(X_train, y_train)
    
    
    # Realizar predicciones
    predictions = model.predict(X_val)


    
    # Diagnóstico de predicciones
    print(f"Predictions - Min: {predictions.min():.2f}, Max: {predictions.max():.2f}, Mean: {predictions.mean():.2f}")

    
    #Restringir las predicciones a mayor o igual a 0
    predictions = np.clip(predictions, 0, None)

    #Discretizar las predicciones a enteros
    predictions = np.round(predictions).astype(int)


    # Diagnóstico de predicciones
    print(f"Predictions - Min: {predictions.min():.2f}, Max: {predictions.max():.2f}, Mean: {predictions.mean():.2f}")
    
    # Calcular métricas (usando sklearn para MAE también)
    mse = mean_squared_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, predictions)  # Usar sklearn
    
    # Métricas adicionales para diagnóstico
    baseline_mae = mean_absolute_error(y_val, [y_train.mean()] * len(y_val))  # Baseline: predecir la media
    
    # Almacenar las métricas
    proms_mae.append((mae, estacion))
    proms_rmse.append((rmse, estacion))
    proms_mse.append((mse, estacion))
    proms_r2.append((r2, estacion))
    
    print(f"RMSE: {rmse:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Baseline MAE (predecir media): {baseline_mae:.6f}")
    print(f"¿Mejor que baseline?: {'Sí' if mae < baseline_mae else 'No'}")
    
    # Verificar si el R² es negativo (muy malo)
    if r2 < 0:
        print("⚠️  WARNING: R² negativo indica que el modelo es peor que predecir la media")
    
    print("-" * 50)

# Calcular promedios
if proms_mae:  # Solo si hay datos
    promedio_mae = np.mean([x[0] for x in proms_mae])
    promedio_rmse = np.mean([x[0] for x in proms_rmse])
    promedio_mse = np.mean([x[0] for x in proms_mse])
    promedio_r2 = np.mean([x[0] for x in proms_r2])
    
    print(f"\n=== PROMEDIOS FINALES ===")
    print(f"Promedio MAE: {promedio_mae:.6f}")
    print(f"Promedio RMSE: {promedio_rmse:.6f}")
    print(f"Promedio MSE: {promedio_mse:.6f}")
    print(f"Promedio R²: {promedio_r2:.6f}")
    
    # Interpretación
    print(f"\n=== INTERPRETACIÓN ===")
    if promedio_r2 > 0.7:
        print("✅ Buen ajuste del modelo")
    elif promedio_r2 > 0.3:
        print("⚠️  Ajuste moderado del modelo")
    else:
        print("❌ Ajuste pobre del modelo")
else:
    print("No se procesaron datos")


print("EL MAE MAS GRANDE ES: ", max(proms_mae, key=lambda x: x[0]), "estacion: ", max(proms_mae, key=lambda x: x[0])[1])
print("EL MSE MAS GRANDE ES: ", max(proms_mse, key=lambda x: x[0]), "estacion: ", max(proms_mse, key=lambda x: x[0])[1])
print("EL RMSE MAS GRANDE ES: ", max(proms_rmse, key=lambda x: x[0]), "estacion: ", max(proms_rmse, key=lambda x: x[0])[1])
print("EL R2 MAS GRANDE ES: ", max(proms_r2, key=lambda x: x[0]), "estacion: ", max(proms_r2, key=lambda x: x[0])[1])