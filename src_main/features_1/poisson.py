import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os


proms_mae = []
proms_rmse = []
proms_mse = []
proms_r2 = []

def calcular_metricas_poisson(dataset_train, dataset_val, estacion):
    
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
 
    
    # Separar características y objetivo
    X_train = dataset_train.drop(columns=['target'])
    y_train = dataset_train['target']
    X_val = dataset_val.drop(columns=['target'])
    y_val = dataset_val['target']

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
    
    # Entrenar el modelo de regresión de Poisson
    model = PoissonRegressor(alpha=1.0, max_iter=1000, tol=1e-4)
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
    baseline_mse = mean_squared_error(y_val, [y_train.mean()] * len(y_val))
    baseline_rmse = np.sqrt(baseline_mse)
    
    # Almacenar las métricas
    proms_mae.append(mae)
    proms_rmse.append(rmse)
    proms_mse.append(mse)
    proms_r2.append(r2)
    
    
    return model, rmse, mse, mae, r2, proms_mae, proms_rmse, proms_mse, proms_r2, baseline_mae, baseline_mse, baseline_rmse, y_val, predictions