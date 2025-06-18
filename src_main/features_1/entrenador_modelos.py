import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_vals = []
predicciones = []


proms_mae = []
proms_rmse = []
proms_mse = []
proms_r2 = []


def entrenador_modelos(regresor):
        
    #Pruebo predecir con una poisson para tener eso como base
    for estacion in range(1, 380):
        ds_train_path = f"/Users/isabelcastaneda/Desktop/Machine Learning Udesa/proyecto_final/BA_BICIS_MyC/data/processed/features1/train/dataset_train_{estacion}.csv"
        ds_val_path = f"/Users/isabelcastaneda/Desktop/Machine Learning Udesa/proyecto_final/BA_BICIS_MyC/data/processed/features1/validation/dataset_val_{estacion}.csv"

        if os.path.exists(ds_train_path) and os.path.exists(ds_val_path):
            print(f"Procesando estación {estacion}...")
        else:
            print(f"Dataset no encontrado para la estación {estacion}.")
            continue



        if not os.path.exists(ds_train_path) or not os.path.exists(ds_val_path):
            print(f"Dataset no encontrado para la estación {estacion}.")
            continue
        dataset_train = pd.read_csv(ds_train_path)
        dataset_val = pd.read_csv(ds_val_path)


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


        if estacion == 11:
            print(y_val)


        
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
        if np.isinf(X_train.select_dtypes(include=[np.number])).any().any() or np.isinf(y_train).any():
            print(f"WARNING: Hay valores infinitos en train")
            print(f"WARNING: Hay valores infinitos en train")

        #Desordeno el dataset
        X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
        y_train = y_train.sample(frac=1, random_state=42).reset_index(drop=True)
        X_val = X_val.sample(frac=1, random_state=42).reset_index(drop=True)
        y_val = y_val.sample(frac=1, random_state=42).reset_index(drop=True)
        
        
        model = regresor
        model.fit(X_train, y_train)




        # Realizar predicciones
        predictions = model.predict(X_val)
        
        # Diagnóstico de predicciones
        print(f"Predictions - Min: {predictions.min():.2f}, Max: {predictions.max():.2f}, Mean: {predictions.mean():.2f}")

        
        #Restringir las predicciones a mayor o igual a 0
        predictions = np.clip(predictions, 0, None)

        #Discretizar las predicciones a enteros
        #predictions = np.round(predictions).astype(int)


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

        y_vals.append(y_val)
        predicciones.append(predictions)

    return model, proms_mae, proms_rmse, proms_mse, proms_r2, baseline_mae, baseline_mse, baseline_rmse, y_vals, predicciones