import numpy as np
def equiparar_columnas(ds_train_3, ds_val_3):
    print("🔄 Sincronizando columnas entre train y val...")
    
    # Obtener columnas de features (sin targets)
    train_feature_cols = set([col for col in ds_train_3.columns if not col.startswith('target_estacion_')])
    val_feature_cols = set([col for col in ds_val_3.columns if not col.startswith('target_estacion_')])
    
    # Obtener columnas de targets
    train_target_cols = set([col for col in ds_train_3.columns if col.startswith('target_estacion_')])
    val_target_cols = set([col for col in ds_val_3.columns if col.startswith('target_estacion_')])
    
    print(f"📊 Features - Train: {len(train_feature_cols)}, Val: {len(val_feature_cols)}")
    print(f"🎯 Targets - Train: {len(train_target_cols)}, Val: {len(val_target_cols)}")
    
    # ✅ NUEVO: Encontrar columnas que están en train pero NO en val
    features_only_in_train = train_feature_cols - val_feature_cols
    targets_only_in_train = train_target_cols - val_target_cols
    
    # Encontrar columnas faltantes en val
    missing_features_in_val = train_feature_cols - val_feature_cols
    missing_targets_in_val = train_target_cols - val_target_cols
    
    print(f"❌ Features solo en train (serán eliminadas): {len(features_only_in_train)}")
    print(f"❌ Targets solo en train (serán eliminadas): {len(targets_only_in_train)}")
    print(f"❌ Features faltantes en val: {len(missing_features_in_val)}")
    print(f"❌ Targets faltantes en val: {len(missing_targets_in_val)}")
    
    if features_only_in_train:
        print(f"   Primeras 5 features a eliminar de train: {list(features_only_in_train)[:5]}")
    if targets_only_in_train:
        print(f"   Primeras 5 targets a eliminar de train: {list(targets_only_in_train)[:5]}")
    
    # ✅ NUEVO: Crear train sin las columnas que no están en val
    ds_train_3_fixed = ds_train_3.copy()
    
    # Eliminar de train las columnas que no están en val
    columns_to_drop_from_train = features_only_in_train | targets_only_in_train
    for col in columns_to_drop_from_train:
        if col in ds_train_3_fixed.columns:
            ds_train_3_fixed = ds_train_3_fixed.drop(columns=[col])
            print(f"   ✅ Eliminada de train: {col}")
    
    # Agregar a val las columnas faltantes con valor 0
    ds_val_3_fixed = ds_val_3.copy()
    
    # Recalcular columnas faltantes después de eliminar de train
    train_feature_cols_final = set([col for col in ds_train_3_fixed.columns if not col.startswith('target_estacion_')])
    train_target_cols_final = set([col for col in ds_train_3_fixed.columns if col.startswith('target_estacion_')])
    
    missing_features_in_val_final = train_feature_cols_final - val_feature_cols
    missing_targets_in_val_final = train_target_cols_final - val_target_cols
    
    for col in missing_features_in_val_final:
        ds_val_3_fixed[col] = 0
        print(f"   ✅ Agregada a val: {col}")
    
    for col in missing_targets_in_val_final:
        ds_val_3_fixed[col] = 0
        print(f"   ✅ Agregada a val: {col}")
    
    # Reordenar columnas para que coincidan
    train_columns_order = list(ds_train_3_fixed.columns)
    ds_val_3_fixed = ds_val_3_fixed[train_columns_order]
    
    print(f"✅ Train sincronizado - Shape: {ds_train_3_fixed.shape}")
    print(f"✅ Val sincronizado - Shape: {ds_val_3_fixed.shape}")
    print(f"✅ Columnas coinciden: {list(ds_train_3_fixed.columns) == list(ds_val_3_fixed.columns)}")
    
    # Actualizar las variables X_val e y_val
    feature_columns = [col for col in ds_val_3_fixed.columns if not col.startswith('target_estacion_')]
    target_columns = [col for col in ds_val_3_fixed.columns if col.startswith('target_estacion_')]
    
    # Extraer estaciones y ordenar
    estaciones_ids = [int(col.replace('target_estacion_', '')) for col in target_columns]
    estaciones_ids_sorted = sorted(estaciones_ids)
    target_columns_sorted_val = [f'target_estacion_{est_id}' for est_id in estaciones_ids_sorted]
    
    # Actualizar X_val e y_val
    X_val_fixed = ds_val_3_fixed[feature_columns]
    y_val_fixed = ds_val_3_fixed[target_columns_sorted_val].values.tolist()
    y_val_array_fixed = np.array(y_val_fixed)
    
    # ✅ NUEVO: También actualizar X_train e y_train
    X_train_fixed = ds_train_3_fixed[feature_columns]
    y_train_fixed = ds_train_3_fixed[target_columns_sorted_val].values.tolist()
    y_train_array_fixed = np.array(y_train_fixed)
    
    print(f"🎯 X_train_fixed shape: {X_train_fixed.shape}")
    print(f"🎯 y_train_fixed shape: {y_train_array_fixed.shape}")
    print(f"🎯 X_val_fixed shape: {X_val_fixed.shape}")
    print(f"🎯 y_val_fixed shape: {y_val_array_fixed.shape}")
    
    # Usar datos sincronizados correctamente
    print("🔮 Preparando datos sincronizados...")
    
    X_train_clean_fixed = X_train_fixed.drop(columns=['fecha_hora'])
    X_val_clean_fixed = X_val_fixed.drop(columns=['fecha_hora'])

    
    # Mezclar train (X e y deben mantenerse sincronizados)
    train_indices = np.random.permutation(len(X_train_clean_fixed))
    X_train_clean_fixed = X_train_clean_fixed.iloc[train_indices].reset_index(drop=True)
    y_train_array_fixed = y_train_array_fixed[train_indices]
    
    # Mezclar val (X e y deben mantenerse sincronizados)
    val_indices = np.random.permutation(len(X_val_clean_fixed))
    X_val_clean_fixed = X_val_clean_fixed.iloc[val_indices].reset_index(drop=True)
    y_val_array_fixed = y_val_array_fixed[val_indices]
    
    
    print(f"📊 Verificando dimensiones finales:")
    print(f"   • X_train shape: {X_train_clean_fixed.shape}")
    print(f"   • X_val shape: {X_val_clean_fixed.shape}")
    print(f"   • Columnas coinciden: {list(X_train_clean_fixed.columns) == list(X_val_clean_fixed.columns)}")
    
    return (y_train_array_fixed, X_train_clean_fixed, 
            y_val_array_fixed, X_val_clean_fixed, target_columns_sorted_val)
