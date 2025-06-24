import numpy as np
import pandas as pd

def filtrar_dataset_por_estaciones(df, estaciones_incluir, verbose=True):
    """
    Filtra el dataset para incluir solo las features de las estaciones especificadas.
    Agrega columnas totales para compensar las features excluidas.
    
    Args:
        df (pd.DataFrame): DataFrame original con todas las features
        estaciones_incluir (list): Lista de IDs de estaciones a incluir
        verbose (bool): Si True, muestra información detallada del proceso
    
    Returns:
        pd.DataFrame: DataFrame filtrado con features solo de las estaciones especificadas
                     y columnas totales agregadas
    """
    
    df_filtered = df.copy()
    
    if verbose:
        print(f"=== FILTRADO POR ESTACIONES ===")
        print(f"Estaciones a incluir: {estaciones_incluir}")
        print(f"Shape original: {df.shape}")
    
    # 1. Identificar y filtrar columnas de bicis_salieron_estacion_x
    bicis_salieron_cols = [col for col in df.columns if col.startswith('bicis_salieron_estacion_')]
    bicis_salieron_incluir = []
    bicis_salieron_excluir = []
    
    for col in bicis_salieron_cols:
        # Extraer el número de estación del nombre de la columna
        try:
            # Formato esperado: bicis_salieron_estacion_X
            estacion_num = int(col.split('_')[-1])
            if estacion_num in estaciones_incluir:
                bicis_salieron_incluir.append(col)
            else:
                bicis_salieron_excluir.append(col)
        except:
            # Si no se puede extraer el número, mantener la columna
            bicis_salieron_incluir.append(col)
    
    if verbose:
        print(f"Bicis salieron - Incluir: {len(bicis_salieron_incluir)} columnas")
        print(f"Bicis salieron - Excluir: {len(bicis_salieron_excluir)} columnas")
    
    # Crear columna total de bicis salieron (suma de las excluidas)
    if bicis_salieron_excluir:
        df_filtered['bicis_salieron_total'] = df[bicis_salieron_excluir].sum(axis=1)
        if verbose:
            print(f"✓ Agregada columna 'bicis_salieron_total' (suma de {len(bicis_salieron_excluir)} estaciones)")
    
    # Eliminar columnas de bicis salieron excluidas
    df_filtered = df_filtered.drop(columns=bicis_salieron_excluir)
    
    # 2. Identificar y filtrar columnas de llegadas_estacion_X_hY
    llegadas_cols = [col for col in df.columns if col.startswith('llegadas_estacion_') and '_h' in col]
    llegadas_incluir = []
    llegadas_excluir = []
    
    for col in llegadas_cols:
        try:
            # Formato esperado: llegadas_estacion_X_hY
            partes = col.split('_')
            if len(partes) >= 3:
                estacion_num = int(partes[2])  # posición de X en llegadas_estacion_X_hY
                if estacion_num in estaciones_incluir:
                    llegadas_incluir.append(col)
                else:
                    llegadas_excluir.append(col)
        except:
            # Si no se puede extraer el número, mantener la columna
            llegadas_incluir.append(col)
    
    if verbose:
        print(f"Llegadas lag - Incluir: {len(llegadas_incluir)} columnas")
        print(f"Llegadas lag - Excluir: {len(llegadas_excluir)} columnas")
    
    # Agrupar llegadas excluidas por hora para crear totales
    horas_disponibles = set()
    for col in llegadas_excluir:
        try:
            # Extraer la hora del formato llegadas_estacion_X_hY
            hora = col.split('_h')[-1]
            horas_disponibles.add(hora)
        except:
            continue
    
    # Crear columnas totales de llegadas por hora
    for hora in sorted(horas_disponibles):
        cols_hora = [col for col in llegadas_excluir if col.endswith(f'_h{hora}')]
        if cols_hora:
            nombre_total = f'llegadas_total_h{hora}'
            df_filtered[nombre_total] = df[cols_hora].sum(axis=1)
            if verbose:
                print(f"✓ Agregada columna '{nombre_total}' (suma de {len(cols_hora)} estaciones)")
    
    # Eliminar columnas de llegadas excluidas
    df_filtered = df_filtered.drop(columns=llegadas_excluir)
    
    # 3. Mostrar resumen de features incluidas para las estaciones seleccionadas
    if verbose:
        print(f"\n📊 FEATURES INCLUIDAS POR ESTACIÓN:")
        for estacion in estaciones_incluir:
            bicis_cols = [col for col in bicis_salieron_incluir if col.endswith(f'_{estacion}')]
            llegadas_cols_est = [col for col in llegadas_incluir if f'_estacion_{estacion}_' in col]
            
            print(f"Estación {estacion}:")
            print(f"  - Bicis salieron: {len(bicis_cols)} columnas")
            print(f"  - Llegadas lag: {len(llegadas_cols_est)} columnas")
            if llegadas_cols_est:
                horas_est = [col.split('_h')[-1] for col in llegadas_cols_est]
                print(f"    Horas disponibles: {sorted(set(horas_est))}")
        
        print(f"\nShape final: {df_filtered.shape}")
        print(f"Columnas eliminadas: {len(df.columns) - len(df_filtered.columns)}")
        print(f"Columnas agregadas: {len([col for col in df_filtered.columns if col not in df.columns])}")
    else:
        # Print básico cuando verbose=False
        print(f"Dataset filtrado: {df.shape} → {df_filtered.shape} (estaciones: {estaciones_incluir})")
    
    return df_filtered

def dividir_dataset_estacion(df, estacion_id, verbose=True):
    """
    Divide el dataset en X (features) e y (target específico de una estación).
    
    Args:
        df (pd.DataFrame): DataFrame con features y targets
        estacion_id (int): ID de la estación para la cual extraer el target
        verbose (bool): Si True, muestra información detallada del proceso
    
    Returns:
        tuple: (X, y, feature_names)
            - X: DataFrame con todas las features (excluye columnas target_*)
            - y: Serie con el target de la estación específica
            - feature_names: Lista con nombres de las features
    """
    
    # Identificar el nombre de la columna target específica
    target_column = f'target_estacion_{estacion_id}'
    
    # Verificar que la columna target existe
    if target_column not in df.columns:
        available_targets = [col for col in df.columns if col.startswith('target_')]
        raise ValueError(f"Columna '{target_column}' no encontrada. Targets disponibles: {available_targets}")
    
    # Identificar columnas de features (todas excepto las que empiezan con 'target_')
    feature_columns = [col for col in df.columns if not col.startswith('target_')]
    
    # Crear X e y
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    if verbose:
        print(f"=== DIVISIÓN DATASET ESTACIÓN {estacion_id} ===")
        print(f"Shape original: {df.shape}")
        print(f"Features (X): {X.shape}")
        print(f"Target (y): {y.shape}")
        print(f"Target columna: {target_column}")
        print(f"Rango de y: {y.min():.2f} - {y.max():.2f}")
        print(f"Media de y: {y.mean():.2f}")
        print(f"Features incluidos: {len(feature_columns)}")
        
        # Mostrar algunas estadísticas del target
        print(f"\n📊 ESTADÍSTICAS DEL TARGET:")
        print(f"  - Valores nulos: {y.isnull().sum()}")
        print(f"  - Valores cero: {(y == 0).sum()}")
        print(f"  - Percentiles: 25%={y.quantile(0.25):.2f}, 50%={y.quantile(0.5):.2f}, 75%={y.quantile(0.75):.2f}")
    else:
        # Print básico cuando verbose=False
        print(f"Dataset dividido estación {estacion_id}: X{X.shape}, y{y.shape}")
    
    return X, y, feature_columns


def dividir_dataset_multiples_estaciones(df, estaciones_ids, verbose=True):
    """
    Divide el dataset en X (features) e y (targets de múltiples estaciones).
    
    Args:
        df (pd.DataFrame): DataFrame con features y targets
        estaciones_ids (list): Lista de IDs de estaciones para extraer targets
        verbose (bool): Si True, muestra información detallada del proceso
    
    Returns:
        tuple: (X, y, feature_names, target_names)
            - X: DataFrame con todas las features
            - y: DataFrame con los targets de las estaciones especificadas
            - feature_names: Lista con nombres de las features
            - target_names: Lista con nombres de los targets
    """
    
    # Identificar columnas target específicas
    target_columns = []
    for estacion_id in estaciones_ids:
        target_col = f'target_estacion_{estacion_id}'
        if target_col in df.columns:
            target_columns.append(target_col)
        else:
            if verbose:
                print(f"⚠️  Advertencia: {target_col} no encontrada")
    
    if not target_columns:
        available_targets = [col for col in df.columns if col.startswith('target_')]
        raise ValueError(f"Ningún target encontrado para las estaciones {estaciones_ids}. Targets disponibles: {available_targets}")
    
    # Identificar columnas de features
    feature_columns = [col for col in df.columns if not col.startswith('target_')]
    
    # Crear X e y
    X = df[feature_columns].copy()
    y = df[target_columns].copy()
    
    if verbose:
        print(f"=== DIVISIÓN DATASET MÚLTIPLES ESTACIONES ===")
        print(f"Estaciones solicitadas: {estaciones_ids}")
        print(f"Estaciones encontradas: {[int(col.split('_')[-1]) for col in target_columns]}")
        print(f"Shape original: {df.shape}")
        print(f"Features (X): {X.shape}")
        print(f"Targets (y): {y.shape}")
        
        # Estadísticas de cada target
        print(f"\n📊 ESTADÍSTICAS POR TARGET:")
        for col in target_columns:
            estacion = col.split('_')[-1]
            target_data = y[col]
            print(f"  Estación {estacion}:")
            print(f"    - Rango: {target_data.min():.2f} - {target_data.max():.2f}")
            print(f"    - Media: {target_data.mean():.2f}")
            print(f"    - Valores nulos: {target_data.isnull().sum()}")
            print(f"    - Valores cero: {(target_data == 0).sum()}")
    else:
        # Print básico cuando verbose=False
        estaciones_encontradas = [int(col.split('_')[-1]) for col in target_columns]
        print(f"Dataset dividido múltiples estaciones {estaciones_encontradas}: X{X.shape}, y{y.shape}")
    
    return X, y, feature_columns, target_columns


def obtener_targets_disponibles(df, verbose=True):
    """
    Obtiene información sobre todos los targets disponibles en el dataset.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
        verbose (bool): Si True, muestra información detallada del análisis
    
    Returns:
        dict: Información sobre los targets disponibles
    """
    
    # Encontrar todas las columnas target
    target_columns = [col for col in df.columns if col.startswith('target_')]
    
    # Extraer IDs de estaciones
    estaciones_ids = []
    otros_targets = []
    
    for col in target_columns:
        if col.startswith('target_estacion_'):
            try:
                estacion_id = int(col.split('_')[-1])
                estaciones_ids.append(estacion_id)
            except:
                otros_targets.append(col)
        else:
            otros_targets.append(col)
    
    estaciones_ids = sorted(estaciones_ids)
    
    info = {
        'total_targets': len(target_columns),
        'estaciones_ids': estaciones_ids,
        'targets_estaciones': [f'target_estacion_{id}' for id in estaciones_ids],
        'otros_targets': otros_targets,
        'todos_targets': target_columns
    }
    
    if verbose:
        print(f"=== TARGETS DISPONIBLES ===")
        print(f"Total de targets: {len(target_columns)}")
        print(f"Targets de estaciones: {len(estaciones_ids)}")
        print(f"  Estaciones: {estaciones_ids}")
        print(f"Otros targets: {len(otros_targets)}")
        if otros_targets:
            print(f"  Nombres: {otros_targets}")
        
        # Mostrar estadísticas básicas
        if estaciones_ids:
            print(f"\n📊 ESTADÍSTICAS RÁPIDAS:")
            sample_targets = [f'target_estacion_{id}' for id in estaciones_ids[:3]]
            for col in sample_targets:
                if col in df.columns:
                    data = df[col]
                    print(f"  {col}: media={data.mean():.2f}, std={data.std():.2f}")
    else:
        # Print básico cuando verbose=False
        print(f"Targets disponibles: {len(estaciones_ids)} estaciones, {len(otros_targets)} otros")
    
    return info


def crear_dataset_completo_estacion(df, estacion_id, verbose=True):
    """
    Función completa que filtra el dataset por estación y lo divide en X, y.
    Combina filtrar_dataset_por_estaciones() con dividir_dataset_estacion().
    
    Args:
        df (pd.DataFrame): DataFrame original
        estacion_id (int): ID de la estación
        verbose (bool): Si True, muestra información detallada del proceso
    
    Returns:
        tuple: (X, y, feature_names, df_filtrado)
            - X: Features filtradas para la estación
            - y: Target de la estación específica
            - feature_names: Nombres de las features
            - df_filtrado: Dataset completo filtrado (para inspección)
    """
    
    if verbose:
        print(f"=== CREACIÓN DATASET COMPLETO ESTACIÓN {estacion_id} ===")
    
    # Paso 1: Filtrar dataset por estación
    df_filtrado = filtrar_dataset_por_estaciones(df, [estacion_id], verbose=verbose)
    
    # Paso 2: Dividir en X, y
    X, y, feature_names = dividir_dataset_estacion(df_filtrado, estacion_id, verbose=verbose)
    
    if verbose:
        print(f"\n✅ DATASET COMPLETO CREADO:")
        print(f"  - Dataset filtrado: {df_filtrado.shape}")
        print(f"  - Features (X): {X.shape}")
        print(f"  - Target (y): {y.shape}")
        print(f"  - Features específicas de estación {estacion_id}: incluidas")
        print(f"  - Features totales compensatorias: incluidas")
    else:
        # Print básico cuando verbose=False
        print(f"Dataset completo estación {estacion_id}: X{X.shape}, y{y.shape}")
    
    return X, y, feature_names, df_filtrado


def obtener_info_estaciones_dataset(df, verbose=True):
    """
    Analiza el dataset para mostrar información sobre las estaciones disponibles.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
        verbose (bool): Si True, muestra información detallada del análisis
    
    Returns:
        dict: Información sobre las estaciones en el dataset
    """
    
    info = {
        'estaciones_bicis_salieron': [],
        'estaciones_llegadas_lag': [],
        'horas_disponibles': set(),
        'total_features_bicis': 0,
        'total_features_llegadas': 0
    }
    
    # Analizar columnas de bicis salieron
    bicis_cols = [col for col in df.columns if col.startswith('bicis_salieron_estacion_')]
    for col in bicis_cols:
        try:
            estacion_num = int(col.split('_')[-1])
            info['estaciones_bicis_salieron'].append(estacion_num)
        except:
            continue
    
    # Analizar columnas de llegadas lag
    llegadas_cols = [col for col in df.columns if col.startswith('llegadas_estacion_') and '_h' in col]
    for col in llegadas_cols:
        try:
            partes = col.split('_')
            if len(partes) >= 4:
                estacion_num = int(partes[2])
                hora = col.split('_h')[-1]
                info['estaciones_llegadas_lag'].append(estacion_num)
                info['horas_disponibles'].add(hora)
        except:
            continue
    
    # Remover duplicados y ordenar
    info['estaciones_bicis_salieron'] = sorted(set(info['estaciones_bicis_salieron']))
    info['estaciones_llegadas_lag'] = sorted(set(info['estaciones_llegadas_lag']))
    info['horas_disponibles'] = sorted(info['horas_disponibles'])
    info['total_features_bicis'] = len(bicis_cols)
    info['total_features_llegadas'] = len(llegadas_cols)
    
    if verbose:
        print(f"=== ANÁLISIS DE ESTACIONES EN DATASET ===")
        print(f"Estaciones con features 'bicis_salieron': {len(info['estaciones_bicis_salieron'])}")
        if info['estaciones_bicis_salieron']:
            print(f"  Rango: {min(info['estaciones_bicis_salieron'])}-{max(info['estaciones_bicis_salieron'])}")
            print(f"  Algunas: {info['estaciones_bicis_salieron'][:10]}{'...' if len(info['estaciones_bicis_salieron']) > 10 else ''}")
        
        print(f"Estaciones con features 'llegadas_lag': {len(info['estaciones_llegadas_lag'])}")
        if info['estaciones_llegadas_lag']:
            print(f"  Rango: {min(info['estaciones_llegadas_lag'])}-{max(info['estaciones_llegadas_lag'])}")
            print(f"  Algunas: {info['estaciones_llegadas_lag'][:10]}{'...' if len(info['estaciones_llegadas_lag']) > 10 else ''}")
        
        print(f"Horas disponibles para llegadas lag: {info['horas_disponibles']}")
        print(f"Total features bicis salieron: {info['total_features_bicis']}")
        print(f"Total features llegadas lag: {info['total_features_llegadas']}")
    else:
        # Print básico cuando verbose=False
        print(f"Estaciones disponibles: {len(info['estaciones_bicis_salieron'])} bicis, {len(info['estaciones_llegadas_lag'])} llegadas")
    
    return info


def crear_dataset_estacion_especifica(df, estacion_id, verbose=True):
    """
    Función de conveniencia para crear un dataset enfocado en una sola estación.
    
    Args:
        df (pd.DataFrame): DataFrame original
        estacion_id (int): ID de la estación a incluir
        verbose (bool): Si True, muestra información detallada del proceso
    
    Returns:
        pd.DataFrame: Dataset filtrado para la estación específica
    """
    
    if verbose:
        print(f"Creando dataset específico para estación {estacion_id}")
    
    return filtrar_dataset_por_estaciones(df, [estacion_id], verbose=verbose)