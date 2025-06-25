import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
from typing import List, Tuple, Dict, Optional

def crear_dataset_unificado_bicis(df: pd.DataFrame, features_requeridas: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[int]]:
    """
    Crea un dataset unificado OPTIMIZADO para predecir llegadas de bicis por hora.
    Cada fila = una hora específica con toda la info global y por estación.
    
    Args:
        df: DataFrame con los datos de recorridos
        features_requeridas: Lista opcional de features que debe tener el dataset final.
                           Si no está en df, se crea con valor 0.
                           Features en df que no estén en esta lista se eliminan.
    
    Returns:
        Tuple[DataFrame, List]: (dataset_unificado, lista_estaciones)
    """
    
    print("🚀 Iniciando creación de dataset unificado...")
    
    # ============= PREPARACIÓN INICIAL =============
    df = df.copy()
    
    # Convertir fechas de una vez (más eficiente)
    print("📅 Convirtiendo fechas...")
    df['fecha_origen_recorrido'] = pd.to_datetime(df['fecha_origen_recorrido'])
    df['fecha_destino_recorrido'] = pd.to_datetime(df['fecha_destino_recorrido'])
    df['hora_origen'] = df['fecha_origen_recorrido'].dt.floor('H')
    df['hora_destino'] = df['fecha_destino_recorrido'].dt.floor('H')
    
    # Obtener estaciones únicas (optimizado)
    todas_estaciones = sorted(list(set(df['id_estacion_origen'].unique()) | 
                                  set(df['id_estacion_destino'].unique())))
    n_estaciones = len(todas_estaciones)
    
    print(f"🏪 Encontradas {n_estaciones} estaciones únicas")
    
    # Crear rango temporal
    fecha_min = df['fecha_origen_recorrido'].min().floor('H')
    fecha_max = df['fecha_destino_recorrido'].max().floor('H')
    horas_completas = pd.date_range(start=fecha_min, end=fecha_max, freq='H')
    n_horas = len(horas_completas)
    
    print(f"⏰ Rango temporal: {fecha_min} a {fecha_max} ({n_horas:,} horas)")
    print(f"📊 Dataset final tendrá ~{n_horas:,} filas × {5 + 8 + n_estaciones + (n_estaciones * 12) + n_estaciones:,} columnas")
    
    # ============= CREAR DATASET BASE =============
    print("🏗️ Creando estructura base...")
    dataset = pd.DataFrame({'fecha_hora': horas_completas})
    
    # Features temporales (vectorizado)
    dataset['hora'] = dataset['fecha_hora'].dt.hour
    dataset['dia_semana'] = dataset['fecha_hora'].dt.dayofweek
    dataset['mes'] = dataset['fecha_hora'].dt.month
    dataset['año'] = dataset['fecha_hora'].dt.year
    
    # Feriados (optimizado)
    ar_holidays = holidays.Argentina()
    fechas_unicas = dataset['fecha_hora'].dt.date.unique()
    feriados_dict = {fecha: fecha in ar_holidays for fecha in fechas_unicas}
    dataset['es_feriado'] = dataset['fecha_hora'].dt.date.map(feriados_dict).astype(int)
    
    # ============= FEATURES GLOBALES DE SALIDAS =============
    print("🌍 Procesando features globales de salidas...")
    
    # Limpiar datos una sola vez
    df['género_clean'] = df['género'].fillna('OTHER')
    df['edad_clean'] = pd.to_numeric(df['edad_usuario'], errors='coerce')
    df['modelo_clean'] = df['modelo_bicicleta'].fillna('OTHER')
    
    # Agregar estadísticas globales (optimizado con agg múltiple)
    salidas_globales = df.groupby('hora_origen').agg({
        'Id_recorrido': 'count',
        'género_clean': [
            lambda x: (x == 'FEMALE').mean(),
            lambda x: (x == 'MALE').mean()
        ],
        'edad_clean': [
            lambda x: np.nanquantile(x.dropna(), 0.25) if len(x.dropna()) > 0 else 30,
            lambda x: np.nanmean(x.dropna()) if len(x.dropna()) > 0 else 30,
            lambda x: np.nanquantile(x.dropna(), 0.75) if len(x.dropna()) > 0 else 30
        ],
        'modelo_clean': [
            lambda x: (x == 'ICONIC').mean(),
            lambda x: (x == 'FIT').mean()
        ]
    }).reset_index()
    
    # Aplanar columnas
    salidas_globales.columns = [
        'fecha_hora', 'total_bicis_salieron_global',
        'pct_mujeres_salieron_global', 'pct_hombres_salieron_global',
        'q1_edad_salieron_global', 'media_edad_salieron_global', 'q3_edad_salieron_global',
        'pct_iconic_salieron_global', 'pct_fit_salieron_global'
    ]
    
    # Merge optimizado
    dataset = dataset.merge(salidas_globales, on='fecha_hora', how='left')
    
    # Fill NAs eficientemente
    edad_default = df['edad_clean'].mean() if df['edad_clean'].notna().any() else 30
    fillna_dict = {
        'total_bicis_salieron_global': 0,
        'pct_mujeres_salieron_global': 0,
        'pct_hombres_salieron_global': 0,
        'q1_edad_salieron_global': edad_default,
        'media_edad_salieron_global': edad_default,
        'q3_edad_salieron_global': edad_default,
        'pct_iconic_salieron_global': 0,
        'pct_fit_salieron_global': 0
    }
    dataset = dataset.fillna(fillna_dict)
    
    # ============= SALIDAS POR ESTACIÓN =============
    print("🚉 Procesando salidas por estación...")
    
    # Crear tabla pivot (MUY optimizado)
    salidas_pivot = (df.groupby(['hora_origen', 'id_estacion_origen'])
                       .size()
                       .unstack(fill_value=0)
                       .reset_index())
    salidas_pivot.columns.name = None
    
    # Renombrar columnas
    rename_dict = {'hora_origen': 'fecha_hora'}
    for est in todas_estaciones:
        if est in salidas_pivot.columns:
            rename_dict[est] = f'bicis_salieron_estacion_{est}'
    salidas_pivot = salidas_pivot.rename(columns=rename_dict)
    
    # Agregar columnas faltantes si alguna estación no tiene salidas
    for est in todas_estaciones:
        col_name = f'bicis_salieron_estacion_{est}'
        if col_name not in salidas_pivot.columns:
            salidas_pivot[col_name] = 0
    
    # Merge
    dataset = dataset.merge(salidas_pivot, on='fecha_hora', how='left')
    
    # Fill NAs para salidas por estación
    cols_salidas = [f'bicis_salieron_estacion_{est}' for est in todas_estaciones]
    dataset[cols_salidas] = dataset[cols_salidas].fillna(0)
    
    # ============= LLEGADAS POR ESTACIÓN + LAGS =============
    print("🎯 Procesando llegadas por estación con lags...")
    
    # Crear tabla pivot para llegadas (optimizado)
    llegadas_pivot = (df.groupby(['hora_destino', 'id_estacion_destino'])
                        .size()
                        .unstack(fill_value=0)
                        .reset_index())
    llegadas_pivot.columns.name = None
    
    # Renombrar columnas
    rename_dict = {'hora_destino': 'fecha_hora'}
    for est in todas_estaciones:
        if est in llegadas_pivot.columns:
            rename_dict[est] = f'llegadas_estacion_{est}_h0'
    llegadas_pivot = llegadas_pivot.rename(columns=rename_dict)
    
    # Agregar columnas faltantes
    for est in todas_estaciones:
        col_name = f'llegadas_estacion_{est}_h0'
        if col_name not in llegadas_pivot.columns:
            llegadas_pivot[col_name] = 0
    
    # Merge llegadas
    dataset = dataset.merge(llegadas_pivot, on='fecha_hora', how='left')
    
    # Fill NAs para llegadas
    cols_llegadas_h0 = [f'llegadas_estacion_{est}_h0' for est in todas_estaciones]
    dataset[cols_llegadas_h0] = dataset[cols_llegadas_h0].fillna(0)
    
    # Crear lags de llegadas (OPTIMIZADO - vectorizado)
    print("⏮️ Creando lags de llegadas...")
    lags = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]
    
    for est in todas_estaciones:
        col_base = f'llegadas_estacion_{est}_h0'
        for lag in lags:
            dataset[f'llegadas_estacion_{est}_h{lag}'] = dataset[col_base].shift(lag, fill_value=0)
    
    # ============= CREAR TARGETS =============
    print("🎯 Creando variables target...")
    for est in todas_estaciones:
        dataset[f'target_estacion_{est}'] = dataset[f'llegadas_estacion_{est}_h0'].shift(-1)
    
    # ============= LIMPIEZA FINAL =============
    print("🧹 Limpieza final...")
    
    # Eliminar última fila (sin targets) y columnas h0
    dataset = dataset[:-1].copy()
    cols_to_drop = [col for col in dataset.columns if col.endswith('_h0')]
    dataset = dataset.drop(columns=cols_to_drop)
    
    # Ordenar por fecha
    dataset = dataset.sort_values('fecha_hora').reset_index(drop=True)
    
    # ============= FILTRADO DE FEATURES =============
    if features_requeridas is not None:
        print("🔧 Aplicando filtrado de features...")
        
        # Features actuales en el dataset
        features_actuales = set(dataset.columns)
        features_requeridas_set = set(features_requeridas)
        
        # Features que se necesitan agregar (con valor 0)
        features_agregar = features_requeridas_set - features_actuales
        
        # Features que se van a eliminar
        features_eliminar = features_actuales - features_requeridas_set
        
        # Agregar features faltantes con valor 0
        if features_agregar:
            print(f"➕ Agregando {len(features_agregar)} features faltantes con valor 0:")
            for feature in sorted(features_agregar):
                dataset[feature] = 0
                print(f"   + {feature}")
        
        # Eliminar features no requeridas
        if features_eliminar:
            print(f"➖ Eliminando {len(features_eliminar)} features no requeridas:")
            for feature in sorted(features_eliminar):
                print(f"   - {feature}")
            dataset = dataset.drop(columns=list(features_eliminar))
        
        # Reordenar columnas según el orden de features_requeridas
        # (manteniendo las que existen)
        columnas_ordenadas = [col for col in features_requeridas if col in dataset.columns]
        dataset = dataset[columnas_ordenadas]
        
        print(f"✅ Filtrado completado: {len(dataset.columns)} features finales")
    
    print(f"✅ Dataset creado: {len(dataset):,} filas × {len(dataset.columns):,} columnas")
    print(f"📈 Memoria aproximada: {dataset.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Mostrar muestra del dataset
    print("\n📋 Primeras 3 filas del dataset:")
    if features_requeridas is None:
        cols_muestra = ['fecha_hora', 'hora', 'dia_semana', 'total_bicis_salieron_global', 
                       'pct_mujeres_salieron_global'] + [f'bicis_salieron_estacion_{todas_estaciones[0]}'] + \
                       [f'llegadas_estacion_{todas_estaciones[0]}_h1'] + [f'target_estacion_{todas_estaciones[0]}']
        cols_muestra = [col for col in cols_muestra if col in dataset.columns]
    else:
        # Mostrar las primeras 5 columnas del dataset filtrado
        cols_muestra = dataset.columns[:5].tolist()
    
    print(dataset[cols_muestra].head(3).to_string())
    
    return dataset, todas_estaciones


def preparar_para_ml_estacion_especifica(dataset: pd.DataFrame, 
                                       estacion_objetivo: int, 
                                       todas_estaciones: List[int], 
                                       incluir_tiempo: bool = False) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepara dataset OPTIMIZADO para una estación específica.
    "Apaga" las columnas de llegadas de otras estaciones.
    """
    
    print(f"🔧 Preparando dataset para estación {estacion_objetivo}...")
    
    # Columnas base (features globales)
    cols_base = ['hora', 'dia_semana', 'mes', 'año', 'es_feriado']
    cols_globales = [
        'total_bicis_salieron_global', 'pct_mujeres_salieron_global', 'pct_hombres_salieron_global',
        'q1_edad_salieron_global', 'media_edad_salieron_global', 'q3_edad_salieron_global',
        'pct_iconic_salieron_global', 'pct_fit_salieron_global'
    ]
    
    # Salidas por estación (todas)
    cols_salidas = [f'bicis_salieron_estacion_{est}' for est in todas_estaciones]
    
    # Llegadas SOLO de la estación objetivo
    lags = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]
    cols_llegadas_objetivo = [f'llegadas_estacion_{estacion_objetivo}_h{lag}' for lag in lags]
    
    # Construir lista de features
    feature_cols = cols_base + cols_globales + cols_salidas + cols_llegadas_objetivo
    if incluir_tiempo:
        feature_cols = ['fecha_hora'] + feature_cols
    
    # Target específico
    target_col = f'target_estacion_{estacion_objetivo}'
    
    # Extraer datos
    X = dataset[feature_cols].copy()
    y = dataset[target_col].copy()
    
    # Verificar tipos (solo columnas numéricas, excluyendo tiempo)
    if incluir_tiempo:
        numeric_cols = [col for col in feature_cols if col != 'fecha_hora']
        non_numeric = X[numeric_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    else:
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if non_numeric:
        print(f"⚠️ Convirtiendo columnas no numéricas: {non_numeric}")
        for col in non_numeric:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    print(f"✅ Dataset listo para estación {estacion_objetivo}:")
    print(f"   📊 Shape: {X.shape}")
    print(f"   🎯 Target: {y.name} (media: {y.mean():.2f})")
    print(f"   📈 Features por tipo:")
    print(f"      - Temporales: {len(cols_base)}")
    print(f"      - Globales: {len(cols_globales)}")
    print(f"      - Salidas estaciones: {len(cols_salidas)}")
    print(f"      - Llegadas objetivo: {len(cols_llegadas_objetivo)}")
    
    return X, y, feature_cols


def generar_datasets_todas_estaciones(dataset: pd.DataFrame, 
                                     todas_estaciones: List[int], 
                                     incluir_tiempo: bool = False) -> Dict[int, Dict]:
    """
    Genera datasets ML para todas las estaciones de manera OPTIMIZADA.
    """
    
    print(f"🏭 Generando datasets para {len(todas_estaciones)} estaciones...")
    
    # Pre-computar columnas base (una sola vez)
    cols_base = ['hora', 'dia_semana', 'mes', 'año', 'es_feriado']
    cols_globales = [
        'total_bicis_salieron_global', 'pct_mujeres_salieron_global', 'pct_hombres_salieron_global',
        'q1_edad_salieron_global', 'media_edad_salieron_global', 'q3_edad_salieron_global',
        'pct_iconic_salieron_global', 'pct_fit_salieron_global'
    ]
    cols_salidas = [f'bicis_salieron_estacion_{est}' for est in todas_estaciones]
    lags = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]
    
    # Extraer datos base una vez
    base_features = cols_base + cols_globales + cols_salidas
    if incluir_tiempo:
        base_features = ['fecha_hora'] + base_features
    
    X_base = dataset[base_features].copy()
    
    datasets_ml = {}
    
    for i, estacion in enumerate(todas_estaciones):
        if i % 10 == 0:
            print(f"   Procesando estación {i+1}/{len(todas_estaciones)}: {estacion}")
        
        try:
            # Llegadas específicas de esta estación
            cols_llegadas_est = [f'llegadas_estacion_{estacion}_h{lag}' for lag in lags]
            
            # Verificar que las columnas existan
            cols_disponibles = [col for col in cols_llegadas_est if col in dataset.columns]
            if len(cols_disponibles) == 0:
                print(f"⚠️ Sin datos de llegadas para estación {estacion}")
                continue
            
            # Combinar features
            X_est = pd.concat([X_base, dataset[cols_disponibles]], axis=1)
            y_est = dataset[f'target_estacion_{estacion}']
            
            # Lista de features
            features_est = X_est.columns.tolist()
            
            datasets_ml[estacion] = {
                'X': X_est,
                'y': y_est,
                'features': features_est,
                'estacion': estacion,
                'n_samples': len(X_est),
                'target_mean': y_est.mean(),
                'target_std': y_est.std()
            }
            
        except Exception as e:
            print(f"❌ Error procesando estación {estacion}: {e}")
            continue
    
    print(f"✅ Generados {len(datasets_ml)} datasets exitosamente")
    
    return datasets_ml


# ============= FUNCIONES DE UTILIDAD =============

def mostrar_resumen_dataset(dataset: pd.DataFrame, todas_estaciones: List[int]) -> None:
    """Muestra resumen estadístico del dataset."""
    
    print("\n" + "="*60)
    print("📊 RESUMEN DEL DATASET UNIFICADO")
    print("="*60)
    
    print(f"🏪 Estaciones: {len(todas_estaciones)}")
    print(f"⏰ Período: {dataset['fecha_hora'].min()} a {dataset['fecha_hora'].max()}")
    print(f"📅 Total horas: {len(dataset):,}")
    print(f"🔢 Total columnas: {len(dataset.columns):,}")
    
    # Estadísticas por tipo de feature
    n_temporales = 5  # hora, dia_semana, mes, año, es_feriado
    n_globales = 8    # features globales de salidas
    n_salidas = len(todas_estaciones)
    n_llegadas = len(todas_estaciones) * 12  # 12 lags por estación
    n_targets = len(todas_estaciones)
    
    print(f"\n📈 Features por tipo:")
    print(f"   ⏰ Temporales: {n_temporales}")
    print(f"   🌍 Globales: {n_globales}")
    print(f"   📤 Salidas por estación: {n_salidas}")
    print(f"   📥 Llegadas con lags: {n_llegadas}")
    print(f"   🎯 Targets: {n_targets}")
    
    # Estadísticas de targets
    target_cols = [col for col in dataset.columns if col.startswith('target_estacion_')]
    if target_cols:
        target_stats = dataset[target_cols].describe()
        print(f"\n🎯 Estadísticas de targets:")
        print(f"   📊 Promedio global: {target_stats.loc['mean'].mean():.2f} bicis/hora")
        print(f"   📈 Máximo global: {target_stats.loc['max'].max():.0f} bicis/hora")
        print(f"   📉 Mínimo global: {target_stats.loc['min'].min():.0f} bicis/hora")
    
    print("="*60)

def obtener_nombres_features(df: pd.DataFrame) -> List[str]:
    """
    Función simple que devuelve los nombres de las features como una lista.
    
    Args:
        df: DataFrame del cual extraer los nombres de columnas
    
    Returns:
        List[str]: Lista con los nombres de las columnas
    """
    return df.columns.tolist()


def obtener_features_disponibles(df: pd.DataFrame, mostrar=True) -> Dict[str, List[str]]:
    """
    Función para obtener las features disponibles categorizadas.
    Útil para saber qué features están disponibles antes de crear el dataset.
    
    Args:
        df: DataFrame original con datos de recorridos
        mostrar: Si mostrar el resumen por consola
    
    Returns:
        Dict con features categorizadas
    """
    
    # Obtener todas las columnas
    todas_columnas = df.columns.tolist()
    
    # Categorizar features típicas
    features_temporales = []
    features_estacion = []
    features_usuario = []
    features_bicicleta = []
    features_otras = []
    
    for col in todas_columnas:
        col_lower = col.lower()
        if any(word in col_lower for word in ['fecha', 'hora', 'tiempo', 'date', 'time']):
            features_temporales.append(col)
        elif any(word in col_lower for word in ['estacion', 'station']):
            features_estacion.append(col)
        elif any(word in col_lower for word in ['usuario', 'user', 'género', 'edad', 'genero']):
            features_usuario.append(col)
        elif any(word in col_lower for word in ['bicicleta', 'bike', 'modelo']):
            features_bicicleta.append(col)
        else:
            features_otras.append(col)
    
    resultado = {
        'temporales': features_temporales,
        'estacion': features_estacion,
        'usuario': features_usuario,
        'bicicleta': features_bicicleta,
        'otras': features_otras,
        'todas': todas_columnas
    }
    
    if mostrar:
        print("📋 FEATURES DISPONIBLES EN EL DATASET:")
        print("="*50)
        print(f"⏰ Temporales ({len(features_temporales)}): {features_temporales}")
        print(f"🏪 Estación ({len(features_estacion)}): {features_estacion}")
        print(f"👤 Usuario ({len(features_usuario)}): {features_usuario}")
        print(f"🚲 Bicicleta ({len(features_bicicleta)}): {features_bicicleta}")
        print(f"🔧 Otras ({len(features_otras)}): {features_otras}")
        print(f"📊 Total: {len(todas_columnas)} features")
        print("="*50)
    
    return resultado

