import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays

def crear_dataset_prediccion_bicis(df, id_estacion):
    """
    Crea un dataset para predecir cuántas bicis llegarán a una estación por hora.
    
    Args:
        df: DataFrame con los datos de recorridos
        id_estacion: ID de la estación para la cual generar el dataset
    
    Returns:
        DataFrame con features para predicción
    """
    
    # Limpiar y preparar datos
    df = df.copy()
    
    # Convertir fechas a datetime
    df['fecha_origen_recorrido'] = pd.to_datetime(df['fecha_origen_recorrido'])
    df['fecha_destino_recorrido'] = pd.to_datetime(df['fecha_destino_recorrido'])
    
    # Crear columnas auxiliares
    df['hora_origen'] = df['fecha_origen_recorrido'].dt.floor('H')
    df['hora_destino'] = df['fecha_destino_recorrido'].dt.floor('H')
    
    # Filtrar datos relevantes para la estación
    llegadas = df[df['id_estacion_destino'] == id_estacion].copy()
    salidas = df[df['id_estacion_origen'] == id_estacion].copy()
    
    # Obtener rango de fechas
    fecha_min = df['fecha_origen_recorrido'].min().floor('H')
    fecha_max = df['fecha_destino_recorrido'].max().floor('H')
    
    # Crear rango completo de horas
    horas_completas = pd.date_range(start=fecha_min, end=fecha_max, freq='H')
    
    # Crear DataFrame base con todas las horas
    dataset = pd.DataFrame({'fecha_hora': horas_completas})
    dataset['id_estacion'] = id_estacion
    
    # Agregar columnas de tiempo separadas
    dataset['fecha'] = dataset['fecha_hora'].dt.date
    dataset['hora'] = dataset['fecha_hora'].dt.hour
    
    # Agregar features temporales
    dataset['dia_semana'] = dataset['fecha_hora'].dt.dayofweek
    dataset['mes'] = dataset['fecha_hora'].dt.month
    dataset['año'] = dataset['fecha_hora'].dt.year
    
    # Feature de feriados (usando holidays de Argentina)
    ar_holidays = holidays.Argentina()
    dataset['es_feriado'] = dataset['fecha_hora'].dt.date.apply(lambda x: x in ar_holidays).astype(int)
    
    # LLEGADAS: Contar bicis que llegaron por hora
    llegadas_por_hora = llegadas.groupby('hora_destino').size().reset_index()
    llegadas_por_hora.columns = ['fecha_hora', 'bicis_llegaron_h0']
    dataset = dataset.merge(llegadas_por_hora, on='fecha_hora', how='left')
    dataset['bicis_llegaron_h0'] = dataset['bicis_llegaron_h0'].fillna(0)
    
    # Features de llegadas en horas anteriores (h-1, h-2, ..., h-6)
    for i in range(1, 7):
        dataset[f'bicis_llegaron_h{i}'] = dataset['bicis_llegaron_h0'].shift(i).fillna(0)
    
    # SALIDAS: Agregar features de salidas
    # Limpiar datos de salidas antes de agregar
    salidas_clean = salidas.copy()
    
    # Limpiar columna de género
    salidas_clean['género'] = salidas_clean['género'].fillna('OTHER')
    
    # Limpiar columna de edad (convertir a numérico)
    salidas_clean['edad_usuario'] = pd.to_numeric(salidas_clean['edad_usuario'], errors='coerce')
    
    # Limpiar columna de modelo
    salidas_clean['modelo_bicicleta'] = salidas_clean['modelo_bicicleta'].fillna('OTHER')
    
    # Preparar datos de salidas con todas las features necesarias
    salidas_stats = salidas_clean.groupby('hora_origen').agg({
        'Id_recorrido': 'count',  # total bicis que salieron
        'género': lambda x: (x == 'F').sum() / len(x) if len(x) > 0 else 0,  # % mujeres
        'edad_usuario': lambda x: x.mean() if x.notna().sum() > 0 else 0,   # promedio edad (manejando NaN)
        'modelo_bicicleta': lambda x: (x == 'ICONIC').sum() / len(x) if len(x) > 0 else 0  # % ICONIC
    }).reset_index()
    
    salidas_stats.columns = [
        'fecha_hora', 'total_bicis_salieron', 'pct_mujeres_salieron', 
        'promedio_edad_salieron', 'pct_iconic_salieron'
    ]
    
    # Calcular porcentaje de hombres
    salidas_stats['pct_hombres_salieron'] = 1 - salidas_stats['pct_mujeres_salieron']
    
    # Mergear con dataset principal
    dataset = dataset.merge(salidas_stats, on='fecha_hora', how='left')
    
    # Rellenar valores faltantes
    cols_salidas = ['total_bicis_salieron', 'pct_mujeres_salieron', 
                   'pct_hombres_salieron', 'promedio_edad_salieron', 'pct_iconic_salieron']
    
    for col in cols_salidas:
        if col.startswith('pct_'):
            dataset[col] = dataset[col].fillna(0)
        elif col == 'promedio_edad_salieron':
            # Calcular media global de edades válidas para rellenar
            edad_media_global = salidas_clean['edad_usuario'].mean()
            if pd.isna(edad_media_global):
                edad_media_global = 30  # valor por defecto si no hay edades válidas
            dataset[col] = dataset[col].fillna(edad_media_global)
        else:
            dataset[col] = dataset[col].fillna(0)
    
    # Ordenar por fecha_hora
    dataset = dataset.sort_values('fecha_hora').reset_index(drop=True)
    
    # Crear variable objetivo (siguiente hora)
    dataset['target'] = dataset['bicis_llegaron_h0'].shift(-1)
    
    # Eliminar la última fila (no tiene target)
    dataset = dataset[:-1].copy()
    
    # Reordenar columnas para ML (fecha_hora y fecha primero, luego features numéricas, target al final)
    feature_cols = [
        'fecha_hora', 'fecha', 'hora', 'id_estacion', 'dia_semana', 'mes', 'año', 'es_feriado',
        'bicis_llegaron_h1', 'bicis_llegaron_h2', 'bicis_llegaron_h3',
        'bicis_llegaron_h4', 'bicis_llegaron_h5', 'bicis_llegaron_h6',
        'total_bicis_salieron', 'pct_mujeres_salieron', 'pct_hombres_salieron',
        'promedio_edad_salieron', 'pct_iconic_salieron', 'target'
    ]
    
    return dataset[feature_cols]


def preparar_para_ml(dataset, incluir_tiempo=False):
    """
    Prepara el dataset para ser usado directamente en modelos de ML.
    
    Args:
        dataset: DataFrame generado por crear_dataset_prediccion_bicis
        incluir_tiempo: Si True, mantiene las columnas de tiempo para referencia
    
    Returns:
        X: Features para el modelo (todas numéricas)
        y: Variable objetivo
        feature_names: Nombres de las features
        dataset_ml: Dataset completo preparado para ML
    """
    
    dataset_ml = dataset.copy()
    
    # Separar features y target
    if incluir_tiempo:
        feature_cols = [col for col in dataset_ml.columns if col not in ['target']]
    else:
        feature_cols = [col for col in dataset_ml.columns if col not in ['fecha_hora', 'fecha', 'target']]
    
    X = dataset_ml[feature_cols]
    y = dataset_ml['target']
    
    # Verificar que todas las features sean numéricas (excluyendo columnas de tiempo)
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if incluir_tiempo:
        # Filtrar columnas de tiempo que son esperadas
        non_numeric = [col for col in non_numeric if col not in ['fecha_hora', 'fecha']]
    
    if non_numeric:
        print(f"Advertencia: Columnas no numéricas detectadas: {non_numeric}")
        # Convertir columnas no numéricas si es necesario
        for col in non_numeric:
            if col not in ['fecha_hora', 'fecha']:  # Mantener columnas de tiempo
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    return X, y, feature_cols, dataset_ml


def generar_dataset_multiple_estaciones(df, lista_estaciones=None):
    """
    Genera dataset para múltiples estaciones y las concatena.
    
    Args:
        df: DataFrame con los datos de recorridos
        lista_estaciones: Lista de IDs de estaciones. Si None, usa todas las estaciones.
    
    Returns:
        DataFrame concatenado con datos de todas las estaciones
    """
    
    if lista_estaciones is None:
        # Obtener todas las estaciones únicas
        estaciones_origen = set(df['id_estacion_origen'].unique())
        estaciones_destino = set(df['id_estacion_destino'].unique())
        lista_estaciones = list(estaciones_origen.union(estaciones_destino))
    
    datasets = []
    
    for estacion in lista_estaciones:
        print(f"Procesando estación {estacion}...")
        try:
            dataset_estacion = crear_dataset_prediccion_bicis(df, estacion)
            datasets.append(dataset_estacion)
        except Exception as e:
            print(f"Error procesando estación {estacion}: {e}")
            continue
    
    # Concatenar todos los datasets
    dataset_final = pd.concat(datasets, ignore_index=True)
    
    return dataset_final
