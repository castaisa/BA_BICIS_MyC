
import pandas as pd
import numpy as np
import holidays
from typing import Tuple, List

def crear_dataset_f3(df: pd.DataFrame, estaciones_globales: List[int] = None) -> Tuple[pd.DataFrame, List[int]]:
    """
    Crea un dataset unificado OPTIMIZADO para predecir llegadas de bicis por hora.
    
    Args:
        df: DataFrame con los datos de recorridos
        estaciones_globales: Lista de estaciones a usar (si None, usa las del df)
    
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
    
    # Obtener estaciones únicas
    if estaciones_globales is None:
        todas_estaciones = sorted(list(set(df['id_estacion_origen'].unique()) | 
                                      set(df['id_estacion_destino'].unique())))
        print(f"🏪 Encontradas {len(todas_estaciones)} estaciones únicas en este dataset")
    else:
        todas_estaciones = sorted(estaciones_globales)
        print(f"🏪 Usando {len(todas_estaciones)} estaciones predefinidas")
    
    n_estaciones = len(todas_estaciones)
        
    
    print(f"🏪 Encontradas {n_estaciones} estaciones únicas")
    
    # Crear rango temporal
    fecha_min = df['fecha_origen_recorrido'].min().floor('H')
    fecha_max = df['fecha_destino_recorrido'].max().floor('H')
    horas_completas = pd.date_range(start=fecha_min, end=fecha_max, freq='H')
    n_horas = len(horas_completas)
    
    print(f"⏰ Rango temporal: {fecha_min} a {fecha_max} ({n_horas:,} horas)")

    print(f"📊 Dataset final tendrá ~{n_horas:,} filas × {6 + 8 + n_estaciones + n_estaciones:,} columnas")

    
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
    
    # Crear tabla pivot para CANTIDAD de salidas por estación
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
    
    # Merge cantidad de salidas
    dataset = dataset.merge(salidas_pivot, on='fecha_hora', how='left')
    
    # Fill NAs para salidas por estación
    cols_salidas = [f'bicis_salieron_estacion_{est}' for est in todas_estaciones]
    dataset[cols_salidas] = dataset[cols_salidas].fillna(0)
    
    
    # ============= CREAR TARGETS =============
    print("🎯 Creando variables target...")
    
    # Crear llegadas por hora SOLO para calcular targets (no se agregan al dataset)
    llegadas_por_hora = (df.groupby(['hora_destino', 'id_estacion_destino'])
                          .size()
                          .unstack(fill_value=0)
                          .reset_index())
    llegadas_por_hora.columns.name = None
    
    # Renombrar columna de hora
    llegadas_por_hora = llegadas_por_hora.rename(columns={'hora_destino': 'fecha_hora'})
    
    # Agregar columnas faltantes para estaciones sin llegadas
    for est in todas_estaciones:
        if est not in llegadas_por_hora.columns:
            llegadas_por_hora[est] = 0
    
    # Crear targets haciendo merge temporal SOLO para calcular targets
    dataset_temp = dataset.merge(llegadas_por_hora, on='fecha_hora', how='left')
    
    # Crear targets (llegadas de la próxima hora)
    for est in todas_estaciones:
        if est in dataset_temp.columns:
            dataset[f'target_estacion_{est}'] = dataset_temp[est].shift(-1)
        else:
            dataset[f'target_estacion_{est}'] = 0
    
    # Eliminar NAs en targets
    target_cols = [f'target_estacion_{est}' for est in todas_estaciones]
    dataset[target_cols] = dataset[target_cols].fillna(0)
    
    # ============= LIMPIEZA FINAL =============
    print("🧹 Limpieza final...")
    
    # Eliminar última fila (sin targets)
    dataset = dataset[:-1].copy()
    
    # Ordenar por fecha
    dataset = dataset.sort_values('fecha_hora').reset_index(drop=True)
    
    # Actualizar cálculo de columnas
    n_cols_temporales = 6  # fecha_hora, hora, dia_semana, mes, año, es_feriado
    n_cols_globales = 8    # estadísticas globales
    n_cols_cantidad_por_estacion = n_estaciones  # bicis_salieron_estacion_{est}
    n_cols_stats_por_estacion = n_estaciones * 7  # 7 estadísticas por estación
    n_cols_targets = n_estaciones
    
    total_cols = n_cols_temporales + n_cols_globales + n_cols_cantidad_por_estacion + n_cols_stats_por_estacion + n_cols_targets
    
    print(f"✅ Dataset creado: {len(dataset):,} filas × {len(dataset.columns):,} columnas")
    print(f"📊 Desglose de columnas:")
    print(f"   • Temporales: {n_cols_temporales}")
    print(f"   • Globales: {n_cols_globales}")
    print(f"   • Cantidad por estación: {n_cols_cantidad_por_estacion}")
    print(f"   • Estadísticas por estación: {n_cols_stats_por_estacion}")
    print(f"   • Targets: {n_cols_targets}")
    print(f"   • Total esperado: {total_cols}")
    print(f"📈 Memoria aproximada: {dataset.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Mostrar muestra del dataset
    print("\n📋 Primeras 3 filas del dataset:")
    cols_muestra = ['fecha_hora', 'hora', 'dia_semana', 'total_bicis_salieron_global']
    
    # Agregar columnas que realmente existen
    if f'bicis_salieron_estacion_{todas_estaciones[0]}' in dataset.columns:
        cols_muestra.append(f'bicis_salieron_estacion_{todas_estaciones[0]}')
    if f'pct_mujeres_salieron_estacion_{todas_estaciones[0]}' in dataset.columns:
        cols_muestra.append(f'pct_mujeres_salieron_estacion_{todas_estaciones[0]}')
    if f'target_estacion_{todas_estaciones[0]}' in dataset.columns:
        cols_muestra.append(f'target_estacion_{todas_estaciones[0]}')
    
    print(dataset[cols_muestra].head(3).to_string())
    
    return dataset, todas_estaciones