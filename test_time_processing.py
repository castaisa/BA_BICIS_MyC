"""
Script de prueba para las funciones de procesamiento de tiempo
Verifica que las funciones de data_processing.py funcionen correctamente
"""

import pandas as pd
import numpy as np
import sys
import os

# Agregar el directorio src_main/features_2 al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src_main', 'features_2'))

try:
    from data_processing import (
        limpiar_tiempo_dataset,
        preparar_dataset_para_ml,
        procesar_dataset_completo,
        quick_clean_time,
        quick_ml_prep,
        procesar_columnas_tiempo
    )
    print("✅ Importaciones exitosas")
except ImportError as e:
    print(f"❌ Error en importaciones: {e}")
    sys.exit(1)


def crear_dataset_prueba():
    """Crear un dataset de prueba con diferentes formatos de tiempo"""
    
    np.random.seed(42)
    n_samples = 100
    
    # Crear dataset con diferentes tipos de columnas de tiempo
    data = {
        'fecha_hora': pd.date_range('2024-01-01', periods=n_samples, freq='h'),
        'hora_string': ['08:30:00', '14:15:30', '22:45:15'] * (n_samples // 3 + 1),
        'hour_numeric': np.random.randint(0, 24, n_samples),
        'time_column': ['12:00', '18:30', '06:15'] * (n_samples // 3 + 1),
        'bicis_salieron_estacion_1': np.random.randint(0, 10, n_samples),
        'bicis_salieron_estacion_2': np.random.randint(0, 15, n_samples),
        'llegadas_estacion_1_h1': np.random.randint(0, 8, n_samples),
        'llegadas_estacion_1_h2': np.random.randint(0, 12, n_samples),
        'target_estacion_1': np.random.randint(0, 20, n_samples),
        'target_estacion_2': np.random.randint(0, 25, n_samples),
        'feature_normal': np.random.normal(0, 1, n_samples)
    }
    
    # Ajustar tamaños de listas
    for key in ['hora_string', 'time_column']:
        data[key] = data[key][:n_samples]
    
    df = pd.DataFrame(data)
    return df


def test_limpiar_tiempo():
    """Probar la función limpiar_tiempo_dataset"""
    
    print("\n🧪 PRUEBA: limpiar_tiempo_dataset")
    print("-" * 40)
    
    df = crear_dataset_prueba()
    print(f"Dataset original: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    print(f"Tipos originales:")
    for col in ['fecha_hora', 'hora_string', 'hour_numeric', 'time_column']:
        print(f"  {col}: {df[col].dtype} (ejemplo: {df[col].iloc[0]})")
    
    # Aplicar limpieza
    df_limpio, cambios = limpiar_tiempo_dataset(df, verbose=True)
    
    print(f"\nDataset limpio: {df_limpio.shape}")
    print("Cambios realizados:")
    for cambio in cambios:
        print(f"  {cambio}")
    
    # Verificar resultados
    print(f"\nVerificación:")
    print(f"  ¿fecha_hora eliminada?: {'fecha_hora' not in df_limpio.columns}")
    
    for col in ['hora_string', 'hour_numeric', 'time_column']:
        if col in df_limpio.columns:
            min_val = df_limpio[col].min()
            max_val = df_limpio[col].max()
            print(f"  {col}: rango {min_val}-{max_val}, tipo {df_limpio[col].dtype}")


def test_preparar_ml():
    """Probar la función preparar_dataset_para_ml"""
    
    print("\n🧪 PRUEBA: preparar_dataset_para_ml")
    print("-" * 40)
    
    df = crear_dataset_prueba()
    df_prep, reporte = preparar_dataset_para_ml(df, verbose=True)
    
    print(f"\nReporte del procesamiento:")
    print(f"  Shape original: {reporte['shape_original']}")
    print(f"  Shape final: {reporte['shape_final']}")
    print(f"  Columnas eliminadas: {len(reporte['columnas_eliminadas'])}")
    print(f"  Columnas convertidas: {len(reporte['columnas_convertidas'])}")
    print(f"  Problemas detectados: {len(reporte['problemas_detectados'])}")


def test_procesamiento_completo():
    """Probar la función procesar_dataset_completo"""
    
    print("\n🧪 PRUEBA: procesar_dataset_completo")
    print("-" * 40)
    
    df = crear_dataset_prueba()
    
    # Prueba 1: Solo limpieza
    print("Prueba 1: Solo limpieza")
    resultado1 = procesar_dataset_completo(df, solo_limpieza=True, verbose=False)
    print(f"  Resultado: {resultado1['modo']}, shape: {resultado1['df_limpio'].shape}")
    
    # Prueba 2: Estación específica
    print("\nPrueba 2: Estación específica")
    resultado2 = procesar_dataset_completo(df, estacion_id=1, verbose=False)
    if 'error' not in resultado2:
        print(f"  Resultado: {resultado2['modo']}")
        print(f"  X: {resultado2['X'].shape}, y: {resultado2['y'].shape}")
        print(f"  Features: {len(resultado2['features'])}")
    else:
        print(f"  Error: {resultado2['error']}")


def test_funciones_rapidas():
    """Probar las funciones rápidas"""
    
    print("\n🧪 PRUEBA: Funciones rápidas")
    print("-" * 40)
    
    df = crear_dataset_prueba()
    
    # quick_clean_time
    df_clean = quick_clean_time(df, verbose=False)
    print(f"quick_clean_time: {df.shape} → {df_clean.shape}")
    print(f"  ¿fecha_hora eliminada?: {'fecha_hora' not in df_clean.columns}")
    
    # quick_ml_prep
    df_prep = quick_ml_prep(df, verbose=False)
    print(f"quick_ml_prep: {df.shape} → {df_prep.shape}")


if __name__ == "__main__":
    print("🚀 INICIANDO PRUEBAS DE PROCESAMIENTO DE TIEMPO")
    print("=" * 50)
    
    try:
        test_limpiar_tiempo()
        test_preparar_ml()
        test_procesamiento_completo()
        test_funciones_rapidas()
        
        print("\n✅ TODAS LAS PRUEBAS COMPLETADAS")
        
    except Exception as e:
        print(f"\n❌ ERROR EN PRUEBAS: {e}")
        import traceback
        traceback.print_exc()
