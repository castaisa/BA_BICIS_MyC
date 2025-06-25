import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score
)
from typing import Dict, Optional, Tuple, Union
import warnings


def calcular_metricas_regresion(y_true, y_pred):
    """
    Calcula todas las métricas de regresión para un modelo.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
    
    Returns:
        dict: Diccionario con todas las métricas
    """
    
    # Convertir a numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Métricas básicas
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Métricas adicionales
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        # Calcular MAPE manualmente si sklearn no lo tiene
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.inf
    
    explained_var = explained_variance_score(y_true, y_pred)
    max_error = np.max(np.abs(y_true - y_pred))
    
    # Métricas estadísticas
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    
    # Correlación
    correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan
    
    # Sesgo (bias)
    bias = np.mean(y_pred - y_true)
    
    # Error relativo medio
    relative_error = np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape,
        'Explained_Variance': explained_var,
        'Max_Error': max_error,
        'Correlation': correlation,
        'Bias': bias,
        'Relative_Error_%': relative_error,
        'Mean_True': mean_true,
        'Mean_Pred': mean_pred,
        'Std_True': std_true,
        'Std_Pred': std_pred,
        'N_Samples': len(y_true)
    }


def crear_tabla_metricas(metricas_dict, nombre_modelo="Modelo", mostrar_estadisticas=True):
    """
    Crea una tabla formateada con las métricas de un modelo.
    
    Args:
        metricas_dict: Diccionario con métricas del modelo
        nombre_modelo: Nombre del modelo para mostrar
        mostrar_estadisticas: Si mostrar estadísticas descriptivas
    
    Returns:
        pd.DataFrame: Tabla con métricas formateadas
    """
    
    # Definir orden y nombres de métricas principales
    metricas_principales = [
        ('MAE', 'Mean Absolute Error'),
        ('MSE', 'Mean Squared Error'),
        ('RMSE', 'Root Mean Squared Error'),
        ('R²', 'R-squared'),
        ('MAPE', 'Mean Absolute Percentage Error (%)'),
        ('Explained_Variance', 'Explained Variance'),
        ('Max_Error', 'Maximum Error'),
        ('Correlation', 'Correlation'),
        ('Bias', 'Bias'),
        ('Relative_Error_%', 'Relative Error (%)')
    ]
    
    # Crear DataFrame principal
    data = []
    for key, desc in metricas_principales:
        if key in metricas_dict:
            value = metricas_dict[key]
            # Formatear valores
            if key in ['R²', 'Explained_Variance', 'Correlation']:
                formatted_value = f"{value:.4f}"
            elif key in ['MAPE', 'Relative_Error_%']:
                formatted_value = f"{value:.2f}%"
            elif key in ['MAE', 'MSE', 'RMSE', 'Max_Error', 'Bias']:
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = f"{value:.4f}"
            
            data.append({
                'Métrica': desc,
                nombre_modelo: formatted_value
            })
    
    df_metricas = pd.DataFrame(data)
    
    # Agregar estadísticas descriptivas si se solicita
    if mostrar_estadisticas:
        estadisticas = [
            ('Mean_True', 'Media Valores Reales'),
            ('Mean_Pred', 'Media Predicciones'),
            ('Std_True', 'Desv. Est. Valores Reales'),
            ('Std_Pred', 'Desv. Est. Predicciones'),
            ('N_Samples', 'Número de Muestras')
        ]
        
        data_stats = []
        for key, desc in estadisticas:
            if key in metricas_dict:
                value = metricas_dict[key]
                if key == 'N_Samples':
                    formatted_value = f"{int(value):,}"
                else:
                    formatted_value = f"{value:.4f}"
                
                data_stats.append({
                    'Métrica': desc,
                    nombre_modelo: formatted_value
                })
        
        df_stats = pd.DataFrame(data_stats)
        
        # Separador
        separador = pd.DataFrame({
            'Métrica': ['--- Estadísticas Descriptivas ---'],
            nombre_modelo: ['']
        })
        
        df_metricas = pd.concat([df_metricas, separador, df_stats], ignore_index=True)
    
    return df_metricas


def evaluar_modelo_regresion(y_true, y_pred, nombre_modelo="Modelo", 
                           mostrar_tabla=True, mostrar_estadisticas=True,
                           mostrar_plot=False, figsize=(10, 6)):
    """
    Evalúa un modelo de regresión y muestra métricas en tabla.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        nombre_modelo: Nombre del modelo
        mostrar_tabla: Si mostrar la tabla de métricas
        mostrar_estadisticas: Si incluir estadísticas descriptivas
        mostrar_plot: Si mostrar gráfico de predicciones vs reales
        figsize: Tamaño de la figura
    
    Returns:
        tuple: (metricas_dict, df_tabla)
    """
    
    # Calcular métricas
    metricas = calcular_metricas_regresion(y_true, y_pred)
    
    # Crear tabla
    df_tabla = crear_tabla_metricas(metricas, nombre_modelo, mostrar_estadisticas)
    
    # Mostrar tabla si se solicita
    if mostrar_tabla:
        print(f"\n{'='*60}")
        print(f"MÉTRICAS DE REGRESIÓN - {nombre_modelo}")
        print(f"{'='*60}")
        print(df_tabla.to_string(index=False))
        print(f"{'='*60}")
    
    # Mostrar plot si se solicita
    if mostrar_plot:
        _plot_predicciones_vs_reales(y_true, y_pred, nombre_modelo, figsize)
    
    return metricas, df_tabla


def comparar_modelos_regresion(modelos_predicciones: Dict[str, np.ndarray], 
                             y_true: np.ndarray,
                             mostrar_tabla=True,
                             mostrar_plots=False,
                             ordenar_por='R²',
                             ascendente=False,
                             guardar_csv=None):
    """
    Compara múltiples modelos de regresión y muestra métricas en tabla comparativa.
    
    Args:
        modelos_predicciones: Dict con {nombre_modelo: predicciones}
        y_true: Valores reales
        mostrar_tabla: Si mostrar tabla comparativa
        mostrar_plots: Si mostrar gráficos comparativos
        ordenar_por: Métrica por la cual ordenar ('R²', 'RMSE', 'MAE', etc.)
        ascendente: Si ordenar de forma ascendente
        guardar_csv: Ruta para guardar la tabla en CSV
    
    Returns:
        tuple: (dict_metricas_todos, df_comparacion)
    """
    
    print(f"\n{'='*80}")
    print("COMPARACIÓN DE MODELOS DE REGRESIÓN")
    print(f"{'='*80}")
    
    # Calcular métricas para todos los modelos
    metricas_todos = {}
    for nombre, predicciones in modelos_predicciones.items():
        metricas_todos[nombre] = calcular_metricas_regresion(y_true, predicciones)
    
    # Crear tabla comparativa
    df_comparacion = crear_tabla_comparacion(metricas_todos, ordenar_por, ascendente)
    
    # Mostrar tabla si se solicita
    if mostrar_tabla:
        print(df_comparacion.to_string(index=False))
        print(f"\n{'='*80}")
        
        # Mostrar resumen de mejores modelos
        mostrar_mejores_modelos(df_comparacion)
    
    # Guardar CSV si se solicita
    if guardar_csv:
        df_comparacion.to_csv(guardar_csv, index=False)
        print(f"\n💾 Tabla guardada en: {guardar_csv}")
    
    # Mostrar gráficos si se solicita
    if mostrar_plots:
        crear_plots_comparacion(modelos_predicciones, y_true)
    
    return metricas_todos, df_comparacion


def crear_tabla_comparacion(metricas_todos, ordenar_por='R²', ascendente=False):
    """
    Crea tabla comparativa de múltiples modelos.
    """
    
    # Métricas principales para comparación
    metricas_comparacion = ['MAE', 'RMSE', 'R²', 'MAPE', 'Correlation', 'Bias']
    
    # Crear DataFrame
    data = []
    for nombre_modelo, metricas in metricas_todos.items():
        fila = {'Modelo': nombre_modelo}
        for metrica in metricas_comparacion:
            if metrica in metricas:
                value = metricas[metrica]
                if metrica in ['R²', 'Correlation']:
                    fila[metrica] = f"{value:.4f}"
                elif metrica == 'MAPE':
                    fila[metrica] = f"{value:.2f}%"
                else:
                    fila[metrica] = f"{value:.4f}"
            else:
                fila[metrica] = "N/A"
        
        # Agregar número de muestras
        fila['N_Samples'] = f"{int(metricas.get('N_Samples', 0)):,}"
        data.append(fila)
    
    df = pd.DataFrame(data)
    
    # Ordenar por métrica especificada si existe
    if ordenar_por in df.columns and ordenar_por != 'Modelo':
        # Convertir a numérico para ordenar (remover % y convertir)
        col_ordenar = df[ordenar_por].str.replace('%', '').astype(float)
        df_sorted = df.iloc[col_ordenar.argsort()]
        if not ascendente:
            df_sorted = df_sorted.iloc[::-1]
        return df_sorted.reset_index(drop=True)
    
    return df


def mostrar_mejores_modelos(df_comparacion):
    """
    Muestra un resumen de los mejores modelos por métrica.
    """
    
    print("🏆 MEJORES MODELOS POR MÉTRICA:")
    print("-" * 50)
    
    # Métricas donde menor es mejor
    metricas_menor_mejor = ['MAE', 'RMSE', 'MAPE', 'Bias']
    # Métricas donde mayor es mejor  
    metricas_mayor_mejor = ['R²', 'Correlation']
    
    for metrica in metricas_menor_mejor + metricas_mayor_mejor:
        if metrica in df_comparacion.columns:
            # Convertir a numérico
            valores = df_comparacion[metrica].str.replace('%', '').astype(float)
            
            if metrica in metricas_menor_mejor:
                idx_mejor = valores.idxmin()
            else:
                idx_mejor = valores.idxmax()
            
            mejor_modelo = df_comparacion.loc[idx_mejor, 'Modelo']
            mejor_valor = df_comparacion.loc[idx_mejor, metrica]
            
            print(f"{metrica:>12}: {mejor_modelo:<20} ({mejor_valor})")


def _plot_predicciones_vs_reales(y_true, y_pred, nombre_modelo, figsize):
    """
    Crea gráfico de predicciones vs valores reales.
    """
    
    plt.figure(figsize=figsize)
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    # Línea diagonal perfecta
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
             label='Predicción Perfecta', alpha=0.8)
    
    # Línea de tendencia
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), 'g-', linewidth=2, alpha=0.8, 
             label=f'Tendencia (pendiente={z[0]:.3f})')
    
    # Calcular R²
    r2 = r2_score(y_true, y_pred)
    
    # Formato
    plt.xlabel('Valores Reales', fontsize=12)
    plt.ylabel('Predicciones', fontsize=12)
    plt.title(f'{nombre_modelo} - Predicciones vs Valores Reales (R² = {r2:.4f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def crear_plots_comparacion(modelos_predicciones, y_true, figsize=(15, 10)):
    """
    Crea gráficos comparativos para múltiples modelos.
    """
    
    n_modelos = len(modelos_predicciones)
    cols = min(3, n_modelos)
    rows = (n_modelos + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_modelos == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, (nombre, y_pred) in enumerate(modelos_predicciones.items()):
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=20)
        
        # Línea diagonal
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
        
        # R² y RMSE
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        ax.set_title(f'{nombre}\nR² = {r2:.4f}, RMSE = {rmse:.4f}')
        ax.set_xlabel('Valores Reales')
        ax.set_ylabel('Predicciones')
        ax.grid(True, alpha=0.3)
    
    # Ocultar subplots vacíos
    for idx in range(n_modelos, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


# Función de conveniencia para evaluación rápida
def evaluar_rapido(y_true, y_pred, nombre="Modelo"):
    """
    Evaluación rápida con tabla básica.
    """
    return evaluar_modelo_regresion(y_true, y_pred, nombre, 
                                  mostrar_estadisticas=False, 
                                  mostrar_plot=False)


# Función de conveniencia para comparación rápida
def comparar_rapido(modelos_dict, y_true, ordenar_por='R²'):
    """
    Comparación rápida sin gráficos.
    """
    return comparar_modelos_regresion(modelos_dict, y_true,
                                    mostrar_plots=False,
                                    ordenar_por=ordenar_por)
