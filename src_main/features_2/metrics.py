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
                           mostrar_plot=True, analisis_completo=False, figsize=(10, 6)):
    """
    Evalúa un modelo de regresión y muestra métricas en tabla.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        nombre_modelo: Nombre del modelo
        mostrar_tabla: Si mostrar la tabla de métricas
        mostrar_estadisticas: Si incluir estadísticas descriptivas
        mostrar_plot: Si mostrar gráfico de predicciones vs reales
        analisis_completo: Si mostrar análisis completo con residuos
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
    
    # Mostrar análisis completo o plot básico
    if analisis_completo:
        grafico_analisis_completo(y_true, y_pred, nombre_modelo, figsize=(15, 10))
    elif mostrar_plot:
        _plot_predicciones_vs_reales(y_true, y_pred, nombre_modelo, figsize)
    
    return metricas, df_tabla


def _plot_predicciones_vs_reales(y_true, y_pred, nombre_modelo, figsize):
    """
    Crea gráfico de predicciones vs valores reales con métricas integradas.
    """
    
    plt.figure(figsize=figsize)
    
    # Convertir a arrays numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Scatter plot con colores según el error
    errors = np.abs(y_true - y_pred)
    scatter = plt.scatter(y_true, y_pred, c=errors, cmap='viridis', alpha=0.6, s=30, 
                         edgecolors='black', linewidth=0.5, label='Datos')
    
    # Colorbar para los errores
    cbar = plt.colorbar(scatter)
    cbar.set_label('Error Absoluto', rotation=270, labelpad=15)
    
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
    
    # Calcular métricas principales
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Agregar texto con métricas en el gráfico
    textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nN = {len(y_true):,}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Formato
    plt.xlabel('Valores Reales', fontsize=12)
    plt.ylabel('Predicciones', fontsize=12)
    plt.title(f'{nombre_modelo} - Predicciones vs Valores Reales', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Igualar aspectos para que la línea diagonal se vea como 45°
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def grafico_analisis_completo(y_true, y_pred, nombre_modelo="Modelo", figsize=(15, 10)):
    """
    Crea un análisis gráfico completo del modelo con múltiples visualizaciones.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos  
        nombre_modelo: Nombre del modelo
        figsize: Tamaño de la figura
    """
    
    # Convertir a arrays numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuos = y_true - y_pred
    
    # Crear subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Análisis Completo del Modelo: {nombre_modelo}', fontsize=16, fontweight='bold')
    
    # 1. Predicciones vs Valores Reales
    ax1 = axes[0, 0]
    errors = np.abs(residuos)
    scatter = ax1.scatter(y_true, y_pred, c=errors, cmap='viridis', alpha=0.6, s=30)
    
    # Línea diagonal perfecta
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
    
    # Línea de tendencia
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax1.plot(y_true, p(y_true), 'g-', linewidth=2, alpha=0.8)
    
    # Métricas
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}'
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel('Valores Reales')
    ax1.set_ylabel('Predicciones')
    ax1.set_title('Predicciones vs Valores Reales')
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuos vs Predicciones
    ax2 = axes[0, 1]
    ax2.scatter(y_pred, residuos, alpha=0.6, s=30)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicciones')
    ax2.set_ylabel('Residuos (Real - Predicho)')
    ax2.set_title('Residuos vs Predicciones')
    ax2.grid(True, alpha=0.3)
    
    # 3. Histograma de Residuos
    ax3 = axes[1, 0]
    ax3.hist(residuos, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.axvline(x=np.mean(residuos), color='g', linestyle='--', linewidth=2, 
                label=f'Media = {np.mean(residuos):.4f}')
    ax3.set_xlabel('Residuos')
    ax3.set_ylabel('Frecuencia')
    ax3.set_title('Distribución de Residuos')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-Q Plot (aproximado)
    ax4 = axes[1, 1]
    from scipy import stats
    try:
        stats.probplot(residuos, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Normalidad de Residuos)')
        ax4.grid(True, alpha=0.3)
    except ImportError:
        # Si scipy no está disponible, hacer un gráfico alternativo
        sorted_residuos = np.sort(residuos)
        n = len(sorted_residuos)
        theoretical_quantiles = stats.norm.ppf((np.arange(1, n+1) - 0.5) / n)
        ax4.scatter(theoretical_quantiles, sorted_residuos, alpha=0.6)
        ax4.plot(theoretical_quantiles, theoretical_quantiles, 'r--', linewidth=2)
        ax4.set_xlabel('Cuantiles Teóricos')
        ax4.set_ylabel('Cuantiles de Residuos')
        ax4.set_title('Q-Q Plot (Normalidad de Residuos)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()