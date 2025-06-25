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
                           mostrar_plot=True, analisis_completo=False, figsize=(8, 6)):
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
        grafico_analisis_completo(y_true, y_pred, nombre_modelo)
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


def grafico_analisis_completo(y_true, y_pred, nombre_modelo="Modelo", figsize=(8, 6)):
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


def tabla_metricas_modelos(modelos_predicciones, y_true):
    """
    Función simple que recibe un diccionario con modelos y predicciones 
    y genera una tabla prolija de métricas.
    
    Args:
        modelos_predicciones: Dict con {nombre_modelo: predicciones}
        y_true: Valores reales
    
    Returns:
        pd.DataFrame: Tabla prolija con métricas de todos los modelos
    """
    
    # Convertir y_true a numpy array
    y_true = np.array(y_true)
    
    # Lista para almacenar resultados
    resultados = []
    
    # Calcular métricas para cada modelo
    for nombre_modelo, y_pred in modelos_predicciones.items():
        y_pred = np.array(y_pred)
        
        # Calcular métricas principales
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE con manejo de división por cero
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        except:
            mask = y_true != 0
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.inf
        
        # Correlación
        correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan
        
        # Agregar a resultados
        resultados.append({
            'Modelo': nombre_modelo,
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R²': round(r2, 4),
            'MAPE (%)': round(mape, 2),
            'Correlación': round(correlation, 4),
            'N_Muestras': len(y_true)
        })
    
    # Crear DataFrame
    df = pd.DataFrame(resultados)
    
    # Ordenar por R² de mayor a menor
    df = df.sort_values('R²', ascending=False).reset_index(drop=True)
    
    # Agregar ranking
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    # # Mostrar tabla prolija
    # print("\n" + "="*80)
    # print("📊 TABLA DE MÉTRICAS DE MODELOS")
    # print("="*80)
    # print(df.to_string(index=False, float_format='%.4f'))
    # print("="*80)
    
    # Mostrar mejor modelo
    mejor_modelo = df.iloc[0]
    print(f"\n🏆 MEJOR MODELO: {mejor_modelo['Modelo']}")
    print(f"   R² = {mejor_modelo['R²']:.4f} | RMSE = {mejor_modelo['RMSE']:.4f} | MAE = {mejor_modelo['MAE']:.4f}")
    
    return df

def extraer_metricas_por_tipo(y_true_lista, y_pred_lista, estaciones_ids):
    """
    Extrae todas las métricas organizadas por tipo para análisis posterior.
    
    Args:
        y_true_lista: Lista de valores verdaderos
        y_pred_lista: Lista de predicciones correspondientes
        estaciones_ids: Lista de IDs de estaciones correspondientes
    
    Returns:
        dict: Diccionario con listas de métricas por tipo
            {
                'estaciones': [202, 5, 175, ...],
                'MAE': [1.23, 2.45, 1.67, ...],
                'RMSE': [1.56, 2.78, 1.89, ...],
                'R²': [0.85, 0.72, 0.91, ...],
                'MAPE': [12.5, 18.3, 9.7, ...],
                'Correlación': [0.92, 0.85, 0.95, ...]
            }
    """
    
    # Convertir a arrays numpy optimizado
    y_true_array = np.asarray(y_true_lista, dtype=np.float64)
    y_pred_array = np.asarray(y_pred_lista, dtype=np.float64)
    estaciones_array = np.asarray(estaciones_ids, dtype=np.int32)
    
    # Verificar longitudes
    if not (len(y_pred_array) == len(estaciones_array) == len(y_true_array)):
        raise ValueError("Todas las listas deben tener la misma longitud")
    
    # Obtener estaciones únicas ordenadas
    estaciones_unicas = np.unique(estaciones_array)
    
    # Función para redondeo a cifras significativas
    def round_significant(x, sig_figs=4):
        if not np.isfinite(x) or x == 0:
            return x
        return round(x, sig_figs - int(np.floor(np.log10(abs(x)))) - 1)
    
    # Inicializar diccionario de resultados
    metricas_por_tipo = {
        'estaciones': [],
        'MAE': [],
        'RMSE': [],
        'R²': [],
        'MAPE': [],
        'Correlación': []
    }
    
    # Calcular métricas para cada estación
    for estacion in estaciones_unicas:
        # Usar máscara booleana
        mask = estaciones_array == estacion
        y_true_est = y_true_array[mask]
        y_pred_est = y_pred_array[mask]
        
        # Cálculos vectorizados
        residuos = y_true_est - y_pred_est
        abs_residuos = np.abs(residuos)
        
        n_est = len(y_true_est)
        
        # Métricas básicas
        mae = np.mean(abs_residuos)
        mse = np.mean(residuos**2)
        rmse = np.sqrt(mse)
        
        # R² optimizado
        ss_res = np.sum(residuos**2)
        ss_tot = np.sum((y_true_est - np.mean(y_true_est))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # MAPE robusto
        mask_nonzero = y_true_est != 0
        mape = np.mean(np.abs(residuos[mask_nonzero] / y_true_est[mask_nonzero])) * 100 if np.any(mask_nonzero) else np.inf
        
        # Correlación
        correlation = np.corrcoef(y_true_est, y_pred_est)[0, 1] if n_est > 1 else np.nan
        
        # Agregar a las listas correspondientes
        metricas_por_tipo['estaciones'].append(int(estacion))
        metricas_por_tipo['MAE'].append(round_significant(mae))
        metricas_por_tipo['RMSE'].append(round_significant(rmse))
        metricas_por_tipo['R²'].append(round_significant(r2))
        metricas_por_tipo['MAPE'].append(round_significant(mape))
        metricas_por_tipo['Correlación'].append(round_significant(correlation))
    
    return metricas_por_tipo


def obtener_lista_metrica(y_true_lista, y_pred_lista, estaciones_ids, metrica='R²'):
    """
    Función de conveniencia para obtener solo una métrica específica.
    
    Args:
        y_true_lista: Lista de valores verdaderos
        y_pred_lista: Lista de predicciones correspondientes
        estaciones_ids: Lista de IDs de estaciones correspondientes
        metrica: Métrica a extraer ('MAE', 'RMSE', 'R²', 'MAPE', 'Correlación')
    
    Returns:
        tuple: (estaciones, valores_metrica)
    """
    
    metricas = extraer_metricas_por_tipo(y_true_lista, y_pred_lista, estaciones_ids)
    
    if metrica not in metricas:
        raise ValueError(f"Métrica '{metrica}' no disponible. Opciones: {list(metricas.keys())[1:]}")
    
    return metricas['estaciones'], metricas[metrica]


def estadisticas_metricas_por_estacion(y_true_lista, y_pred_lista, estaciones_ids, 
                                       mostrar_tabla=True, exportar_csv=None):
    """
    Calcula estadísticas agregadas de métricas por estación (REFACTORIZADO).
    
    Args:
        y_true_lista: Lista de valores verdaderos
        y_pred_lista: Lista de predicciones correspondientes
        estaciones_ids: Lista de IDs de estaciones correspondientes
        mostrar_tabla: Si mostrar la tabla de estadísticas
        exportar_csv: Ruta para exportar resultados a CSV (opcional)
    
    Returns:
        pd.DataFrame: Tabla con estadísticas agregadas por métrica
    """
    
    # Usar la función extraer_metricas_por_tipo para obtener todas las métricas
    metricas_por_tipo = extraer_metricas_por_tipo(y_true_lista, y_pred_lista, estaciones_ids)
    
    estaciones_unicas = metricas_por_tipo['estaciones']
    n_estaciones = len(estaciones_unicas)
    n_samples = len(y_true_lista)
    
    if mostrar_tabla:
        print(f"\n📊 Calculando métricas para {n_estaciones} estaciones...")
        print(f"📈 Total de muestras: {n_samples:,}")
    
    # Métricas para calcular estadísticas
    metricas_principales = ['MAE', 'RMSE', 'R²', 'MAPE', 'Correlación']
    
    # Función para redondeo a cifras significativas
    def round_significant(x, sig_figs=4):
        if not np.isfinite(x) or x == 0:
            return x
        return round(x, sig_figs - int(np.floor(np.log10(abs(x)))) - 1)
    
    # Calcular estadísticas de forma vectorizada
    estadisticas_resultados = []
    
    for metrica in metricas_principales:
        if metrica not in metricas_por_tipo:
            continue
            
        # Obtener valores válidos
        valores_array = np.array([x for x in metricas_por_tipo[metrica] if np.isfinite(x)])
        estaciones_validas = [est for i, est in enumerate(estaciones_unicas) 
                             if np.isfinite(metricas_por_tipo[metrica][i])]
        
        if len(valores_array) == 0:
            continue
        
        # Calcular estadísticas usando numpy (vectorizado)
        min_idx = np.argmin(valores_array)
        max_idx = np.argmax(valores_array)
        
        estadisticas = {
            'Métrica': metrica,
            'Mínimo': round_significant(valores_array[min_idx]),
            'Estación_Min': estaciones_validas[min_idx],
            'Máximo': round_significant(valores_array[max_idx]),
            'Estación_Max': estaciones_validas[max_idx],
            'Promedio': round_significant(np.mean(valores_array)),
            'Mediana': round_significant(np.median(valores_array)),
            'Desv_Est': round_significant(np.std(valores_array)),
            'P25': round_significant(np.percentile(valores_array, 25)),
            'P75': round_significant(np.percentile(valores_array, 75)),
            'Rango': round_significant(np.ptp(valores_array)),  # peak-to-peak
            'CV_%': round_significant(np.std(valores_array) / np.mean(valores_array) * 100) if np.mean(valores_array) != 0 else np.inf,
            'N_Estaciones': len(valores_array)
        }
        
        estadisticas_resultados.append(estadisticas)
    
    # Crear DataFrame optimizado
    df_estadisticas = pd.DataFrame(estadisticas_resultados)
    
    # Mostrar tabla si se solicita
    if mostrar_tabla:
        print("="*120)
        
        # Resumen optimizado
        print(f"\n📈 RESUMEN OPTIMIZADO:")
        print(f"   • Estaciones analizadas: {n_estaciones}")
        print(f"   • Muestras procesadas: {n_samples:,}")
        print(f"   • Promedio muestras/estación: {n_samples/n_estaciones:.1f}")
        print(f"   • Métricas calculadas: {len(metricas_principales)}")
        
        # Destacar mejores y peores estaciones con métricas simplificadas
        if not df_estadisticas.empty:
            metricas_destacar = ['R²', 'RMSE', 'MAE', 'MAPE', 'Correlación']
            
            for metrica in metricas_destacar:
                row = df_estadisticas[df_estadisticas['Métrica'] == metrica]
                if not row.empty:
                    r = row.iloc[0]
                    if metrica in ['R²', 'Correlación']:
                        print(f"\n🏆 {metrica} - MEJOR: Est.{r['Estación_Max']} ({r['Máximo']:.4g}) | PEOR: Est.{r['Estación_Min']} ({r['Mínimo']:.4g})")
                    else:
                        print(f"\n✅ {metrica} - MEJOR: Est.{r['Estación_Min']} ({r['Mínimo']:.4g}) | PEOR: Est.{r['Estación_Max']} ({r['Máximo']:.4g})")
    
    # Exportar con manejo de errores optimizado
    if exportar_csv:
        try:
            df_estadisticas.to_csv(exportar_csv, index=False, float_format='%.4g')
            if mostrar_tabla:
                print(f"\n💾 Estadísticas exportadas a: {exportar_csv}")
        except Exception as e:
            if mostrar_tabla:
                print(f"\n❌ Error al exportar CSV: {str(e)}")
    
    return df_estadisticas


def metricas_detalladas_por_estacion(y_true_lista, y_pred_lista, estaciones_ids, 
                                   mostrar_tabla=True, top_n=5, exportar_csv=None):
    """
    Calcula métricas detalladas para cada estación individual (OPTIMIZADO).
    
    Args:
        y_true_lista: Lista de valores verdaderos
        y_pred_lista: Lista de predicciones correspondientes
        estaciones_ids: Lista de IDs de estaciones correspondientes
        mostrar_tabla: Si mostrar la tabla de métricas
        top_n: Número de mejores/peores estaciones a destacar
        exportar_csv: Ruta para exportar resultados a CSV (opcional)
    
    Returns:
        pd.DataFrame: Tabla con métricas detalladas por estación
    """
    
    # Convertir a arrays numpy optimizado
    y_true_array = np.asarray(y_true_lista, dtype=np.float64)
    y_pred_array = np.asarray(y_pred_lista, dtype=np.float64)
    estaciones_array = np.asarray(estaciones_ids, dtype=np.int32)
    
    # Verificar longitudes
    n_samples = len(y_true_array)
    if not (len(y_pred_array) == len(estaciones_array) == n_samples):
        raise ValueError("Todas las listas deben tener la misma longitud")
    
    # Obtener estaciones únicas
    estaciones_unicas = np.unique(estaciones_array)
    n_estaciones = len(estaciones_unicas)
    
    if mostrar_tabla:
        print(f"\n📊 Calculando métricas detalladas para {n_estaciones} estaciones (OPTIMIZADO)...")
    
    # Función para redondeo a cifras significativas
    def round_significant(x, sig_figs=4):
        if not np.isfinite(x) or x == 0:
            return x
        return round(x, sig_figs - int(np.floor(np.log10(abs(x)))) - 1)
    
    # Lista para almacenar resultados (preallocated para eficiencia)
    resultados_detallados = []
    
    # Calcular métricas para cada estación (vectorizado cuando sea posible)
    for estacion in estaciones_unicas:
        # Usar máscara booleana
        mask = estaciones_array == estacion
        y_true_est = y_true_array[mask]
        y_pred_est = y_pred_array[mask]
        
        # Cálculos vectorizados
        residuos = y_true_est - y_pred_est
        abs_residuos = np.abs(residuos)
        
        n_est = len(y_true_est)
        
        # Métricas básicas optimizadas
        mae = np.mean(abs_residuos)
        mse = np.mean(residuos**2)
        rmse = np.sqrt(mse)
        
        # R² optimizado
        ss_res = np.sum(residuos**2)
        ss_tot = np.sum((y_true_est - np.mean(y_true_est))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # MAPE robusto
        mask_nonzero = y_true_est != 0
        mape = np.mean(np.abs(residuos[mask_nonzero] / y_true_est[mask_nonzero])) * 100 if np.any(mask_nonzero) else np.inf
        
        # Métricas adicionales avanzadas
        max_error = np.max(abs_residuos)
        min_error = np.min(abs_residuos)
        median_error = np.median(abs_residuos)
        
        # Correlación optimizada
        correlation = np.corrcoef(y_true_est, y_pred_est)[0, 1] if n_est > 1 else np.nan
        
        # Bias y estadísticas
        bias = np.mean(residuos)
        std_residuos = np.std(residuos)
        
        # Explained variance
        var_residuos = np.var(residuos)
        var_true = np.var(y_true_est)
        explained_var = 1 - (var_residuos / var_true) if var_true > 0 else 0
        
        # Métricas avanzadas adicionales
        mean_true = np.mean(y_true_est)
        mean_pred = np.mean(y_pred_est)
        
        # SMAPE (Symmetric MAPE)
        smape = np.mean(2 * abs_residuos / (np.abs(y_true_est) + np.abs(y_pred_est) + 1e-8)) * 100
        
        # MASE aproximado
        naive_error = np.mean(np.abs(np.diff(y_true_est))) if n_est > 1 else 1
        mase = mae / naive_error if naive_error > 0 else np.inf
        
        # Normalized RMSE
        range_true = np.ptp(y_true_est)  # peak-to-peak
        nrmse = rmse / range_true * 100 if range_true > 0 else np.inf
        
        # CV of RMSE
        cv_rmse = rmse / mean_true * 100 if mean_true != 0 else np.inf
        
        # Quantile errors
        q25_error = np.percentile(abs_residuos, 25)
        q75_error = np.percentile(abs_residuos, 75)
        iqr_error = q75_error - q25_error
        
        # Relative errors
        relative_error = np.mean(abs_residuos / np.maximum(np.abs(y_true_est), 1e-8)) * 100
        
        # Crear resultado con redondeo a 4 cifras significativas (simplificado)
        resultado = {
            'Estación_ID': int(estacion),
            'N_Muestras': n_est,
            'MAE': round_significant(mae),
            'RMSE': round_significant(rmse),
            'R²': round_significant(r2),
            'MAPE (%)': round_significant(mape),
            'Correlación': round_significant(correlation)
        }
        
        resultados_detallados.append(resultado)
    
    # Crear DataFrame optimizado
    df_detallado = pd.DataFrame(resultados_detallados)
    
    # Ordenar por R² descendente (más eficiente)
    df_detallado = df_detallado.sort_values('R²', ascending=False).reset_index(drop=True)
    
    # Agregar rankings múltiples
    df_detallado.insert(1, 'Rank_R²', range(1, len(df_detallado) + 1))
    df_detallado.insert(2, 'Rank_RMSE', df_detallado['RMSE'].rank(method='min').astype(int))
    df_detallado.insert(3, 'Rank_MAE', df_detallado['MAE'].rank(method='min').astype(int))
    
    # Mostrar tabla si se solicita
    if mostrar_tabla:
        print("\n" + "="*120)
        print("📊 MÉTRICAS DETALLADAS POR ESTACIÓN (ordenadas por R²)")
        print("="*120)
        
        # Mostrar tabla con formato optimizado
        with pd.option_context('display.max_columns', None, 'display.width', None):
            print(df_detallado.to_string(index=False, float_format='%.4g'))
        
        print("="*120)
        
        # Mostrar rankings múltiples
        if len(df_detallado) >= top_n:
            rankings_mostrar = [
                ('R²', 'R²', False),  # (columna, nombre, ascendente)
                ('RMSE', 'RMSE', True),
                ('MAE', 'MAE', True),
                ('MAPE (%)', 'MAPE', True),
                ('Correlación', 'Correlación', False)
            ]
            
            for col, nombre, ascendente in rankings_mostrar:
                if col in df_detallado.columns:
                    df_sorted = df_detallado.sort_values(col, ascending=ascendente).head(top_n)
                    
                    emoji = "🏆" if not ascendente else "✅"
                    direccion = "Mayor" if not ascendente else "Menor"
                    
                    print(f"\n{emoji} TOP {top_n} ESTACIONES ({direccion} {nombre}):")
                    for idx, (_, row) in enumerate(df_sorted.iterrows(), 1):
                        print(f"   {idx}. Estación {row['Estación_ID']}: {nombre} = {row[col]:.4g}")
        
        # Resumen estadístico adicional
        print(f"\n� RESUMEN ESTADÍSTICO:")
        metricas_resumen = ['R²', 'RMSE', 'MAE', 'MAPE (%)', 'Correlación']
        for metrica in metricas_resumen:
            if metrica in df_detallado.columns:
                serie = df_detallado[metrica].replace([np.inf, -np.inf], np.nan).dropna()
                if not serie.empty:
                    print(f"   • {metrica}: Media={serie.mean():.4g}, Mediana={serie.median():.4g}, Std={serie.std():.4g}")
    
    # Exportar con formato optimizado
    if exportar_csv:
        try:
            df_detallado.to_csv(exportar_csv, index=False, float_format='%.4g')
            if mostrar_tabla:
                print(f"\n💾 Métricas detalladas exportadas a: {exportar_csv}")
        except Exception as e:
            if mostrar_tabla:
                print(f"\n❌ Error al exportar CSV: {str(e)}")
    
    return df_detallado


def calcular_metricas_multiples(y_true_lista, y_pred_lista):
    """
    Calcula métricas para múltiples conjuntos de predicciones y devuelve listas separadas.
    
    Args:
        y_true_lista: Lista de arrays con valores verdaderos
        y_pred_lista: Lista de arrays con predicciones correspondientes
    
    Returns:
        tuple: (mse_lista, r2_lista, mae_lista, rmse_lista, mape_lista)
    """
    
    # Verificar que las listas tengan la misma longitud
    if len(y_true_lista) != len(y_pred_lista):
        raise ValueError("Las listas de valores verdaderos y predicciones deben tener la misma longitud")
    
    # Inicializar listas de resultados
    mse_lista = []
    r2_lista = []
    mae_lista = []
    rmse_lista = []
    mape_lista = []
    
    # Función para redondeo a cifras significativas
    def round_significant(x, sig_figs=4):
        if not np.isfinite(x) or x == 0:
            return x
        return round(x, sig_figs - int(np.floor(np.log10(abs(x)))) - 1)
    
    # Calcular métricas para cada conjunto
    for i, (y_true, y_pred) in enumerate(zip(y_true_lista, y_pred_lista)):
        # Convertir a arrays numpy
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        # Verificar que tengan la misma longitud
        if len(y_true) != len(y_pred):
            raise ValueError(f"El conjunto {i} tiene diferentes longitudes: y_true={len(y_true)}, y_pred={len(y_pred)}")
        
        # Calcular métricas básicas
        residuos = y_true - y_pred
        abs_residuos = np.abs(residuos)
        
        # MSE
        mse = np.mean(residuos**2)
        mse_lista.append(round_significant(mse))
        
        # MAE
        mae = np.mean(abs_residuos)
        mae_lista.append(round_significant(mae))
        
        # RMSE
        rmse = np.sqrt(mse)
        rmse_lista.append(round_significant(rmse))
        
        # R²
        ss_res = np.sum(residuos**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r2_lista.append(round_significant(r2))
        
        # MAPE
        mask_nonzero = y_true != 0
        if np.any(mask_nonzero):
            mape = np.mean(np.abs(residuos[mask_nonzero] / y_true[mask_nonzero])) * 100
        else:
            mape = np.inf
        mape_lista.append(round_significant(mape))
    
    return mse_lista, r2_lista, mae_lista, rmse_lista, mape_lista

