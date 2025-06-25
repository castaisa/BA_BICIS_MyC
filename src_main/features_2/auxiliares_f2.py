import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_features(df):
    return list(df.iloc[0])

def obtener_arribos_por_estacion(dataset):
    """
    Obtiene la cantidad total de arribos por estación de un dataset.
    
    Args:
        dataset: DataFrame con columnas target_estacion_X
    
    Returns:
        dict: {estacion_id: total_arribos} ordenado por estacion_id
    """
    # Obtener columnas target
    target_cols = [col for col in dataset.columns if col.startswith('target_estacion_')]
    
    # Diccionario para almacenar resultados
    arribos_por_estacion = {}
    
    # Calcular suma para cada estación
    for col in target_cols:
        estacion_id = int(col.split('_')[-1])  # Extraer ID de estación
        total_arribos = dataset[col].sum()
        arribos_por_estacion[estacion_id] = total_arribos
    
    # Ordenar por ID de estación
    return dict(sorted(arribos_por_estacion.items()))


def graficar_metricas_vs_arribos(arribos_dict, mae_lista, r2_lista, rmse_lista, mape_lista, 
                                estaciones_ids, figsize=(20, 15)):
    """
    Grafica las métricas de rendimiento vs cantidad de arribos por estación.
    
    Args:
        arribos_dict: Diccionario {estacion_id: total_arribos}
        mae_lista: Lista con valores MAE por estación
        r2_lista: Lista con valores R² por estación
        rmse_lista: Lista con valores RMSE por estación
        mape_lista: Lista con valores MAPE por estación
        estaciones_ids: Lista con IDs de estaciones (mismo orden que las métricas)
        figsize: Tamaño de la figura
    """
    
    # Verificar que todas las listas tengan la misma longitud
    n_estaciones = len(estaciones_ids)
    if not all(len(lista) == n_estaciones for lista in [mae_lista, r2_lista, rmse_lista, mape_lista]):
        raise ValueError("Todas las listas de métricas deben tener la misma longitud que estaciones_ids")
    
    # Crear DataFrame con todos los datos
    data = []
    for i, estacion_id in enumerate(estaciones_ids):
        if estacion_id in arribos_dict:
            data.append({
                'estacion_id': estacion_id,
                'arribos': arribos_dict[estacion_id],
                'MAE': mae_lista[i],
                'R²': r2_lista[i],
                'RMSE': rmse_lista[i],
                'MAPE': mape_lista[i]
            })
    
    # Convertir a DataFrame y ordenar por arribos
    df = pd.DataFrame(data)
    df = df.sort_values('arribos').reset_index(drop=True)
    
    print(f"📊 Graficando métricas para {len(df)} estaciones ordenadas por arribos...")
    
    # Crear subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Métricas de Rendimiento vs Cantidad de Arribos por Estación', 
                 fontsize=20, fontweight='bold')
    
    # Colores para cada métrica
    colors = {
        'MAE': '#e74c3c',     # Rojo
        'R²': '#2ecc71',      # Verde
        'RMSE': '#3498db',    # Azul
        'MAPE': '#f39c12'     # Naranja
    }
    
    # Preparar datos para gráficos
    x_pos = np.arange(len(df))
    x_labels = [f"Est.{est}\n({arribos:,})" for est, arribos in zip(df['estacion_id'], df['arribos'])]
    
    # 1. MSE vs Arribos
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(x_pos, df['MAE'], c=colors['MAE'], alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax1.plot(x_pos, df['MAE'], color=colors['MAE'], alpha=0.3, linewidth=1)
    ax1.set_title('MAE vs Arribos', fontweight='bold', fontsize=14)
    ax1.set_ylabel('MAE', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x_pos[::max(1, len(x_pos)//10)])  # Mostrar cada 10 estaciones aprox
    ax1.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), max(1, len(x_labels)//10))], 
                       rotation=45, ha='right', fontsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    
    # 2. R² vs Arribos
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(x_pos, df['R²'], c=colors['R²'], alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax2.plot(x_pos, df['R²'], color=colors['R²'], alpha=0.3, linewidth=1)
    ax2.set_title('R² vs Arribos', fontweight='bold', fontsize=14)
    ax2.set_ylabel('R²', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x_pos[::max(1, len(x_pos)//10)])
    ax2.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), max(1, len(x_labels)//10))], 
                       rotation=45, ha='right', fontsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    
    # 3. RMSE vs Arribos
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(x_pos, df['RMSE'], c=colors['RMSE'], alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax3.plot(x_pos, df['RMSE'], color=colors['RMSE'], alpha=0.3, linewidth=1)
    ax3.set_title('RMSE vs Arribos', fontweight='bold', fontsize=14)
    ax3.set_ylabel('RMSE', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Estaciones (ordenadas por arribos)', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(x_pos[::max(1, len(x_pos)//10)])
    ax3.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), max(1, len(x_labels)//10))], 
                       rotation=45, ha='right', fontsize=10)
    ax3.tick_params(axis='y', labelsize=10)
    
    # 4. MAPE vs Arribos
    ax4 = axes[1, 1]
    scatter4 = ax4.scatter(x_pos, df['MAPE'], c=colors['MAPE'], alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax4.plot(x_pos, df['MAPE'], color=colors['MAPE'], alpha=0.3, linewidth=1)
    ax4.set_title('MAPE vs Arribos', fontweight='bold', fontsize=14)
    ax4.set_ylabel('MAPE (%)', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Estaciones (ordenadas por arribos)', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(x_pos[::max(1, len(x_pos)//10)])
    ax4.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), max(1, len(x_labels)//10))], 
                       rotation=45, ha='right', fontsize=10)
    ax4.tick_params(axis='y', labelsize=10)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Mostrar estadísticas
    print(f"\n📈 ANÁLISIS DE CORRELACIÓN ARRIBOS vs MÉTRICAS:")
    print(f"   • Correlación Arribos-MAE: {np.corrcoef(df['arribos'], df['MAE'])[0,1]:.4f}")
    print(f"   • Correlación Arribos-R²: {np.corrcoef(df['arribos'], df['R²'])[0,1]:.4f}")
    print(f"   • Correlación Arribos-RMSE: {np.corrcoef(df['arribos'], df['RMSE'])[0,1]:.4f}")
    print(f"   • Correlación Arribos-MAPE: {np.corrcoef(df['arribos'], df['MAPE'])[0,1]:.4f}")
    
    # Mostrar extremos
    print(f"\n🎯 ESTACIONES EXTREMAS:")
    print(f"   • Menos arribos: Est.{df.iloc[0]['estacion_id']} ({df.iloc[0]['arribos']:,} arribos)")
    print(f"   • Más arribos: Est.{df.iloc[-1]['estacion_id']} ({df.iloc[-1]['arribos']:,} arribos)")
    
    plt.show()
    
    return df

