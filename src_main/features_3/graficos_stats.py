import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
from sklearn.metrics import mean_absolute_error, r2_score

def plot_distributions_comparison(train_df, val_df, features, figsize=(20, 15)):
    """Compara distribuciones entre train y val"""
    n_features = len(features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, feature in enumerate(features):
        if i < len(axes):
            # Histogramas superpuestos
            axes[i].hist(train_df[feature].dropna(), alpha=0.6, bins=50, 
                        label='Train', density=True, color='blue')
            axes[i].hist(val_df[feature].dropna(), alpha=0.6, bins=50, 
                        label='Val', density=True, color='red')
            axes[i].set_title(f'{feature}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Ocultar ejes vacíos
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_box_comparison(train_df, val_df, features, figsize=(20, 15)):
    """Boxplots comparativos"""
    n_features = len(features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, feature in enumerate(features):
        if i < len(axes):
            data_combined = pd.DataFrame({
                'value': list(train_df[feature].dropna()) + list(val_df[feature].dropna()),
                'dataset': ['Train']*len(train_df[feature].dropna()) + ['Val']*len(val_df[feature].dropna())
            })
            sns.boxplot(data=data_combined, x='dataset', y='value', ax=axes[i])
            axes[i].set_title(f'{feature}')
            axes[i].tick_params(axis='x', rotation=45)
    
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def ks_test_all_features(train_df, val_df, features):
    """Kolmogorov-Smirnov test para todas las features"""
    ks_results = []
    
    for feature in features:
        train_data = train_df[feature].dropna()
        val_data = val_df[feature].dropna()
        
        if len(train_data) > 0 and len(val_data) > 0:
            ks_stat, p_value = stats.ks_2samp(train_data, val_data)
            ks_results.append({
                'feature': feature,
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'different_distributions': p_value < 0.05
            })
    
    ks_df = pd.DataFrame(ks_results)
    ks_df = ks_df.sort_values('ks_statistic', ascending=False)
    
    # Visualizar resultados
    plt.figure(figsize=(15, 8))
    colors = ['red' if x else 'green' for x in ks_df['different_distributions']]
    plt.barh(range(len(ks_df)), ks_df['ks_statistic'], color=colors, alpha=0.7)
    plt.yticks(range(len(ks_df)), ks_df['feature'])
    plt.xlabel('KS Statistic')
    plt.title('Kolmogorov-Smirnov Test Results\n(Red = Significantly Different Distributions)')
    plt.axvline(x=0.1, color='orange', linestyle='--', label='KS=0.1 threshold')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return ks_df

def pca_analysis(train_df, val_df, features, n_components=2):
    """Análisis PCA para ver separación entre datasets"""
    # Combinar datos
    train_data = train_df[features].fillna(train_df[features].median())
    val_data = val_df[features].fillna(val_df[features].median())
    
    combined_data = pd.concat([train_data, val_data])
    labels = ['Train']*len(train_data) + ['Val']*len(val_data)
    
    # PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(combined_data)
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(pca_result[:len(train_data), 0], pca_result[:len(train_data), 1], 
                         alpha=0.6, label='Train', s=20)
    scatter = plt.scatter(pca_result[len(train_data):, 0], pca_result[len(train_data):, 1], 
                         alpha=0.6, label='Val', s=20)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA: Train vs Val Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Contribución de features a los componentes
    feature_importance = pd.DataFrame({
        'feature': features,
        'PC1': pca.components_[0],
        'PC2': pca.components_[1]
    })
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(range(len(features)), feature_importance['PC1'])
    plt.yticks(range(len(features)), features)
    plt.title('PC1 Feature Contributions')
    
    plt.subplot(1, 2, 2)
    plt.barh(range(len(features)), feature_importance['PC2'])
    plt.yticks(range(len(features)), features)
    plt.title('PC2 Feature Contributions')
    
    plt.tight_layout()
    plt.show()
    
    return pca, feature_importance

def tsne_analysis(train_df, val_df, features, perplexity=30):
    """t-SNE para visualización no lineal"""
    train_data = train_df[features].fillna(train_df[features].median())
    val_data = val_df[features].fillna(val_df[features].median())
    
    # Muestrear si hay muchos datos
    if len(train_data) > 10000:
        train_sample = train_data.sample(10000)
        val_sample = val_data.sample(min(10000, len(val_data)))
    else:
        train_sample = train_data
        val_sample = val_data
    
    combined_data = pd.concat([train_sample, val_sample])
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(combined_data)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(tsne_result[:len(train_sample), 0], tsne_result[:len(train_sample), 1], 
               alpha=0.6, label='Train', s=20)
    plt.scatter(tsne_result[len(train_sample):, 0], tsne_result[len(train_sample):, 1], 
               alpha=0.6, label='Val', s=20)
    plt.title('t-SNE: Train vs Val Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def tsne_with_targets(X_train, X_val, y_train, y_val, features, perplexity=30):
    """Hace 2 graficos, una para train y otro para val, donde el color de cada muestra es el target para las primeras x estaciones"""


def correlation_comparison(train_df, val_df, features):
    """Compara matrices de correlación"""
    train_corr = train_df[features].corr()
    val_corr = val_df[features].corr()
    
    # Diferencia en correlaciones
    corr_diff = train_corr - val_corr
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Train correlations
    sns.heatmap(train_corr, annot=False, cmap='coolwarm', center=0, ax=axes[0])
    axes[0].set_title('Train Correlations')
    
    # Val correlations
    sns.heatmap(val_corr, annot=False, cmap='coolwarm', center=0, ax=axes[1])
    axes[1].set_title('Val Correlations')
    
    # Difference
    sns.heatmap(corr_diff, annot=False, cmap='RdBu', center=0, ax=axes[2])
    axes[2].set_title('Correlation Differences (Train - Val)')
    
    plt.tight_layout()
    plt.show()
    
    return train_corr, val_corr, corr_diff


def drift_analysis_by_chunks(train_df, val_df, features, n_chunks=10):
    """Analiza drift dividiendo el dataset en chunks"""
    train_chunks = np.array_split(train_df, n_chunks)
    
    drift_results = []
    
    for i, chunk in enumerate(train_chunks):
        chunk_stats = {}
        for feature in features:
            if feature in chunk.columns:
                chunk_stats[f'{feature}_mean'] = chunk[feature].mean()
                chunk_stats[f'{feature}_std'] = chunk[feature].std()
        
        chunk_stats['chunk'] = i
        chunk_stats['dataset'] = 'train'
        drift_results.append(chunk_stats)
    
    # Agregar val como último chunk
    val_stats = {}
    for feature in features:
        if feature in val_df.columns:
            val_stats[f'{feature}_mean'] = val_df[feature].mean()
            val_stats[f'{feature}_std'] = val_df[feature].std()
    
    val_stats['chunk'] = n_chunks
    val_stats['dataset'] = 'val'
    drift_results.append(val_stats)
    
    drift_df = pd.DataFrame(drift_results)
    
    # Plot evolution of means
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(features[:4]):  # Primeras 4 features
        if i < len(axes):
            mean_col = f'{feature}_mean'
            if mean_col in drift_df.columns:
                train_data = drift_df[drift_df['dataset'] == 'train']
                val_data = drift_df[drift_df['dataset'] == 'val']
                
                axes[i].plot(train_data['chunk'], train_data[mean_col], 
                           'o-', label='Train chunks', alpha=0.7)
                axes[i].scatter(val_data['chunk'], val_data[mean_col], 
                              color='red', s=100, label='Val', zorder=5)
                axes[i].set_title(f'{feature} - Mean Evolution')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return drift_df

def outlier_analysis(train_df, val_df, features):
    """Analiza diferencias en outliers entre train y val"""
    outlier_comparison = []
    
    for feature in features:
        train_data = train_df[feature].dropna()
        val_data = val_df[feature].dropna()
        
        if len(train_data) > 0 and len(val_data) > 0:
            # IQR method
            Q1_train, Q3_train = train_data.quantile([0.25, 0.75])
            IQR_train = Q3_train - Q1_train
            lower_bound_train = Q1_train - 1.5 * IQR_train
            upper_bound_train = Q3_train + 1.5 * IQR_train
            
            outliers_train_pct = ((train_data < lower_bound_train) | 
                                 (train_data > upper_bound_train)).mean() * 100
            
            Q1_val, Q3_val = val_data.quantile([0.25, 0.75])
            IQR_val = Q3_val - Q1_val
            lower_bound_val = Q1_val - 1.5 * IQR_val
            upper_bound_val = Q3_val + 1.5 * IQR_val
            
            outliers_val_pct = ((val_data < lower_bound_val) | 
                               (val_data > upper_bound_val)).mean() * 100
            
            outlier_comparison.append({
                'feature': feature,
                'outliers_train_pct': outliers_train_pct,
                'outliers_val_pct': outliers_val_pct,
                'outlier_diff': outliers_val_pct - outliers_train_pct
            })
    
    outlier_df = pd.DataFrame(outlier_comparison)
    
    plt.figure(figsize=(15, 8))
    x = range(len(outlier_df))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], outlier_df['outliers_train_pct'], 
           width, label='Train', alpha=0.7)
    plt.bar([i + width/2 for i in x], outlier_df['outliers_val_pct'], 
           width, label='Val', alpha=0.7)
    
    plt.xlabel('Features')
    plt.ylabel('Outlier Percentage')
    plt.title('Outlier Comparison: Train vs Val')
    plt.xticks(x, outlier_df['feature'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return outlier_df

def complete_statistical_summary(train_df, val_df, features):
    """Resumen estadístico completo"""
    comparison_stats = []
    
    for feature in features:
        train_data = train_df[feature].dropna()
        val_data = val_df[feature].dropna()
        
        if len(train_data) > 0 and len(val_data) > 0:
            # Test estadísticos
            ks_stat, ks_p = stats.ks_2samp(train_data, val_data)
            
            try:
                mannwhitney_stat, mannwhitney_p = stats.mannwhitneyu(train_data, val_data)
            except:
                mannwhitney_stat, mannwhitney_p = np.nan, np.nan
            
            comparison_stats.append({
                'feature': feature,
                'train_mean': train_data.mean(),
                'val_mean': val_data.mean(),
                'mean_diff_pct': ((val_data.mean() - train_data.mean()) / train_data.mean()) * 100,
                'train_std': train_data.std(),
                'val_std': val_data.std(),
                'std_diff_pct': ((val_data.std() - train_data.std()) / train_data.std()) * 100,
                'train_skew': stats.skew(train_data),
                'val_skew': stats.skew(val_data),
                'train_kurt': stats.kurtosis(train_data),
                'val_kurt': stats.kurtosis(val_data),
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'mannwhitney_p': mannwhitney_p,
                'significant_diff': ks_p < 0.05
            })
    
    stats_df = pd.DataFrame(comparison_stats)
    
    # Visualización de diferencias más significativas
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Diferencias en media
    axes[0,0].barh(range(len(stats_df)), stats_df['mean_diff_pct'])
    axes[0,0].set_yticks(range(len(stats_df)))
    axes[0,0].set_yticklabels(stats_df['feature'])
    axes[0,0].set_xlabel('Mean Difference (%)')
    axes[0,0].set_title('Mean Differences (Val - Train)')
    axes[0,0].axvline(x=0, color='red', linestyle='--')
    
    # Diferencias en std
    axes[0,1].barh(range(len(stats_df)), stats_df['std_diff_pct'])
    axes[0,1].set_yticks(range(len(stats_df)))
    axes[0,1].set_yticklabels(stats_df['feature'])
    axes[0,1].set_xlabel('Std Difference (%)')
    axes[0,1].set_title('Standard Deviation Differences (Val - Train)')
    axes[0,1].axvline(x=0, color='red', linestyle='--')
    
    # KS Statistics
    colors = ['red' if x else 'green' for x in stats_df['significant_diff']]
    axes[1,0].barh(range(len(stats_df)), stats_df['ks_statistic'], color=colors, alpha=0.7)
    axes[1,0].set_yticks(range(len(stats_df)))
    axes[1,0].set_yticklabels(stats_df['feature'])
    axes[1,0].set_xlabel('KS Statistic')
    axes[1,0].set_title('Distribution Differences (Red = Significant)')
    
    # Skewness comparison
    x = range(len(stats_df))
    width = 0.35
    axes[1,1].barh([i - width/2 for i in x], stats_df['train_skew'], 
                   width, label='Train', alpha=0.7)
    axes[1,1].barh([i + width/2 for i in x], stats_df['val_skew'], 
                   width, label='Val', alpha=0.7)
    axes[1,1].set_yticks(x)
    axes[1,1].set_yticklabels(stats_df['feature'])
    axes[1,1].set_xlabel('Skewness')
    axes[1,1].set_title('Skewness Comparison')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return stats_df




def analizar_metricas_por_estacion(y_true, y_pred, target_columns_sorted, df_original):
    """
    Analiza MAE y R² por estación y las grafica vs cantidad de salidas
    """
    print("📊 Calculando métricas por estación...")
    
    # Calcular métricas por estación
    mae_por_estacion = []
    r2_por_estacion = []
    salidas_por_estacion = []
    estaciones_ids = []
    
    for i, target_col in enumerate(target_columns_sorted):
        # Extraer ID de estación
        estacion_id = int(target_col.replace('target_estacion_', ''))
        estaciones_ids.append(estacion_id)
        
        # Calcular métricas para esta estación
        y_true_est = y_true[:, i]
        y_pred_est = y_pred[:, i]
        
        mae = mean_absolute_error(y_true_est, y_pred_est)
        r2 = r2_score(y_true_est, y_pred_est)
        
        mae_por_estacion.append(mae)
        r2_por_estacion.append(r2)
        
        # Contar salidas desde el dataset original
        # Asumir que tenemos acceso al dataset unificado para contar salidas
        # Si no, usar el dataset de train como aproximación
        salidas = (df_original['id_estacion_origen'] == estacion_id).sum()
        salidas_por_estacion.append(salidas)
        
        if i < 5:  # Mostrar primeras 5 estaciones
            print(f"Estación {estacion_id}: MAE={mae:.3f}, R²={r2:.3f}, Salidas={salidas}")
    
    # Crear DataFrame para análisis
    import pandas as pd
    metricas_df = pd.DataFrame({
        'estacion_id': estaciones_ids,
        'mae': mae_por_estacion,
        'r2': r2_por_estacion,
        'total_salidas': salidas_por_estacion
    })
    
    print(f"\n📈 Estadísticas generales:")
    print(f"   • MAE promedio: {np.mean(mae_por_estacion):.3f}")
    print(f"   • R² promedio: {np.mean(r2_por_estacion):.3f}")
    print(f"   • Rango salidas: {min(salidas_por_estacion)} - {max(salidas_por_estacion)}")
    
    return metricas_df

def graficar_metricas_vs_salidas(metricas_df):
    """
    Grafica MAE y R² vs cantidad de salidas por estación
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico 1: MAE vs Salidas
    scatter1 = ax1.scatter(metricas_df['total_salidas'], metricas_df['mae'], 
                          alpha=0.6, c=metricas_df['mae'], cmap='viridis_r', s=50)
    ax1.set_xlabel('Total de Salidas por Estación')
    ax1.set_ylabel('MAE (Mean Absolute Error)')
    ax1.set_title('MAE vs Total de Salidas por Estación')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')  # Escala logarítmica para mejor visualización
    
    # Añadir línea de tendencia
    z1 = np.polyfit(np.log(metricas_df['total_salidas'] + 1), metricas_df['mae'], 1)
    p1 = np.poly1d(z1)
    x_trend = np.logspace(np.log10(metricas_df['total_salidas'].min()), 
                         np.log10(metricas_df['total_salidas'].max()), 100)
    ax1.plot(x_trend, p1(np.log(x_trend + 1)), "r--", alpha=0.8, label='Tendencia')
    ax1.legend()
    
    # Colorbar para MAE
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('MAE')
    
    # Gráfico 2: R² vs Salidas
    scatter2 = ax2.scatter(metricas_df['total_salidas'], metricas_df['r2'], 
                          alpha=0.6, c=metricas_df['r2'], cmap='viridis', s=50)
    ax2.set_xlabel('Total de Salidas por Estación')
    ax2.set_ylabel('R² (Coeficiente de Determinación)')
    ax2.set_title('R² vs Total de Salidas por Estación')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')  # Escala logarítmica para mejor visualización
    
    # Añadir línea de tendencia
    z2 = np.polyfit(np.log(metricas_df['total_salidas'] + 1), metricas_df['r2'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(x_trend, p2(np.log(x_trend + 1)), "r--", alpha=0.8, label='Tendencia')
    ax2.legend()
    
    # Colorbar para R²
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('R²')
    
    plt.tight_layout()
    plt.show()
    
    # Análisis de correlación
    corr_mae_salidas = np.corrcoef(np.log(metricas_df['total_salidas'] + 1), metricas_df['mae'])[0,1]
    corr_r2_salidas = np.corrcoef(np.log(metricas_df['total_salidas'] + 1), metricas_df['r2'])[0,1]
    
    print(f"\n🔍 Análisis de correlación:")
    print(f"   • Correlación MAE vs log(Salidas): {corr_mae_salidas:.3f}")
    print(f"   • Correlación R² vs log(Salidas): {corr_r2_salidas:.3f}")
    
    return metricas_df

def mostrar_estaciones_extremas(metricas_df, n=5):
    """
    Muestra las estaciones con mejores y peores métricas
    """
    print(f"\n🏆 TOP {n} ESTACIONES CON MENOR MAE:")
    best_mae = metricas_df.nsmallest(n, 'mae')
    for _, row in best_mae.iterrows():
        print(f"   Estación {row['estacion_id']:3.0f}: MAE={row['mae']:.3f}, R²={row['r2']:.3f}, Salidas={row['total_salidas']:,}")
    
    print(f"\n🥇 TOP {n} ESTACIONES CON MAYOR R²:")
    best_r2 = metricas_df.nlargest(n, 'r2')
    for _, row in best_r2.iterrows():
        print(f"   Estación {row['estacion_id']:3.0f}: R²={row['r2']:.3f}, MAE={row['mae']:.3f}, Salidas={row['total_salidas']:,}")
    
    print(f"\n⚠️ TOP {n} ESTACIONES CON MAYOR MAE:")
    worst_mae = metricas_df.nlargest(n, 'mae')
    for _, row in worst_mae.iterrows():
        print(f"   Estación {row['estacion_id']:3.0f}: MAE={row['mae']:.3f}, R²={row['r2']:.3f}, Salidas={row['total_salidas']:,}")
    
    print(f"\n📉 TOP {n} ESTACIONES CON MENOR R²:")
    worst_r2 = metricas_df.nsmallest(n, 'r2')
    for _, row in worst_r2.iterrows():
        print(f"   Estación {row['estacion_id']:3.0f}: R²={row['r2']:.3f}, MAE={row['mae']:.3f}, Salidas={row['total_salidas']:,}")