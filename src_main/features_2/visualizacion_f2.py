def graficar_pca(X_train, y_train, percentile_cutoff=80, n_features_grafico=10):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Combinar X e y para filtrar por valor del target
    combined = pd.DataFrame(X_train.copy())
    combined['target'] = y_train
    
    # Calcular el valor de corte del percentil
    cutoff_value = combined['target'].quantile(percentile_cutoff/100)
    print(f"Eliminando el {100-percentile_cutoff}% superior de las muestras (target > {cutoff_value})")
    
    # Filtrar las muestras
    filtered = combined[combined['target'] <= cutoff_value]
    
    # Separar nuevamente X e y
    X_filtered = filtered.drop('target', axis=1)
    y_filtered = filtered['target']
    
    print(f"Muestras originales: {len(X_train)}, Muestras filtradas: {len(X_filtered)}")
    
    # Estandarizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    
    # Aplicar PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Graficar los resultados con colores según el target
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_filtered, alpha=0.7, cmap='viridis', s=50)
    plt.colorbar(scatter, label='Cantidad de bicis')
    plt.title('PCA de las características de la estación (sin outliers)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(alpha=0.3)
    plt.show()
    
    # Mostrar la varianza explicada por cada componente
    explained_variance = pca.explained_variance_ratio_
    print(f"Varianza explicada por los 2 primeros componentes: {sum(explained_variance)*100:.2f}%")
    print(f"Componente 1: {explained_variance[0]*100:.2f}%")
    print(f"Componente 2: {explained_variance[1]*100:.2f}%")
    
    # Analizar importancia de las características
    feature_names = X_filtered.columns
    components_df = pd.DataFrame(pca.components_.T, index=feature_names, columns=['PC1', 'PC2'])
    
    # Calcular importancia total (suma de valores absolutos de ambos componentes)
    components_df['Importancia_Total'] = components_df['PC1'].abs() + components_df['PC2'].abs()
    
    # Ordenar por importancia total
    top_features = components_df.sort_values('Importancia_Total', ascending=False)
    
    # Obtener las mejores n features para el gráfico
    top_n_features = top_features.head(n_features_grafico)
    
    # Imprimir las mejores 3 features
    print(f"\n🏆 TOP 3 FEATURES MÁS IMPORTANTES:")
    for i, (feature_name, row) in enumerate(top_features.head(3).iterrows(), 1):
        print(f"{i}. {feature_name} (Importancia: {row['Importancia_Total']:.4f})")
        print(f"   PC1: {row['PC1']:.4f}, PC2: {row['PC2']:.4f}")

    print(f"\n📊 TOP {n_features_grafico} FEATURES:")
    print("-" * 70)
    for i, (feature_name, row) in enumerate(top_n_features.iterrows(), 1):
        print(f"{i:2d}. {feature_name:<35} | Total: {row['Importancia_Total']:.4f} | PC1: {row['PC1']:6.4f} | PC2: {row['PC2']:6.4f}")
    
    return pca, scaler

# Aplicar la función a los datos
