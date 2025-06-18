import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from math import radians, sin, cos, sqrt, atan2
import seaborn as sns



def get_muestras(df, n_muestras=1000):
    if len(df) <= n_muestras:
        return df
    else:
        return df.sample(n=n_muestras, random_state=42).reset_index(drop=True)

def count_samples(df):
    """
    Cuenta cuántas muestras hay para cada estación (id_estacion_destino).
    Devuelve un diccionario con ID de estación como clave y cantidad como valor.
    """
    if 'id_estacion_destino' not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'id_estacion_destino'.")
    
    count = df['id_estacion_destino'].value_counts().to_dict()

    # ordenamos por valor de llave
    sorted_counts = dict(sorted(count.items(), key=lambda x: int(x[0]) if isinstance(x[0], (int, float, np.integer)) else float('inf')))

    return sorted_counts

# aux_d.plot_histogram(stations, arrivals, 'Llegadas a estaciones', 'Estaciones', 'Llegadas')

def plot_histogram(x:list, y:list, title, xlabel, ylabel):
    """
    Dibuja un histograma de barras.
    """

    # Convert x to string to avoid dtype issues with matplotlib
    x = [str(val) for val in x]

    plt.figure(figsize=(7, 3))
    plt.bar(x, y, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks([])  # Remove numbers/labels from x axis
    plt.tight_layout()
    plt.show()


def split_dataframe(df, train_size=0.8, val_size=0.1, test_size=0.1, chronological=True, date_column='fecha_destino_recorrido'):    
    if chronological:
        # Ordenar por fecha si se requiere división cronológica
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)
    
    total_samples = len(df)
    
    # Calcular índices para los splits
    train_end = int(train_size * total_samples)
    val_end = train_end + int(val_size * total_samples)
    
    # Dividir los datos
    if chronological:
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
    else:
        # División aleatoria si no es cronológica
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
    
    # Resetear índices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Verificación de tamaños
    print(f"Total muestras: {total_samples}")
    print(f"Train: {len(train_df)} ({len(train_df)/total_samples:.1%})")
    print(f"Val: {len(val_df)} ({len(val_df)/total_samples:.1%})")
    print(f"Test: {len(test_df)} ({len(test_df)/total_samples:.1%})")
    
    return train_df, val_df, test_df    


def count_station_arrivals(df):
    """
    Cuenta cuántas llegadas tiene cada estación.
    Devuelve un diccionario con ID de estación como clave y cantidad de llegadas como valor.
    """
    if 'id_estacion_destino' not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'id_estacion_destino'.")
    
    arrivals = df['id_estacion_destino'].value_counts().to_dict()
    print("ids: ", arrivals.keys())

    
    # Ordenar por ID de estación
    sorted_arrivals = dict(sorted(arrivals.items(), key=lambda x: int(x[0]) if isinstance(x[0], (int, float, np.integer)) else float('inf')))
    
    return sorted_arrivals

def filter_by_value(df, column, value):
    """
    Filtra un DataFrame devolviendo todas las filas que tienen un valor específico en una columna.
    
    Args:
        df (pd.DataFrame): DataFrame a filtrar
        column (str): Nombre de la columna por la cual filtrar
        value: Valor a buscar en la columna
    
    Returns:
        pd.DataFrame: DataFrame filtrado con las filas que coinciden
    """
    if column not in df.columns:
        raise ValueError(f"La columna '{column}' no existe en el DataFrame.")
    
    filtered_df = df[df[column] == value].reset_index(drop=True)
    
    return filtered_df


def plot_station_network_optimized(df, min_trips=5, figsize=(12, 8), layout_algorithm='spring', 
                                   show_labels=True, label_threshold=50):
    """
    Versión optimizada que dibuja un grafo de viajes entre estaciones.
    
    Args:
        df (pd.DataFrame): DataFrame con datos de recorridos
        min_trips (int): Número mínimo de viajes para mostrar una conexión
        figsize (tuple): Tamaño de la figura
        layout_algorithm (str): 'spring', 'circular', 'kamada_kawai', 'random'
        show_labels (bool): Si mostrar etiquetas de los nodos
        label_threshold (int): Máximo número de nodos para mostrar todas las etiquetas
    """
    
    if 'id_estacion_origen' not in df.columns or 'id_estacion_destino' not in df.columns:
        raise ValueError("El DataFrame debe contener las columnas 'id_estacion_origen' e 'id_estacion_destino'.")
    
    print("Procesando datos...")
    
    # 1. OPTIMIZACIÓN: Usar groupby más eficiente
    trip_counts = (df.groupby(['id_estacion_origen', 'id_estacion_destino'])
                     .size()
                     .reset_index(name='count'))
    
    # Filtrar conexiones con pocos viajes
    trip_counts = trip_counts[trip_counts['count'] >= min_trips]
    
    if len(trip_counts) == 0:
        print("No hay suficientes datos para mostrar el grafo.")
        return
    
    print(f"Conexiones encontradas: {len(trip_counts)}")
    
    # 2. OPTIMIZACIÓN: Crear grafo más eficientemente
    G = nx.from_pandas_edgelist(trip_counts, 
                               source='id_estacion_origen',
                               target='id_estacion_destino', 
                               edge_attr='count',
                               create_using=nx.DiGraph())
    
    # 3. OPTIMIZACIÓN: Calcular grados usando pandas (MUCHO más rápido)
    print("Calculando estadísticas de estaciones...")
    
    # Calcular in-degree y out-degree por separado
    in_degrees = trip_counts.groupby('id_estacion_destino')['count'].sum()
    out_degrees = trip_counts.groupby('id_estacion_origen')['count'].sum()
    
    # Combinar grados (manejar estaciones que solo tienen in o out)
    all_stations = set(trip_counts['id_estacion_origen']) | set(trip_counts['id_estacion_destino'])
    station_degrees = {}
    
    for station in all_stations:
        in_deg = in_degrees.get(station, 0)
        out_deg = out_degrees.get(station, 0)
        station_degrees[station] = in_deg + out_deg
    
    # 4. OPTIMIZACIÓN: Layout más eficiente según el tamaño del grafo
    print(f"Calculando layout para {len(G.nodes())} nodos...")
    
    if len(G.nodes()) > 100:
        # Para grafos grandes, usar layout más rápido
        if layout_algorithm == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=20)  # Menos iteraciones
        elif layout_algorithm == 'circular':
            pos = nx.circular_layout(G)
        elif layout_algorithm == 'random':
            pos = nx.random_layout(G)
        else:
            pos = nx.spring_layout(G, k=1, iterations=20)
    else:
        # Para grafos pequeños, usar layout de mejor calidad
        if layout_algorithm == 'spring':
            pos = nx.spring_layout(G, k=3, iterations=50)
        elif layout_algorithm == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout_algorithm == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, k=3, iterations=50)
    
    # 5. Preparar visualización
    max_degree = max(station_degrees.values())
    min_degree = min(station_degrees.values())
    
    # Crear arrays ordenados para los nodos del grafo
    node_list = list(G.nodes())
    node_sizes = [300 + (station_degrees[node] / max_degree) * 2000 for node in node_list]
    node_colors = [(station_degrees[node] - min_degree) / (max_degree - min_degree) 
                   for node in node_list]
    
    # 6. OPTIMIZACIÓN: Limitar edges para grafos muy densos
    edges_to_draw = list(G.edges())
    if len(edges_to_draw) > 1000:
        print(f"Grafo muy denso ({len(edges_to_draw)} edges). Mostrando solo los {1000} más importantes...")
        # Ordenar edges por peso y tomar los top 1000
        edge_weights = [(u, v, G[u][v]['count']) for u, v in edges_to_draw]
        edge_weights.sort(key=lambda x: x[2], reverse=True)
        edges_to_draw = [(u, v) for u, v, w in edge_weights[:1000]]
    
    # 7. Crear visualización
    print("Creando visualización...")
    plt.figure(figsize=figsize)
    
    # Dibujar edges (solo los seleccionados)
    nx.draw_networkx_edges(G, pos, 
                          edgelist=edges_to_draw,
                          alpha=0.3, 
                          edge_color='gray', 
                          arrows=True, 
                          arrowsize=10,
                          arrowstyle='->')
    
    # Dibujar nodes
    nodes = nx.draw_networkx_nodes(G, pos,
                                  nodelist=node_list,
                                  node_size=node_sizes,
                                  node_color=node_colors,
                                  cmap=plt.cm.Blues,
                                  alpha=0.8)
    
    # Labels configurables
    if show_labels:
        if len(G.nodes()) <= label_threshold:
            # Mostrar todas las etiquetas si el grafo no es muy grande
            nx.draw_networkx_labels(G, pos, font_size=8)
            print(f"Mostrando etiquetas de todas las {len(G.nodes())} estaciones")
        else:
            # Solo mostrar labels de las estaciones más importantes
            num_labels = min(20, len(G.nodes()) // 5)  # Máximo 20 o 1/5 del total
            important_stations = sorted(station_degrees.items(), key=lambda x: x[1], reverse=True)[:num_labels]
            important_labels = {station: station for station, _ in important_stations}
            nx.draw_networkx_labels(G, pos, labels=important_labels, font_size=8)
            print(f"Mostrando etiquetas de las {num_labels} estaciones más activas")
    else:
        print("Etiquetas deshabilitadas")
    
    plt.title(f'Red de Viajes entre Estaciones\n(Tamaño y color según cantidad de viajes, min_trips={min_trips})')
    plt.axis('off')
    
    # Colorbar
    plt.colorbar(nodes, label='Cantidad de viajes')
    plt.tight_layout()
    plt.show()
    
    # Estadísticas detalladas
    print(f"\n{'='*60}")
    print(f"📊 ESTADÍSTICAS DETALLADAS DE LA RED DE ESTACIONES")
    print(f"{'='*60}")
    
    # Estadísticas básicas
    total_trips_filtered = trip_counts['count'].sum()
    avg_trips_per_connection = total_trips_filtered / len(trip_counts)
    
    print(f"\n🚴 DATOS GENERALES:")
    print(f"   • Total de estaciones en la red: {len(G.nodes())}")
    print(f"   • Total de conexiones (rutas): {len(G.edges())}")
    print(f"   • Conexiones mostradas: {len(edges_to_draw)}")
    print(f"   • Total de viajes (≥{min_trips}): {total_trips_filtered:,}")
    print(f"   • Promedio viajes por conexión: {avg_trips_per_connection:.1f}")
    
    # Densidad del grafo
    max_possible_edges = len(G.nodes()) * (len(G.nodes()) - 1)  # Grafo dirigido
    density = len(G.edges()) / max_possible_edges if max_possible_edges > 0 else 0
    print(f"   • Densidad del grafo: {density:.3f} ({density*100:.1f}%)")
    
    # Estadísticas de grados
    degrees_values = list(station_degrees.values())
    print(f"\n📈 DISTRIBUCIÓN DE ACTIVIDAD:")
    print(f"   • Estación más activa: {max_degree:,} viajes")
    print(f"   • Estación menos activa: {min_degree:,} viajes")
    print(f"   • Promedio de viajes por estación: {np.mean(degrees_values):.1f}")
    print(f"   • Mediana de viajes por estación: {np.median(degrees_values):.1f}")
    print(f"   • Desviación estándar: {np.std(degrees_values):.1f}")
    
    # Top estaciones más activas
    top_stations = sorted(station_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n🏆 TOP 10 ESTACIONES MÁS ACTIVAS:")
    for i, (station, trips) in enumerate(top_stations, 1):
        percentage = (trips / total_trips_filtered) * 100
        print(f"   {i:2d}. Estación {station}: {trips:,} viajes ({percentage:.1f}%)")
    
    # Análisis de conectividad
    if len(G.nodes()) > 1:
        # Componentes conectados (en versión no dirigida para conectividad general)
        G_undirected = G.to_undirected()
        connected_components = list(nx.connected_components(G_undirected))
        largest_component_size = len(max(connected_components, key=len))
        
        print(f"\n🔗 ANÁLISIS DE CONECTIVIDAD:")
        print(f"   • Componentes conectados: {len(connected_components)}")
        print(f"   • Tamaño del componente principal: {largest_component_size} estaciones ({largest_component_size/len(G.nodes())*100:.1f}%)")
        
        if len(connected_components) == 1:
            print(f"   ✅ La red está completamente conectada")
        else:
            print(f"   ⚠️  La red tiene {len(connected_components)} subgrupos separados")
    
    # Análisis de centralidad (solo para grafos no muy grandes)
    if len(G.nodes()) <= 200:
        print(f"\n🎯 ANÁLISIS DE CENTRALIDAD:")
        
        # Centralidad por grados (ya la tenemos)
        degree_centrality = {node: station_degrees[node] for node in G.nodes()}
        top_degree = max(degree_centrality.items(), key=lambda x: x[1])
        
        # Centralidad de intermediación (betweenness)
        betweenness = nx.betweenness_centrality(G, weight='count')
        top_betweenness = max(betweenness.items(), key=lambda x: x[1])
        
        print(f"   • Estación con mayor centralidad por grado: {top_degree[0]} ({top_degree[1]:,} viajes)")
        print(f"   • Estación con mayor centralidad de intermediación: {top_betweenness[0]} ({top_betweenness[1]:.3f})")
    else:
        print(f"\n🎯 ANÁLISIS DE CENTRALIDAD: Omitido (grafo muy grande, >200 nodos)")
    
    # Información sobre filtros aplicados
    original_connections = len(df.groupby(['id_estacion_origen', 'id_estacion_destino']).size())
    filtered_out = original_connections - len(trip_counts)
    
    print(f"\n🔍 FILTROS APLICADOS:")
    print(f"   • Conexiones originales: {original_connections:,}")
    print(f"   • Conexiones filtradas (< {min_trips} viajes): {filtered_out:,}")
    print(f"   • Conexiones mostradas: {len(trip_counts):,} ({len(trip_counts)/original_connections*100:.1f}%)")
    
    print(f"\n{'='*60}")
    
    return {
        'graph': G,
        'station_degrees': station_degrees,
        'trip_counts': trip_counts,
        'stats': {
            'nodes': len(G.nodes()),
            'edges': len(G.edges()),
            'total_trips': total_trips_filtered,
            'density': density,
            'max_degree': max_degree,
            'min_degree': min_degree,
            'avg_degree': np.mean(degrees_values)
        }
    }

def filter_stations_by_min_trips(df, min_trips=5, show_stats=True):
    """
    Filtra conexiones entre estaciones manteniendo aquellas que cumplen el mínimo de viajes,
    similar al comportamiento de plot_station_network_optimized.
    
    Args:
        df (pd.DataFrame): DataFrame con datos de recorridos
        min_trips (int): Número mínimo de viajes requeridos por conexión
        
    Returns:
        dict: Diccionario con resultados del filtrado
    """
    if not all(col in df.columns for col in ['id_estacion_origen', 'id_estacion_destino']):
        raise ValueError("El DataFrame debe contener columnas de origen y destino")

    # 1. Contar viajes por cada conexión (par origen-destino)
    conexiones = df.groupby(['id_estacion_origen', 'id_estacion_destino']).size().reset_index(name='count')
    
    # 2. Identificar conexiones que cumplen el mínimo
    conexiones_validas = conexiones[conexiones['count'] >= min_trips]
    
    # 3. Filtrar el DataFrame original manteniendo solo las conexiones válidas
    # (usando merge para conservar todas las columnas originales)
    filtered_df = pd.merge(df, 
                         conexiones_validas[['id_estacion_origen', 'id_estacion_destino']],
                         on=['id_estacion_origen', 'id_estacion_destino'],
                         how='inner')
    
    # 4. Identificar estaciones afectadas
    estaciones_originales = set(df['id_estacion_origen']).union(set(df['id_estacion_destino']))
    estaciones_finales = set(filtered_df['id_estacion_origen']).union(set(filtered_df['id_estacion_destino']))
    estaciones_eliminadas = estaciones_originales - estaciones_finales
    
    # 5. Calcular estadísticas
    stats = {
        'initial_trips': len(df),
        'final_trips': len(filtered_df),
        'initial_stations': len(estaciones_originales),
        'final_stations': len(estaciones_finales),
        'removed_stations': len(estaciones_eliminadas),
        'initial_connections': len(conexiones),
        'final_connections': len(conexiones_validas),
        'retention_trips': len(filtered_df) / len(df) * 100,
        'retention_stations': len(estaciones_finales) / len(estaciones_originales) * 100,
        'min_trips_threshold': min_trips,
        'removed_stations_list': list(estaciones_eliminadas)
    }
    
    # 6. Reporte detallado
    if show_stats:
        print(f"=== FILTRADO (min_trips={min_trips}) ===")
        print(f"Viajes originales: {stats['initial_trips']:,}")
        print(f"Viajes conservados: {stats['final_trips']:,} ({stats['retention_trips']:.2f}%)")
        print(f"\nEstaciones originales: {stats['initial_stations']}")
        print(f"Estaciones conservadas: {stats['final_stations']} ({stats['retention_stations']:.2f}%)")
        print(f"Estaciones eliminadas: {stats['removed_stations']}")
        print(f"\nConexiones originales: {stats['initial_connections']}")
        print(f"Conexiones conservadas: {stats['final_connections']}")
        
        if stats['removed_stations'] > 0:
            print(f"\nEstaciones eliminadas (sin conexiones válidas):")
            print(stats['removed_stations_list'])
    
    return {
        'filtered_df': filtered_df.reset_index(drop=True),
        'stats': stats,
        'valid_stations': list(estaciones_finales),
        'valid_connections': conexiones_validas
    }