import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

