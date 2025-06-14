import numpy as np
import matplotlib.pyplot as plt


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

    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()