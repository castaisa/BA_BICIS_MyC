import numpy as np

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
    return count

def histogram(x,y, bins=10):
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    return hist, xedges, yedges