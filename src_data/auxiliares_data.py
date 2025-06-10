import numpy as np

def get_muestras(df, n_muestras=1000):
    if len(df) <= n_muestras:
        return df
    else:
        return df.sample(n=n_muestras, random_state=42).reset_index(drop=True)