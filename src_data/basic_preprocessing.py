import numpy as np
import pandas as pd

def unite_usuarios(csv_2024, csv_2023):

    df_2024 = pd.read_csv(csv_2024)
    df_2023 = pd.read_csv(csv_2023)

    # eliminar la última columna de 2023
    if df_2023.shape[1] > 0:
        df_2023 = df_2023.iloc[:, :-1]

    # eliminar primera fila de 2023 (nombres de features)
    if not df_2023.empty:
        df_2023 = df_2023.iloc[1:]
    
    # sacar las comillas de los datos en 2023
    df_2023 = df_2023.replace('"', '', regex=True)
    
    # Combine the two DataFrames
    # Usar la primera fila de 2024 como nombres de columnas
    new_columns = df_2024.iloc[0].tolist()
    df_2024 = df_2024.iloc[1:]
    df_2024.columns = new_columns
    df_2023.columns = new_columns

    # Concatenar primero 2023 y luego 2024
    df_combined = pd.concat([df_2023, df_2024], ignore_index=True)
    
    # Remove duplicate rows
    df_combined.drop_duplicates(inplace=True)
    
    return df_combined
    