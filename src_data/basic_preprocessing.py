import numpy as np
import pandas as pd

def preprocess_old_csv(csv_path):
    df = pd.read_csv(csv_path)
    # eliminar la última columna
    if df.shape[1] > 0:
        df = df.iloc[:, :-1]
    # eliminar primera fila (nombres de features)
    if not df.empty:
        df = df.iloc[1:]
    # sacar las comillas de los datos
    df = df.replace('"', '', regex=True)
    return df

def unite_usuarios(csv_2024, csv_2023, csv_2022, csv_2021, csv_2020):
    # Procesar archivos antiguos
    dfs_old = [preprocess_old_csv(f) for f in [csv_2023, csv_2022, csv_2021, csv_2020]]

    df_2024 = pd.read_csv(csv_2024)
    new_columns = df_2024.columns.tolist()

    # print("df 2024: ", df_2024.head(3))

    # Asignar columnas a los antiguos
    for i in range(len(dfs_old)):
        dfs_old[i].columns = new_columns

    # Concatenar todos: 2020, 2021, 2022, 2023, 2024
    df_combined = pd.concat([df_2024] + dfs_old, ignore_index=True)
    # df_combined.drop_duplicates(inplace=True)
    # print("df_combined: ", df_combined.head(3))
    return df_combined

def 