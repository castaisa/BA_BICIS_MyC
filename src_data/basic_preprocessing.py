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
    dfs_old = [preprocess_old_csv(f) for f in [csv_2020, csv_2021, csv_2022, csv_2023]]

    df_2024 = pd.read_csv(csv_2024)
    new_columns = df_2024.columns.tolist()

    # print("df 2024: ", df_2024.head(3))

    # Asignar columnas a los antiguos
    for i in range(len(dfs_old)):
        dfs_old[i].columns = new_columns

    # Concatenar todos: 2020, 2021, 2022, 2023, 2024
    df_combined = pd.concat(dfs_old + [df_2024], ignore_index=True)

    df_cut, eliminated = cut_users(df_combined, '2024-08-31')
    # df_combined.drop_duplicates(inplace=True)
    # print("df_combined: ", df_combined.head(3))
    return df_cut, eliminated

def cut_users(df, fecha_limite):
    # Convertir columna fecha_alta a datetime
    df['fecha_alta'] = pd.to_datetime(df['fecha_alta'], errors='coerce')

    # Convertir fecha_limite también si viene como string
    if isinstance(fecha_limite, str):
        fecha_limite = pd.to_datetime(fecha_limite)

    # Filas a eliminar: fecha posterior al límite
    eliminated_rows = df[df['fecha_alta'] > fecha_limite].index

    # Eliminar esas filas
    df.drop(eliminated_rows, inplace=True)

    return df, eliminated_rows


#Concatenar los datsets de '20 '21 '22 '23 y '24
def concat_recorridos(csv_2024, csv_2023, csv_2022, csv_2021, csv_2020):
    """
    Concatenates multiple CSV files containing trip data into a single DataFrame.
    
    Parameters:
    csv_2020 (str): Path to the CSV file for the year 2020.
    csv_2021 (str): Path to the CSV file for the year 2021.
    csv_2022 (str): Path to the CSV file for the year 2022.
    csv_2023 (str): Path to the CSV file for the year 2023.
    csv_2024 (str): Path to the CSV file for the year 2024.
    
    Returns:
    pd.DataFrame: A DataFrame containing the combined trip data.
    """
    df_2020 = pd.read_csv(csv_2020)
    df_2021 = pd.read_csv(csv_2021)
    df_2022 = pd.read_csv(csv_2022)
    df_2023 = pd.read_csv(csv_2023)
    df_2024 = pd.read_csv(csv_2024)

    # Combine the DataFrames
    df_combined = pd.concat([df_2020, df_2021, df_2022, df_2023, df_2024], ignore_index=True)
    
    # Remove duplicate rows
    df_combined.drop_duplicates(inplace=True)
    
    return df_combined

