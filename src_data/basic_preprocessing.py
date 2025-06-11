import numpy as np
import pandas as pd

def preprocess_old_csv(csv_path):
    df = pd.read_csv(csv_path)
    # eliminar la última columna
    # if df.shape[1] > 0:
    #     df = df.iloc[:, :-1]
    dni_column = 'Customer.Has.Dni..Yes...No.'
    if dni_column in df.columns:
        df.drop(columns=[dni_column], inplace=True)
    # eliminar primera fila (nombres de features)

    # if not df.empty:
    #     df = df.iloc[1:]
    
    if df.shape[0] > 0:
        first_row = df.iloc[0].astype(str).str.strip().str.lower().tolist()
        col_names = df.columns.str.strip().str.lower().tolist()
        if first_row == col_names:
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

    # ordenar por fecha_alta
    df_combined['fecha_alta'] = pd.to_datetime(df_combined['fecha_alta'], errors='coerce')
    df_combined.sort_values(by='fecha_alta', inplace=True)
    df_combined.reset_index(drop=True, inplace=True)


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
# def limpiar_recorridos(csv_2024, csv_2023, csv_2022, csv_2021, csv_2020):
#     """
#     Concatenates multiple CSV files containing trip data into a single DataFrame.
    
#     Parameters:
#     csv_2020 (str): Path to the CSV file for the year 2020.
#     csv_2021 (str): Path to the CSV file for the year 2021.
#     csv_2022 (str): Path to the CSV file for the year 2022.
#     csv_2023 (str): Path to the CSV file for the year 2023.
#     csv_2024 (str): Path to the CSV file for the year 2024.
    
#     Returns:
#     pd.DataFrame: A DataFrame containing the combined trip data.
#     """
#     df_2020 = pd.read_csv(csv_2020)
#     df_2021 = pd.read_csv(csv_2021)
#     df_2022 = pd.read_csv(csv_2022)
#     df_2023 = pd.read_csv(csv_2023)
#     df_2024 = pd.read_csv(csv_2024)

#     #A todos menos 2024 se les saca la primera columna """""
#     df_2020 = df_2020.iloc[:, 1:]
#     df_2021 = df_2021.iloc[:, 1:]
#     df_2022 = df_2022.iloc[:, 1:]
#     df_2023 = df_2023.iloc[:, 1:]

#     #a 2021 tambien le saco la ultima
#     df_2021 = df_2021.iloc[:, :-1]

#     #A 2022 le cambio el nobre de la columna Género a género
#     df_2022.rename(columns={'Género': 'género'}, inplace=True)

#     #A 2024 tambien
#     df_2024.rename(columns={'genero': 'género'}, inplace=True)
#     df_2024.rename(columns={'id_recorrido': 'Id_recorrido'}, inplace=True)

#     #A esos mismos datasets, se les saca BAEcobici del id, la estacion y el usuario
#     df_2020['Id_recorrido'] = df_2020['Id_recorrido'].str.replace('BAEcobici', '', regex=False)
#     df_2021['Id_recorrido'] = df_2021['Id_recorrido'].str.replace('BAEcobici', '', regex=False)
#     df_2022['Id_recorrido'] = df_2022['Id_recorrido'].str.replace('BAEcobici', '', regex=False)
#     df_2023['Id_recorrido'] = df_2023['Id_recorrido'].str.replace('BAEcobici', '', regex=False)

#     df_2020['id_estacion_origen'] = df_2020['id_estacion_origen'].str.replace('BAEcobici', '', regex=False)
#     df_2021['id_estacion_origen'] = df_2021['id_estacion_origen'].str.replace('BAEcobici', '', regex=False)
#     df_2022['id_estacion_origen'] = df_2022['id_estacion_origen'].str.replace('BAEcobici', '', regex=False)
#     df_2023['id_estacion_origen'] = df_2023['id_estacion_origen'].str.replace('BAEcobici', '', regex=False)

#     df_2020['id_estacion_destino'] = df_2020['id_estacion_destino'].str.replace('BAEcobici', '', regex=False)
#     df_2021['id_estacion_destino'] = df_2021['id_estacion_destino'].str.replace('BAEcobici', '', regex=False)
#     df_2022['id_estacion_destino'] = df_2022['id_estacion_destino'].str.replace('BAEcobici', '', regex=False)
#     df_2023['id_estacion_destino'] = df_2023['id_estacion_destino'].str.replace('BAEcobici', '', regex=False)

#     df_2020['id_usuario'] = df_2020['id_usuario'].str.replace('BAEcobici', '', regex=False)
#     df_2021['id_usuario'] = df_2021['id_usuario'].str.replace('BAEcobici', '', regex=False)
#     df_2022['id_usuario'] = df_2022['id_usuario'].str.replace('BAEcobici', '', regex=False)
#     df_2023['id_usuario'] = df_2023['id_usuario'].str.replace('BAEcobici', '', regex=False)
    
#     # Combine the DataFrames
#     df_combined = pd.concat([df_2020, df_2021, df_2022, df_2023, df_2024], ignore_index=True)
    
#     # Remove duplicate rows
#     df_combined.drop_duplicates(inplace=True)
    
#     return df_combined

def limpiar_recorridos(csv_2024, csv_2023, csv_2022, csv_2021, csv_2020):
    """
    Concatenates multiple CSV files containing trip data into a single DataFrame,
    with consistent cleaning across all years and no ghost columns.
    
    Parameters:
    csv_2020 (str): Path to the CSV file for the year 2020.
    csv_2021 (str): Path to the CSV file for the year 2021.
    csv_2022 (str): Path to the CSV file for the year 2022.
    csv_2023 (str): Path to the CSV file for the year 2023.
    csv_2024 (str): Path to the CSV file for the year 2024.
    Returns:
    pd.DataFrame: A cleaned DataFrame containing the combined trip data without ghost columns.
    """
    # Configuración común para lectura de CSVs
    read_csv_params = {
        'index_col': False,  # Evita que se tome la primera columna como índice
        'skipinitialspace': True,  # Elimina espacios después de separadores
        'na_values': ['', ' '],  # Trata cadenas vacías como NaN
        'keep_default_na': False  # Evita conversión de strings como 'NA' a NaN
    }
    
    # Leer los archivos con configuración consistente
    df_2020 = pd.read_csv(csv_2020, **read_csv_params)
    df_2021 = pd.read_csv(csv_2021, **read_csv_params)
    df_2022 = pd.read_csv(csv_2022, **read_csv_params)
    df_2023 = pd.read_csv(csv_2023, **read_csv_params)
    df_2024 = pd.read_csv(csv_2024, **read_csv_params)

    # Fix 2021 dataset gender columns issue
    if 'género' in df_2021.columns and 'Género' in df_2021.columns:
            # Combine the two gender columns, taking non-null values from either
            # Handle both NaN and 'NA' string values
            mask_genero_empty = (df_2021['género'].isna()) | (df_2021['género'] == 'NA') | (df_2021['género'] == '')
            mask_genero_cap_valid = (df_2021['Género'].notna()) & (df_2021['Género'] != 'NA') & (df_2021['Género'] != '')
            
            df_2021.loc[mask_genero_empty & mask_genero_cap_valid, 'género'] = df_2021.loc[mask_genero_empty & mask_genero_cap_valid, 'Género']
            df_2021.drop(columns=['Género'], inplace=True)

    #cortar el dataset de 2024 hasta el 31 de agosto inclusive
    df_2024['fecha_origen_recorrido'] = pd.to_datetime(df_2024['fecha_origen_recorrido'], errors='coerce')
    df_2024 = df_2024[df_2024['fecha_origen_recorrido'].dt.date <= pd.to_datetime('2024-08-31').date()]

    # Eliminar columnas no deseadas (primera columna para años 2020-2023)
    dfs_to_trim = [df_2020, df_2021, df_2022, df_2023]
    for df in dfs_to_trim:
        if len(df.columns) > 0:
            df.drop(df.columns[0], axis=1, inplace=True)

    # Estandarización de nombres de columnas
    rename_dict = {
        'Género': 'género',  # Para 2022
        'genero': 'género',  # Para 2024
        'id_recorrido': 'Id_recorrido'  # Para 2024
    }
    
    df_2022.rename(columns=rename_dict, inplace=True)
    df_2024.rename(columns=rename_dict, inplace=True)

    # Limpieza de prefijos BAEcobici en múltiples columnas (2020-2023)
    cols_to_clean = ['Id_recorrido', 'id_estacion_origen', 'id_estacion_destino', 'id_usuario']
    dfs_to_clean = [df_2020, df_2021, df_2022, df_2023]
    
    for df in dfs_to_clean:
        for col in cols_to_clean:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('BAEcobici', '', regex=False)

    # Combinar todos los DataFrames
    df_combined = pd.concat([df_2020, df_2021, df_2022, df_2023, df_2024], 
                           ignore_index=True, 
                           verify_integrity=True)

    # Eliminar duplicados y posibles columnas fantasma
    df_combined = df_combined.loc[:, ~df_combined.columns.str.contains('^Unnamed')]
    df_combined = df_combined.loc[:, ~df_combined.columns.str.contains('^X$')]
    df_combined.drop_duplicates(inplace=True)

    # Limpieza final de espacios en blancos en nombres de columnas
    df_combined.columns = df_combined.columns.str.strip()
    
    return df_combined