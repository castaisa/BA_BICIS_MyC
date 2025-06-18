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

def cut_recorridos(df, fecha_limite):
    """
    Corta el DataFrame de recorridos hasta una fecha límite específica.
    
    Parameters:
    df (pd.DataFrame): DataFrame de recorridos
    fecha_limite (str or datetime): Fecha límite hasta la cual mantener los recorridos
    
    Returns:
    tuple: (DataFrame filtrado, índices de filas eliminadas)
    """
    # Convertir columna fecha_origen_recorrido a datetime
    df['fecha_origen_recorrido'] = pd.to_datetime(df['fecha_origen_recorrido'], 
                                                 format='%Y-%m-%d %H:%M:%S', 
                                                 errors='coerce')
    
    # Debug: Verificar el rango de fechas antes de cortar
    print("Rango de fechas original:")
    print(df['fecha_origen_recorrido'].min(), "a", df['fecha_origen_recorrido'].max())

    # Convertir fecha_limite
    if isinstance(fecha_limite, str):
        if len(fecha_limite) == 10:  # Solo fecha sin hora
            fecha_limite = pd.to_datetime(fecha_limite + ' 23:59:59', format='%Y-%m-%d %H:%M:%S')
        else:
            fecha_limite = pd.to_datetime(fecha_limite, format='%Y-%m-%d %H:%M:%S')
    
    print(f"Fecha límite aplicada: {fecha_limite}")

    # Filtrar el DataFrame
    mask = df['fecha_origen_recorrido'] <= fecha_limite
    df_filtered = df[mask].copy()
    eliminated_rows = df[~mask].index
    
    # Debug: Verificar el rango de fechas después de cortar
    print("Rango de fechas después de cortar:")
    print(df_filtered['fecha_origen_recorrido'].min(), "a", df_filtered['fecha_origen_recorrido'].max())
    print(f"Filas eliminadas: {len(eliminated_rows)}")

    return df_filtered, eliminated_rows



def limpiar_recorridos(csv_2024, csv_2023, csv_2022, csv_2021, csv_2020=None):
    """
    Concatenates multiple CSV files containing trip data into a single DataFrame,
    with consistent cleaning across all years and no ghost columns.
    
    Parameters:
    csv_2020 (str or None): Path to the CSV file for the year 2020, or None to skip.
    csv_2021 (str): Path to the CSV file for the year 2021.
    csv_2022 (str): Path to the CSV file for the year 2022.
    csv_2023 (str): Path to the CSV file for the year 2023.
    csv_2024 (str): Path to the CSV file for the year 2024.
    Returns:
    tuple: (DataFrame limpio, índices eliminados)
    """
    # Configuración común para lectura de CSVs
    read_csv_params = {
        'index_col': False,  # Evita que se tome la primera columna como índice
        'skipinitialspace': True,  # Elimina espacios después de separadores
        'na_values': ['', ' '],  # Trata cadenas vacías como NaN
        'keep_default_na': False  # Evita conversión de strings como 'NA' a NaN
    }
    
    # Leer los archivos con configuración consistente
    # Solo leer 2020 si se proporciona un path
    if csv_2020 is not None:
        df_2020 = pd.read_csv(csv_2020, **read_csv_params)
    else:
        df_2020 = pd.DataFrame()  # DataFrame vacío
    
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

    # Cortar el dataset de 2024 hasta el 31 de agosto inclusive
    df_2024, eliminated_2024 = cut_recorridos(df_2024, '2024-08-31')

    # Eliminar columnas no deseadas (primera columna para años 2020-2023)
    # Solo procesar 2020 si no está vacío
    dfs_to_trim = [df for df in [df_2020, df_2021, df_2022, df_2023] if not df.empty]
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
    dfs_to_clean = [df for df in [df_2020, df_2021, df_2022, df_2023] if not df.empty]
    
    for df in dfs_to_clean:
        for col in cols_to_clean:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('BAEcobici', '', regex=False)
    
    #Cambio la columna id_estacion_destino y id_estacion_origen a int
    for df in [df_2020, df_2021, df_2022, df_2023, df_2024]:
        if 'id_estacion_destino' in df.columns:
            df['id_estacion_destino'] = pd.to_numeric(df['id_estacion_destino'], errors='coerce').astype('Int64')
        if 'id_estacion_origen' in df.columns:
            df['id_estacion_origen'] = pd.to_numeric(df['id_estacion_origen'], errors='coerce').astype('Int64')

    #para todos los datasets, la columna id_usuario la pasa a int
    all_dfs = [df for df in [df_2020, df_2021, df_2022, df_2023, df_2024] if not df.empty]
    for df in all_dfs:
        if 'id_usuario' in df.columns:
            df['id_usuario'] = pd.to_numeric(df['id_usuario'], errors='coerce').astype('Int64')

    # Combinar todos los DataFrames (solo los que no están vacíos)
    df_combined = pd.concat(all_dfs, ignore_index=True, verify_integrity=True)

    # Eliminar duplicados y posibles columnas fantasma
    df_combined = df_combined.loc[:, ~df_combined.columns.str.contains('^Unnamed')]
    df_combined = df_combined.loc[:, ~df_combined.columns.str.contains('^X$')]
    df_combined.drop_duplicates(inplace=True)

    # Limpieza final de espacios en blancos en nombres de columnas
    df_combined.columns = df_combined.columns.str.strip()
    
    return df_combined, eliminated_2024

import pandas as pd
def unificar_datasets(df_recorridos, df_usuarios):
    """
    Une los datasets de recorridos y usuarios basándose en el id_usuario.
    Agrega las columnas edad_usuario, fecha_alta y hora_alta del dataset usuarios 
    al dataset recorridos usando id_usuario como clave.
    
    Parameters:
    df_recorridos (pd.DataFrame): DataFrame con los datos de recorridos
    df_usuarios (pd.DataFrame): DataFrame con los datos de usuarios
    
    Returns:
    pd.DataFrame: DataFrame unificado con información de recorridos y usuarios
    """
    
    # Verificar que las columnas necesarias existan
    required_cols_recorridos = ['id_usuario']
    required_cols_usuarios = ['id_usuario', 'edad_usuario', 'fecha_alta', 'hora_alta']
    
    # for col in required_cols_recorridos:
    #     if col not in df_recorridos.columns:
    #         raise ValueError(f"Columna '{col}' no encontrada en dataset de recorridos")
    
    for col in required_cols_usuarios:
        if col not in df_usuarios.columns:
            print(df_usuarios.head())
            raise ValueError(f"Columna '{col}' no encontrada en dataset de usuarios")
    
    # Convertir id_usuario a tipo consistente (string) en ambos datasets para evitar problemas de tipo
    df_recorridos['id_usuario'] = df_recorridos['id_usuario'].astype(str)
    df_usuarios['id_usuario'] = df_usuarios['id_usuario'].astype(str)
    #
    # Realizar el merge (LEFT JOIN) para mantener todos los recorridos
    # Solo seleccionamos las columnas que queremos agregar del dataset usuarios
    df_unified = df_recorridos.merge(
        df_usuarios[['id_usuario', 'edad_usuario', 'fecha_alta', 'hora_alta']], 
        on='id_usuario', 
        how='left'
    )
    
    # Reportar estadísticas del merge
    total_recorridos = len(df_recorridos)
    recorridos_con_usuario = df_unified['edad_usuario'].notna().sum()
    recorridos_sin_usuario = total_recorridos - recorridos_con_usuario
    
    print(f"Estadísticas del merge:")
    print(f"Total recorridos: {total_recorridos}")
    print(f"Recorridos con información de usuario: {recorridos_con_usuario}")
    print(f"Recorridos sin información de usuario: {recorridos_sin_usuario}")
    print(f"Porcentaje de match: {(recorridos_con_usuario/total_recorridos)*100:.2f}%")
    
    return df_unified


def cut_df(df, start_date=None, end_date=None):
    """
    Corta el DataFrame entre fechas específicas. Si alguna fecha es None, no aplica ese filtro.
    
    Args:
        df (pd.DataFrame): DataFrame con users y recorridos unificados
        start_date (str, optional): Fecha de inicio en formato 'YYYY-MM-DD'. Si es None, no filtra por inicio.
        end_date (str, optional): Fecha de fin en formato 'YYYY-MM-DD'. Si es None, no filtra por fin.
    
    Returns:
        pd.DataFrame: DataFrame filtrado entre las fechas especificadas
    """
    if 'fecha_destino_recorrido' not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'fecha_destino_recorrido'.")
    
    df_copy = df.copy()
    df_copy['fecha_destino_recorrido'] = pd.to_datetime(df_copy['fecha_destino_recorrido'])
    
    # Aplicar filtro de fecha inicial si se proporciona
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df_copy = df_copy[df_copy['fecha_destino_recorrido'] >= start_date]
    
    # Aplicar filtro de fecha final si se proporciona
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df_copy = df_copy[df_copy['fecha_destino_recorrido'] <= end_date]
    
    return df_copy.reset_index(drop=True)