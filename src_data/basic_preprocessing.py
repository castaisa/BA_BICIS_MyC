import numpy as np
import pandas as pd

def unite_usuarios(csv_2024, csv_2023):
    """
    Unites two CSV files containing user data from different years into a single DataFrame.
    
    Parameters:
    csv_2024 (str): Path to the CSV file for the year 2024.
    csv_2023 (str): Path to the CSV file for the year 2023.
    
    Returns:
    pd.DataFrame: A DataFrame containing the combined user data.
    """
    df_2024 = pd.read_csv(csv_2024)
    df_2023 = pd.read_csv(csv_2023)

    # eliminar la última columna de 2023
    if df_2023.shape[1] > 0:
        df_2023 = df_2023.iloc[:, :-1]
    
    
    # Combine the two DataFrames
    df_combined = pd.concat([df_2024, df_2023], ignore_index=True)
    
    # Remove duplicate rows
    df_combined.drop_duplicates(inplace=True)
    
    return df_combined
    