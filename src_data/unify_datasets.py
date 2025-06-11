import pandas as pd

def unify_datasets(df_recorridos, df_usuarios):
    """
    Une los datasets de recorridos y usuarios basándose en el id_usuario.
    
    Parameters:
    df_recorridos (pd.DataFrame): DataFrame con los datos de recorridos
    df_usuarios (pd.DataFrame): DataFrame con los datos de usuarios
    
    Returns:
    pd.DataFrame: DataFrame unificado con información de recorridos y usuarios
    """
    
    # Verificar que las columnas necesarias existan
    required_cols_recorridos = ['id_usuario']
    required_cols_usuarios = ['id_usuario', 'genero_usuario', 'edad_usuario', 'fecha_alta', 'hora_alta']
    
    for col in required_cols_recorridos:
        if col not in df_recorridos.columns:
            raise ValueError(f"Columna '{col}' no encontrada en dataset de recorridos")
    
    for col in required_cols_usuarios:
        if col not in df_usuarios.columns:
            raise ValueError(f"Columna '{col}' no encontrada en dataset de usuarios")
    
    # Convertir id_usuario a tipo consistente (float) en ambos datasets
    df_recorridos['id_usuario'] = pd.to_numeric(df_recorridos['id_usuario'], errors='coerce')
    df_usuarios['id_usuario'] = pd.to_numeric(df_usuarios['id_usuario'], errors='coerce')
    
    # Realizar el merge (LEFT JOIN) para mantener todos los recorridos
    df_unified = df_recorridos.merge(
        df_usuarios[['id_usuario', 'genero_usuario', 'edad_usuario', 'fecha_alta', 'hora_alta']], 
        on='id_usuario', 
        how='left'
    )
    
    # Reportar estadísticas del merge
    total_recorridos = len(df_recorridos)
    recorridos_con_usuario = df_unified['genero_usuario'].notna().sum()
    recorridos_sin_usuario = total_recorridos - recorridos_con_usuario
    
    print(f"Estadísticas del merge:")
    print(f"Total recorridos: {total_recorridos}")
    print(f"Recorridos con información de usuario: {recorridos_con_usuario}")
    print(f"Recorridos sin información de usuario: {recorridos_sin_usuario}")
    print(f"Porcentaje de match: {(recorridos_con_usuario/total_recorridos)*100:.2f}%")
    
    return df_unified