import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('default')
sns.set_palette("husl")

def prepare_data(df):
    """
    Prepara el dataset con las columnas de tiempo necesarias
    """
    df = df.copy()
    
    # Convertir fechas
    df['fecha_destino_recorrido'] = pd.to_datetime(df['fecha_destino_recorrido'])
    
    # Extraer componentes temporales
    df['hora'] = df['fecha_destino_recorrido'].dt.hour
    df['dia_semana'] = df['fecha_destino_recorrido'].dt.dayofweek  # 0=Lunes, 6=Domingo
    df['mes'] = df['fecha_destino_recorrido'].dt.month
    df['fecha'] = df['fecha_destino_recorrido'].dt.date
    df['año'] = df['fecha_destino_recorrido'].dt.year
    
    # Crear etiquetas más legibles
    df['dia_semana_nombre'] = df['fecha_destino_recorrido'].dt.day_name()
    df['mes_nombre'] = df['fecha_destino_recorrido'].dt.month_name()
    df['es_fin_semana'] = df['dia_semana'].isin([5, 6])  # Sábado y Domingo
    
    return df

def heatmap_llegadas_por_hora(df, top_n_estaciones=20, figsize=(15, 10)):
    """
    Crea un heatmap de llegadas por estación vs hora del día
    """
    # Preparar datos
    df_prep = prepare_data(df)
    
    # Filtrar estaciones con más tráfico
    top_estaciones = df_prep['nombre_estacion_destino'].value_counts().head(top_n_estaciones).index
    df_filtered = df_prep[df_prep['nombre_estacion_destino'].isin(top_estaciones)]
    
    # Crear matriz de llegadas por hora
    llegadas_por_hora = df_filtered.groupby(['nombre_estacion_destino', 'hora']).size().unstack(fill_value=0)
    
    # Crear el heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(llegadas_por_hora, 
                cmap='YlOrRd', 
                annot=False, 
                fmt='d',
                cbar_kws={'label': 'Número de llegadas'})
    
    plt.title(f'Heatmap de Llegadas por Hora del Día\n(Top {top_n_estaciones} estaciones con más tráfico)', 
              fontsize=16, pad=20)
    plt.xlabel('Hora del día', fontsize=12)
    plt.ylabel('Estación de destino', fontsize=12)
    plt.xticks(range(24), [f'{i:02d}:00' for i in range(24)], rotation=45)
    plt.tight_layout()
    
    return llegadas_por_hora

def patrones_semanales(df, figsize=(15, 8)):
    """
    Analiza patrones entre días laborables y fines de semana
    """
    df_prep = prepare_data(df)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Llegadas por día de la semana
    llegadas_por_dia = df_prep.groupby('dia_semana_nombre').size()
    dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    llegadas_por_dia = llegadas_por_dia.reindex(dias_orden)
    
    axes[0,0].bar(range(7), llegadas_por_dia.values, color='skyblue', alpha=0.7)
    axes[0,0].set_title('Llegadas por Día de la Semana')
    axes[0,0].set_xlabel('Día')
    axes[0,0].set_ylabel('Número de llegadas')
    axes[0,0].set_xticks(range(7))
    axes[0,0].set_xticklabels(['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'], rotation=45)
    
    # 2. Patrones horarios: Laborables vs Fin de semana
    df_laborables = df_prep[~df_prep['es_fin_semana']]
    df_finde = df_prep[df_prep['es_fin_semana']]
    
    llegadas_lab = df_laborables.groupby('hora').size()
    llegadas_finde = df_finde.groupby('hora').size()
    
    axes[0,1].plot(llegadas_lab.index, llegadas_lab.values, label='Días laborables', marker='o', linewidth=2)
    axes[0,1].plot(llegadas_finde.index, llegadas_finde.values, label='Fin de semana', marker='s', linewidth=2)
    axes[0,1].set_title('Patrones Horarios: Laborables vs Fin de Semana')
    axes[0,1].set_xlabel('Hora del día')
    axes[0,1].set_ylabel('Número de llegadas')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Box plot por tipo de día
    df_plot = df_prep.copy()
    df_plot['tipo_dia'] = df_plot['es_fin_semana'].map({False: 'Laborable', True: 'Fin de semana'})
    
    llegadas_por_hora_dia = df_plot.groupby(['fecha', 'hora', 'tipo_dia']).size().reset_index(name='llegadas')
    sns.boxplot(data=llegadas_por_hora_dia, x='hora', y='llegadas', hue='tipo_dia', ax=axes[1,0])
    axes[1,0].set_title('Distribución de Llegadas por Hora')
    axes[1,0].set_xlabel('Hora del día')
    axes[1,0].set_ylabel('Llegadas por hora')
    
    # 4. Promedio de llegadas por estación
    avg_laborables = df_laborables.groupby('nombre_estacion_destino').size().mean()
    avg_finde = df_finde.groupby('nombre_estacion_destino').size().mean()
    
    categorias = ['Días laborables', 'Fin de semana']
    promedios = [avg_laborables, avg_finde]
    
    axes[1,1].bar(categorias, promedios, color=['coral', 'lightblue'], alpha=0.7)
    axes[1,1].set_title('Promedio de Llegadas por Estación')
    axes[1,1].set_ylabel('Promedio de llegadas')
    
    plt.tight_layout()
    
    # Estadísticas adicionales
    print("=== ANÁLISIS SEMANAL ===")
    print(f"Total viajes laborables: {len(df_laborables):,}")
    print(f"Total viajes fin de semana: {len(df_finde):,}")
    print(f"Ratio laborables/finde: {len(df_laborables)/len(df_finde):.2f}")
    print(f"Promedio diario laborables: {len(df_laborables)/5:.0f}")
    print(f"Promedio diario fin de semana: {len(df_finde)/2:.0f}")

def estacionalidad(df, figsize=(15, 10)):
    """
    Analiza variaciones por mes y estación del año usando promedios diarios
    """
    df_prep = prepare_data(df)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Llegadas promedio por mes (para evitar sesgo por meses incompletos)
    llegadas_diarias_por_mes = df_prep.groupby(['mes', 'fecha']).size().reset_index(name='llegadas_dia')
    llegadas_promedio_mes = llegadas_diarias_por_mes.groupby('mes')['llegadas_dia'].mean()
    meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dic']
    
    axes[0,0].bar(llegadas_promedio_mes.index, llegadas_promedio_mes.values, color='lightgreen', alpha=0.7)
    axes[0,0].set_title('Promedio de Llegadas Diarias por Mes')
    axes[0,0].set_xlabel('Mes')
    axes[0,0].set_ylabel('Promedio de llegadas por día')
    axes[0,0].set_xticks(range(1, 13))
    axes[0,0].set_xticklabels(meses_nombres, rotation=45)
    
    # 2. Heatmap mes vs hora (promedio por hora)
    llegadas_mes_hora_dia = df_prep.groupby(['mes', 'hora', 'fecha']).size().reset_index(name='llegadas')
    llegadas_mes_hora_promedio = llegadas_mes_hora_dia.groupby(['mes', 'hora'])['llegadas'].mean().unstack(fill_value=0)
    sns.heatmap(llegadas_mes_hora_promedio, cmap='Blues', ax=axes[0,1], cbar_kws={'label': 'Promedio llegadas/hora'})
    axes[0,1].set_title('Promedio de Patrones Horarios por Mes')
    axes[0,1].set_xlabel('Hora del día')
    axes[0,1].set_ylabel('Mes')
    axes[0,1].set_yticklabels(meses_nombres, rotation=0)
    
    # 3. Estaciones del año (hemisferio sur) - usando promedios diarios
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Verano'
        elif month in [3, 4, 5]:
            return 'Otoño'
        elif month in [6, 7, 8]:
            return 'Invierno'
        else:
            return 'Primavera'
    
    df_prep['estacion'] = df_prep['mes'].apply(get_season)
    llegadas_diarias_estacion = df_prep.groupby(['estacion', 'fecha']).size().reset_index(name='llegadas_dia')
    llegadas_promedio_estacion = llegadas_diarias_estacion.groupby('estacion')['llegadas_dia'].mean()
    
    colores_estaciones = {'Verano': 'orange', 'Otoño': 'brown', 'Invierno': 'lightblue', 'Primavera': 'lightgreen'}
    colors = [colores_estaciones[est] for est in llegadas_promedio_estacion.index]
    
    axes[1,0].bar(llegadas_promedio_estacion.index, llegadas_promedio_estacion.values, color=colors, alpha=0.7)
    axes[1,0].set_title('Promedio de Llegadas Diarias por Estación del Año')
    axes[1,0].set_ylabel('Promedio de llegadas por día')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Variación semanal por estación del año - usando promedios diarios
    llegadas_semana_estacion = df_prep.groupby(['estacion', 'dia_semana', 'fecha']).size().reset_index(name='llegadas_dia')
    patron_semanal_promedio = llegadas_semana_estacion.groupby(['estacion', 'dia_semana'])['llegadas_dia'].mean().unstack(fill_value=0)
    dias_nombres = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    
    for estacion in patron_semanal_promedio.index:
        axes[1,1].plot(range(7), patron_semanal_promedio.loc[estacion], 
                      marker='o', label=estacion, linewidth=2)
    
    axes[1,1].set_title('Promedio de Patrones Semanales por Estación')
    axes[1,1].set_xlabel('Día de la semana')
    axes[1,1].set_ylabel('Promedio de llegadas por día')
    axes[1,1].set_xticks(range(7))
    axes[1,1].set_xticklabels(dias_nombres)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print("=== ANÁLISIS ESTACIONAL (PROMEDIO DIARIO) ===")
    for estacion, promedio in llegadas_promedio_estacion.items():
        print(f"{estacion}: {promedio:.1f} llegadas/día en promedio")
    
    print("\n=== ANÁLISIS MENSUAL (PROMEDIO DIARIO) ===")
    for mes, promedio in llegadas_promedio_mes.items():
        mes_nombre = meses_nombres[mes-1]
        print(f"{mes_nombre}: {promedio:.1f} llegadas/día en promedio")

def tendencias_temporales(df, figsize=(15, 10)):
    """
    Analiza la evolución temporal del uso del sistema
    """
    df_prep = prepare_data(df)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Tendencia diaria
    llegadas_diarias = df_prep.groupby('fecha').size().reset_index(name='llegadas')
    llegadas_diarias['fecha'] = pd.to_datetime(llegadas_diarias['fecha'])
    llegadas_diarias = llegadas_diarias.sort_values('fecha')
    
    axes[0,0].plot(llegadas_diarias['fecha'], llegadas_diarias['llegadas'], alpha=0.7, linewidth=1)
    axes[0,0].plot(llegadas_diarias['fecha'], llegadas_diarias['llegadas'].rolling(7).mean(), 
                   color='red', linewidth=2, label='Media móvil 7 días')
    axes[0,0].set_title('Tendencia Diaria de Llegadas')
    axes[0,0].set_xlabel('Fecha')
    axes[0,0].set_ylabel('Llegadas por día')
    axes[0,0].legend()
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Tendencia mensual
    df_prep['año_mes'] = df_prep['fecha_destino_recorrido'].dt.to_period('M')
    llegadas_mensuales = df_prep.groupby('año_mes').size()
    
    axes[0,1].plot(range(len(llegadas_mensuales)), llegadas_mensuales.values, marker='o', linewidth=2)
    axes[0,1].set_title('Tendencia Mensual')
    axes[0,1].set_xlabel('Período')
    axes[0,1].set_ylabel('Llegadas por mes')
    axes[0,1].set_xticks(range(0, len(llegadas_mensuales), max(1, len(llegadas_mensuales)//6)))
    axes[0,1].set_xticklabels([str(llegadas_mensuales.index[i]) for i in range(0, len(llegadas_mensuales), max(1, len(llegadas_mensuales)//6))], rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Crecimiento de usuarios únicos
    usuarios_por_mes = df_prep.groupby('año_mes')['id_usuario'].nunique()
    
    axes[1,0].bar(range(len(usuarios_por_mes)), usuarios_por_mes.values, alpha=0.7, color='lightcoral')
    axes[1,0].set_title('Usuarios Únicos por Mes')
    axes[1,0].set_xlabel('Período')
    axes[1,0].set_ylabel('Usuarios únicos')
    axes[1,0].set_xticks(range(0, len(usuarios_por_mes), max(1, len(usuarios_por_mes)//6)))
    axes[1,0].set_xticklabels([str(usuarios_por_mes.index[i]) for i in range(0, len(usuarios_por_mes), max(1, len(usuarios_por_mes)//6))], rotation=45)
      # 4. Distribución de duraciones por período
    # Limpiar la columna de duración: remover puntos (separadores de miles) y cambiar comas por puntos decimales
    df_prep['duracion_limpia'] = df_prep['duracion_recorrido'].astype(str)
    df_prep['duracion_limpia'] = df_prep['duracion_limpia'].str.replace('.', '', regex=False)  # Remover separadores de miles
    df_prep['duracion_limpia'] = df_prep['duracion_limpia'].str.replace(',', '.', regex=False)  # Cambiar coma decimal por punto
    
    # Convertir a float y luego a minutos
    try:
        df_prep['duracion_minutos'] = pd.to_numeric(df_prep['duracion_limpia'], errors='coerce') / 60
        # Filtrar outliers (menos de 5 horas) y valores nulos
        df_duracion = df_prep[(df_prep['duracion_minutos'] < 300) & (df_prep['duracion_minutos'].notna())]
        
        if len(df_duracion) > 0:
            duracion_por_mes = df_duracion.groupby('año_mes')['duracion_minutos'].median()
            
            axes[1,1].plot(range(len(duracion_por_mes)), duracion_por_mes.values, marker='s', linewidth=2, color='green')
            axes[1,1].set_title('Mediana de Duración de Viajes')
            axes[1,1].set_xlabel('Período')
            axes[1,1].set_ylabel('Duración mediana (minutos)')
            axes[1,1].set_xticks(range(0, len(duracion_por_mes), max(1, len(duracion_por_mes)//6)))
            axes[1,1].set_xticklabels([str(duracion_por_mes.index[i]) for i in range(0, len(duracion_por_mes), max(1, len(duracion_por_mes)//6))], rotation=45)
        else:
            # Si no hay datos válidos, mostrar mensaje
            axes[1,1].text(0.5, 0.5, 'No hay datos válidos\nde duración', 
                          horizontalalignment='center', verticalalignment='center', 
                          transform=axes[1,1].transAxes, fontsize=12)
            axes[1,1].set_title('Mediana de Duración de Viajes')
            
    except Exception as e:
        print(f"Error procesando duraciones: {e}")
        # Gráfico vacío con mensaje de error
        axes[1,1].text(0.5, 0.5, f'Error procesando\ndatos de duración:\n{str(e)[:50]}...', 
                      horizontalalignment='center', verticalalignment='center', 
                      transform=axes[1,1].transAxes, fontsize=10)
        axes[1,1].set_title('Mediana de Duración de Viajes')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Estadísticas de crecimiento
    print("=== ANÁLISIS DE TENDENCIAS ===")
    print(f"Período analizado: {llegadas_diarias['fecha'].min().date()} a {llegadas_diarias['fecha'].max().date()}")
    print(f"Promedio diario: {llegadas_diarias['llegadas'].mean():.0f} llegadas")
    print(f"Día con más llegadas: {llegadas_diarias.loc[llegadas_diarias['llegadas'].idxmax(), 'fecha'].date()} ({llegadas_diarias['llegadas'].max():,} llegadas)")
    print(f"Día con menos llegadas: {llegadas_diarias.loc[llegadas_diarias['llegadas'].idxmin(), 'fecha'].date()} ({llegadas_diarias['llegadas'].min():,} llegadas)")
    
    if len(llegadas_mensuales) > 1:
        crecimiento_mensual = ((llegadas_mensuales.iloc[-1] / llegadas_mensuales.iloc[0]) ** (1/(len(llegadas_mensuales)-1)) - 1) * 100
        print(f"Tasa de crecimiento mensual promedio: {crecimiento_mensual:.2f}%")

# Función principal para ejecutar todos los análisis
def analisis_completo(df):
    """
    Ejecuta todos los análisis temporales
    """
    print("Iniciando análisis temporal completo...")
    print("="*50)
    
    # 1. Heatmap horario
    print("\n1. Generando heatmap de patrones horarios...")
    heatmap_data = heatmap_llegadas_por_hora(df)
    plt.show()
    
    # 2. Patrones semanales
    print("\n2. Analizando patrones semanales...")
    patrones_semanales(df)
    plt.show()
    
    # 3. Estacionalidad
    print("\n3. Analizando estacionalidad...")
    estacionalidad(df)
    plt.show()
    
    # 4. Tendencias temporales
    print("\n4. Analizando tendencias temporales...")
    tendencias_temporales(df)
    plt.show()
    
    print("\n" + "="*50)
    print("Análisis temporal completado!")


