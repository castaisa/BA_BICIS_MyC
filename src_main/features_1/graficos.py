import pandas as pd

import matplotlib.pyplot as plt

def graficar_columna(dataset, nombre_columna):
    """
    Función que grafica una columna del dataset vs el número de fila y crea un histograma.
    
    Args:
        dataset: DataFrame de pandas
        nombre_columna: string con el nombre de la columna a graficar
    """
    # Verificar que la columna existe
    if nombre_columna not in dataset.columns:
        print(f"Error: La columna '{nombre_columna}' no existe en el dataset")
        return
    
    # Crear subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico de línea: número de fila vs valor de la columna
    ax1.plot(range(len(dataset)), dataset[nombre_columna])
    ax1.set_xlabel('Número de fila')
    ax1.set_ylabel(nombre_columna)
    ax1.set_title(f'{nombre_columna} vs Número de fila')
    ax1.set_xlim(27000, 27072)
    ax1.grid(True)
    
    # Histograma de la columna
    ax2.hist(dataset[nombre_columna].dropna(), bins=30, alpha=0.7)
    ax2.set_xlabel(nombre_columna)
    ax2.set_ylabel('Frecuencia')
    ax2.set_title(f'Histograma de {nombre_columna}')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def graficar_pred_vs_real(y_val, y_pred, estacion):
    """
    Función que grafica las predicciones vs los valores reales.
    
    Args:
        y_val: Valores reales del conjunto de validación
        y_pred: Predicciones del modelo
        estacion: Número de la estación para el título del gráfico
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.xlim(0, 25)
    plt.ylim(0, 25)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.title(f'Predicciones vs Valores Reales - Estación {estacion}')
    plt.grid(True)
    plt.show()

    #Imprimo estadisticas

    print(f"La media de los valores reales es: {y_val.mean()}")
    print(f"La media de las predicciones es: {y_pred.mean()}")
    