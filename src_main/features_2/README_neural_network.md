# 🧠 Red Neuronal Multioutput - Guía de Uso

Este módulo proporciona una función completa para entrenar redes neuronales multioutput en PyTorch con múltiples mejoras opcionales.

## 🚀 Función Principal

### `neural_network_multioutput()`

Red neuronal multioutput con arquitectura personalizable y mejoras opcionales.

**Retorna**: `(y_pred, train_losses, val_losses)` - predicciones y historiales de pérdidas

### `plot_multiple_loss_histories()`

Grafica múltiples historiales de training y validation loss para comparar modelos.

### `compare_model_architectures()`

Función de conveniencia para entrenar y comparar múltiples arquitecturas automáticamente.

#### ⚙️ Parámetros

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `x_train` | array | - | Features de entrenamiento |
| `y_train` | array | - | Targets de entrenamiento (multioutput) |
| `x_val` | array | - | Features de validación |
| `y_val` | array | - | Targets de validación (multioutput) |
| `hidden_sizes` | list | `[128, 64, 32]` | Tamaños de capas ocultas |
| `learning_rate` | float | `0.001` | Tasa de aprendizaje |
| `epochs` | int | `100` | Número máximo de épocas |
| `batch_size` | int | `32` | Tamaño del batch |
| `dropout_rate` | float/None | `None` | Tasa de dropout (ej: 0.2) |
| `early_stopping_patience` | int/None | `None` | Paciencia para early stopping (ej: 10) |
| `normalize` | bool | `False` | Si normalizar datos |
| `verbose` | bool | `False` | Si mostrar progreso |
| `use_adam` | bool | `True` | Usar Adam (True) o SGD (False) |
| `l2_regularization` | float/None | `None` | Weight decay L2 (ej: 1e-4) |
| `use_scheduler` | bool | `False` | Usar ReduceLROnPlateau |
| `plot_losses` | bool | `False` | Generar gráfico training vs validation loss |

## 📊 Ejemplos de Uso

### 1. Configuración Básica (Mínima)
```python
# Retorna predicciones y historiales
y_pred, train_losses, val_losses = neural_network_multioutput(x_train, y_train, x_val, y_val)
```

### 2. Con Normalización y Dropout
```python
y_pred, train_losses, val_losses = neural_network_multioutput(
    x_train, y_train, x_val, y_val,
    dropout_rate=0.2,
    normalize=True,
    verbose=True
)
```

### 3. Configuración Completa (Todas las Mejoras)
```python
y_pred, train_losses, val_losses = neural_network_multioutput(
    x_train, y_train, x_val, y_val,
    hidden_sizes=[256, 128, 64, 32],
    learning_rate=0.0005,
    epochs=200,
    batch_size=64,
    dropout_rate=0.3,
    early_stopping_patience=15,
    normalize=True,
    verbose=True,
    use_adam=True,
    l2_regularization=1e-4,
    use_scheduler=True,
    plot_losses=True  # Genera gráfico de pérdidas
)
```

### 4. Comparar Múltiples Modelos Manualmente
```python
# Entrenar varios modelos
y_pred1, train1, val1 = neural_network_multioutput(x_train, y_train, x_val, y_val, hidden_sizes=[64, 32])
y_pred2, train2, val2 = neural_network_multioutput(x_train, y_train, x_val, y_val, hidden_sizes=[128, 64])
y_pred3, train3, val3 = neural_network_multioutput(x_train, y_train, x_val, y_val, hidden_sizes=[256, 128])

# Comparar en un solo gráfico
plot_multiple_loss_histories(
    [(train1, val1), (train2, val2), (train3, val3)],
    ['Modelo Pequeño', 'Modelo Mediano', 'Modelo Grande'],
    subplot_layout='together'
)

# Comparar en subplots separados
plot_multiple_loss_histories(
    [(train1, val1), (train2, val2), (train3, val3)],
    ['Modelo Pequeño', 'Modelo Mediano', 'Modelo Grande'],
    subplot_layout='separate'
)
```

### 5. Comparación Automática de Arquitecturas
```python
# Definir arquitecturas a comparar
architectures = [
    [64, 32],
    [128, 64, 32],
    [256, 128, 64],
    [512, 256, 128]
]

# Entrenar y comparar automáticamente
results = compare_model_architectures(
    x_train, y_train, x_val, y_val,
    architectures,
    epochs=100,
    dropout_rate=0.2,
    normalize=True,
    verbose=False
)
```

### 4. Usando SGD en lugar de Adam
```python
y_pred = neural_network_multioutput(
    x_train, y_train, x_val,
    learning_rate=0.01,  # LR más alto para SGD
    use_adam=False,
    verbose=True
)
```

## 🎯 Configuraciones Recomendadas

### Para Datasets Grandes (>10k muestras)
```python
y_pred, train_losses, val_losses = neural_network_multioutput(
    x_train, y_train, x_val, y_val,
    hidden_sizes=[512, 256, 128, 64],
    learning_rate=0.001,
    epochs=200,
    batch_size=128,
    dropout_rate=0.3,
    early_stopping_patience=15,
    normalize=True,
    verbose=True,
    use_adam=True,
    l2_regularization=1e-4,
    use_scheduler=True,
    plot_losses=True
)
```

### Para Datasets Pequeños (<1k muestras)
```python
y_pred, train_losses, val_losses = neural_network_multioutput(
    x_train, y_train, x_val, y_val,
    hidden_sizes=[128, 64],
    learning_rate=0.001,
    epochs=100,
    batch_size=32,
    dropout_rate=0.2,
    early_stopping_patience=10,
    normalize=True,
    verbose=True,
    l2_regularization=1e-5,
    plot_losses=True
)
```

### Comparación Rápida de Arquitecturas
```python
architectures = [[64, 32], [128, 64], [256, 128, 64]]
results = compare_model_architectures(
    x_train, y_train, x_val, y_val,
    architectures,
    epochs=100,
    dropout_rate=0.2,
    normalize=True
)
```

## 🛠️ Mejoras Implementadas

### ✅ Optimizadores
- **Adam** (default): Adaptive learning rate, funciona bien en la mayoría de casos
- **SGD**: Gradient descent clásico, a veces más estable

### ✅ Regularización
- **Dropout**: Previene overfitting desactivando neuronas aleatoriamente
- **L2 Regularization**: Weight decay, penaliza pesos grandes

### ✅ Entrenamiento Inteligente
- **Early Stopping**: Para automáticamente si no hay mejora
- **Learning Rate Scheduler**: Reduce LR cuando se estanca
- **Normalización**: StandardScaler para features y targets
- **Validation Loss**: Monitoreo continuo durante entrenamiento

### ✅ Visualización y Comparación
- **Gráfico de pérdidas**: Training vs Validation loss por época
- **Comparación múltiple**: Gráficos juntos o separados
- **Comparación automática**: Entrenar y comparar arquitecturas
- **Marcadores especiales**: Early stopping, mejor modelo
- **Estadísticas**: Métricas finales y rankings automáticos

### ✅ Arquitectura Flexible
- **Capas ocultas personalizables**: Ajusta `hidden_sizes`
- **Batch size variable**: Para optimizar memoria y velocidad
- **Device selection**: CPU o CUDA

## 🔧 Tips de Uso

1. **Siempre empieza simple**: Usa configuración básica primero
2. **Normaliza tus datos**: Especialmente importante para redes neuronales
3. **Usa early stopping**: Evita overfitting automáticamente
4. **GPU automática**: La función usa GPU si está disponible, sino CPU
5. **Experimenta con arquitectura**: Más capas/neuronas para datos complejos
6. **Ajusta learning rate**: 0.001 es buen inicio, 0.01 para SGD

## 🚨 Notas Importantes

- **Nuevo retorno**: Ahora retorna `(y_pred, train_losses, val_losses)` en lugar de solo `y_pred`
- **Breaking change**: Actualizar código existente para manejar múltiples valores de retorno
- **Device automático**: Usa CUDA si está disponible, sino CPU
- **Validation Loss**: Ahora requiere `y_val` para monitoreo durante entrenamiento
- **Early Stopping**: Basado en validation loss (más robusto que training loss)
- **Scheduler**: También usa validation loss para ajustar learning rate
- **Gráficos**: Visualización automática de curvas de aprendizaje
- **Comparación**: Nuevas funciones para comparar múltiples modelos fácilmente
- Las predicciones se redondean y limitan a valores no negativos
- El modelo guarda automáticamente el mejor estado con early stopping
- Todas las mejoras son **opcionales** - puedes usar solo las que necesites
- Compatible con datos multioutput (múltiples targets simultáneos)

## 📋 Dependencias Requeridas

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
```

## 🎨 Clase MultiOutputMLP

La función utiliza internamente la clase `MultiOutputMLP` que implementa:
- Capas fully connected con ReLU
- Dropout opcional entre capas
- Arquitectura flexible

Ejemplo de uso directo de la clase:
```python
model = MultiOutputMLP(
    input_size=10, 
    hidden_sizes=[128, 64, 32], 
    output_size=3, 
    dropout_rate=0.2
)
```
