# Ejemplo de uso de neural_network_multioutput con mejoras opcionales
import numpy as np
import pandas as pd
from models_f2 import neural_network_multioutput, plot_multiple_loss_histories, compare_model_architectures

# Ejemplo de datos (reemplazar con datos reales)
# Supongamos que tenemos 1000 muestras con 10 features y 3 targets (estaciones)
np.random.seed(42)
n_samples = 1000
n_features = 10
n_targets = 3

# Generar datos sintéticos para demostración
X_train = np.random.randn(800, n_features)
y_train = np.random.randint(0, 50, (800, n_targets))  # Arribos entre 0-50

X_val = np.random.randn(200, n_features)
y_val = np.random.randint(0, 50, (200, n_targets))  # Validación

print("🔹 Ejemplos de uso de neural_network_multioutput")
print("="*60)

# 1. Uso básico (configuración mínima)
print("\n1️⃣ Uso básico (solo Adam optimizer):")
y_pred_basic, train_losses_basic, val_losses_basic = neural_network_multioutput(
    X_train, y_train, X_val, y_val,
    verbose=True
)
print(f"   Predicciones shape: {y_pred_basic.shape}")
print(f"   Rango: [{y_pred_basic.min():.1f}, {y_pred_basic.max():.1f}]")
print(f"   Historial train losses: {len(train_losses_basic)} épocas")
print(f"   Historial val losses: {len(val_losses_basic)} épocas")

# 2. Con algunas mejoras
print("\n2️⃣ Con dropout y normalización:")
y_pred_medium, train_losses_medium, val_losses_medium = neural_network_multioutput(
    X_train, y_train, X_val, y_val,
    dropout_rate=0.2,
    normalize=True,
    verbose=True
)

# 3. Con todas las mejoras (configuración completa)
print("\n3️⃣ Configuración completa con todas las mejoras:")
y_pred_full, train_losses_full, val_losses_full = neural_network_multioutput(
    X_train, y_train, X_val, y_val,
    hidden_sizes=[256, 128, 64, 32],
    learning_rate=0.0005,
    epochs=50,  # Reducido para demo
    batch_size=64,
    dropout_rate=0.3,
    early_stopping_patience=10,
    normalize=True,
    verbose=True,
    use_adam=True,
    l2_regularization=1e-4,
    use_scheduler=True,
    plot_losses=True  # Nueva funcionalidad!
)

# 4. Comparación con SGD (sin Adam)
print("\n4️⃣ Usando SGD en lugar de Adam:")
y_pred_sgd, train_losses_sgd, val_losses_sgd = neural_network_multioutput(
    X_train, y_train, X_val, y_val,
    learning_rate=0.01,  # LR más alto para SGD
    epochs=30,
    use_adam=False,
    verbose=True
)

# 5. Solo con early stopping y gráfico
print("\n5️⃣ Solo con early stopping y gráfico de pérdidas:")
y_pred_early, train_losses_early, val_losses_early = neural_network_multioutput(
    X_train, y_train, X_val, y_val,
    early_stopping_patience=8,
    epochs=100,  # Máximo, pero se detendrá antes
    verbose=True,
    plot_losses=True
)

print("\n✅ Todos los ejemplos completados!")

# 6. NUEVA FUNCIONALIDAD: Comparar múltiples modelos manualmente
print("\n6️⃣ Comparación manual de múltiples modelos:")

# Entrenar varios modelos con diferentes configuraciones
print("Entrenando modelo pequeño...")
y_pred_small, train_small, val_small = neural_network_multioutput(
    X_train, y_train, X_val, y_val,
    hidden_sizes=[64, 32],
    epochs=30,
    verbose=False,
    plot_losses=False
)

print("Entrenando modelo mediano...")
y_pred_medium_arch, train_medium_arch, val_medium_arch = neural_network_multioutput(
    X_train, y_train, X_val, y_val,
    hidden_sizes=[128, 64, 32],
    epochs=30,
    verbose=False,
    plot_losses=False
)

print("Entrenando modelo grande...")
y_pred_large, train_large, val_large = neural_network_multioutput(
    X_train, y_train, X_val, y_val,
    hidden_sizes=[256, 128, 64],
    epochs=30,
    verbose=False,
    plot_losses=False
)

# Comparar en un solo gráfico
print("\n📊 Comparación en un solo gráfico:")
plot_multiple_loss_histories(
    [(train_small, val_small), (train_medium_arch, val_medium_arch), (train_large, val_large)],
    ['Pequeño (64-32)', 'Mediano (128-64-32)', 'Grande (256-128-64)'],
    subplot_layout='together'
)

# Comparar en subplots separados
print("\n📊 Comparación en subplots separados:")
plot_multiple_loss_histories(
    [(train_small, val_small), (train_medium_arch, val_medium_arch), (train_large, val_large)],
    ['Pequeño (64-32)', 'Mediano (128-64-32)', 'Grande (256-128-64)'],
    subplot_layout='separate'
)

# 7. NUEVA FUNCIONALIDAD: Comparación automática de arquitecturas
print("\n7️⃣ Comparación automática de arquitecturas:")

architectures = [
    [32, 16],
    [64, 32],
    [128, 64],
    [128, 64, 32],
    [256, 128, 64]
]

results = compare_model_architectures(
    X_train, y_train, X_val, y_val,
    architectures,
    epochs=25,  # Reducido para demo
    learning_rate=0.001,
    dropout_rate=0.2,
    early_stopping_patience=8,
    normalize=True,
    verbose=False,
    plot_losses=False
)

print("\n📈 Resultados de comparación:")
for model_name, result in results.items():
    arch = result['architecture']
    print(f"{model_name} ({'-'.join(map(str, arch))}): Val Loss = {result['final_val_loss']:.6f}")

print("\n✅ Todos los ejemplos completados!")
print("\nConfiguración recomendada para datos reales:")
print("""
# Para datasets grandes (>10k muestras)
y_pred, train_losses, val_losses = neural_network_multioutput(
    X_train, y_train, X_val, y_val,
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

# Para datasets pequeños (<1k muestras)  
y_pred, train_losses, val_losses = neural_network_multioutput(
    X_train, y_train, X_val, y_val,
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

# Comparar múltiples arquitecturas automáticamente
architectures = [[64, 32], [128, 64], [256, 128, 64]]
results = compare_model_architectures(
    X_train, y_train, X_val, y_val,
    architectures,
    epochs=100,
    dropout_rate=0.2,
    normalize=True,
    verbose=False
)
""")
