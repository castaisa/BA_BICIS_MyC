import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def linear_regression(x_train, y_train, x_val):
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_pred = np.round(np.maximum(y_pred, 0))
    return y_pred

def random_forest_regressor(x_train, y_train, x_val, n_estimators=100, random_state=42):
    """
    Random Forest Regressor
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_pred = np.round(np.maximum(y_pred, 0))
    return y_pred

def gradient_boosting_regressor(x_train, y_train, x_val, n_estimators=100, learning_rate=0.1, random_state=42):
    """
    Gradient Boosting Regressor
    """
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_pred = np.round(np.maximum(y_pred, 0))
    return y_pred

def decision_tree_regressor(x_train, y_train, x_val, random_state=42):
    """
    Decision Tree Regressor
    """
    model = DecisionTreeRegressor(random_state=random_state)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_pred = np.round(np.maximum(y_pred, 0))
    return y_pred

def knn_regressor(x_train, y_train, x_val, n_neighbors=5):
    """
    K-Nearest Neighbors Regressor
    """
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_pred = np.round(np.maximum(y_pred, 0))
    return y_pred

def svr_regressor(x_train, y_train, x_val, kernel='rbf', C=1.0):
    """
    Support Vector Regressor
    """
    model = SVR(kernel=kernel, C=C)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_pred = np.round(np.maximum(y_pred, 0))
    return y_pred

def ridge_regression(x_train, y_train, x_val, alpha=1.0):
    """
    Ridge Regression (L2 regularization)
    """
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_pred = np.round(np.maximum(y_pred, 0))
    return y_pred

def lasso_regression(x_train, y_train, x_val, alpha=1.0):
    """
    Lasso Regression (L1 regularization)
    """
    model = Lasso(alpha=alpha)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_pred = np.round(np.maximum(y_pred, 0))
    return y_pred

def ada_boost_regressor(x_train, y_train, x_val, n_estimators=50, learning_rate=1.0, random_state=42):
    """
    AdaBoost Regressor
    """
    model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_pred = np.round(np.maximum(y_pred, 0))
    return y_pred

def svr_regressor_multioutput(x_train, y_train, x_val, kernel='rbf', C=1.0):
    """
    Support Vector Regressor para múltiples outputs usando MultiOutputRegressor
    """
    base_model = SVR(kernel=kernel, C=C)
    model = MultiOutputRegressor(base_model)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_pred = np.round(np.maximum(y_pred, 0))
    return y_pred

def gradient_boosting_regressor_multioutput(x_train, y_train, x_val, n_estimators=100, learning_rate=0.1, random_state=42):
    """
    Gradient Boosting Regressor para múltiples outputs usando MultiOutputRegressor
    """
    base_model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    model = MultiOutputRegressor(base_model)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_pred = np.round(np.maximum(y_pred, 0))
    return y_pred

def ada_boost_regressor_multioutput(x_train, y_train, x_val, n_estimators=50, learning_rate=1.0, random_state=42):
    """
    AdaBoost Regressor para múltiples outputs usando MultiOutputRegressor
    """
    base_model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    model = MultiOutputRegressor(base_model)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_pred = np.round(np.maximum(y_pred, 0))
    return y_pred

class MultiOutputMLP(nn.Module):
    """
    Red neuronal multicapa para regresión multioutput.
    """
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(MultiOutputMLP, self).__init__()
        
        # Crear lista de capas
        layers = []
        prev_size = input_size
        
        # Capas ocultas
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Capa de salida
        layers.append(nn.Linear(prev_size, output_size))
        
        # Crear modelo secuencial
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def neural_network_multioutput(x_train, y_train, x_val, y_val, hidden_sizes=[128, 64, 32], 
                              learning_rate=0.001, epochs=100, batch_size=32, 
                              dropout_rate=None, early_stopping_patience=None, 
                              normalize=False, verbose=False,
                              use_adam=True, l2_regularization=None, 
                              use_scheduler=False, plot_losses=True):
    """
    Red neuronal multioutput usando PyTorch con mejoras opcionales.
    
    Args:
        x_train: Features de entrenamiento
        y_train: Targets de entrenamiento (multioutput)
        x_val: Features de validación
        y_val: Targets de validación (multioutput)
        hidden_sizes: Lista con tamaños de capas ocultas [128, 64, 32]
        learning_rate: Tasa de aprendizaje (default: 0.001)
        epochs: Número máximo de épocas (default: 100)
        batch_size: Tamaño del batch (default: 32)
        dropout_rate: Tasa de dropout (None = sin dropout, ej: 0.2)
        early_stopping_patience: Paciencia para early stopping (None = sin early stopping, ej: 10)
        normalize: Si normalizar los datos (default: False)
        verbose: Si mostrar progreso (default: False)
        use_adam: Si usar optimizador Adam, sino SGD (default: True)
        l2_regularization: L2 regularization (weight_decay) (None = sin L2, ej: 0.01)
        use_scheduler: Si usar ReduceLROnPlateau scheduler (default: False)
        plot_losses: Si generar gráfico de training vs validation loss (default: False)
    
    Returns:
        tuple: (y_pred, train_losses, val_losses)
            - y_pred: np.array con predicciones en datos de validación
            - train_losses: list con historial de training loss por época
            - val_losses: list con historial de validation loss por época
        
    Ejemplo de uso básico:
        # Configuración mínima (solo usa Adam optimizer básico)
        y_pred, train_losses, val_losses = neural_network_multioutput(x_train, y_train, x_val, y_val)
        
    Ejemplo con todas las mejoras:
        y_pred, train_losses, val_losses = neural_network_multioutput(
            x_train, y_train, x_val, y_val,
            hidden_sizes=[256, 128, 64],
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
            plot_losses=True
        )
    """
    
    # Convertir a numpy si es necesario
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    
    # Configurar device (usa CPU por defecto, GPU si está disponible)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"🔧 Usando device: {device}")
        print(f"🏗️  Arquitectura: {x_train.shape[1]} -> {' -> '.join(map(str, hidden_sizes))} -> {y_train.shape[1]}")
        print(f"⚙️  Mejoras: Adam={use_adam}, Dropout={dropout_rate}, L2={l2_regularization}, Norm={normalize}, EarlySt={early_stopping_patience is not None}, Sched={use_scheduler}, Plot={plot_losses}")
    
    # Normalización opcional
    scaler_x = None
    scaler_y = None
    
    if normalize:
        # Normalizar features
        scaler_x = StandardScaler()
        x_train_scaled = scaler_x.fit_transform(x_train)
        x_val_scaled = scaler_x.transform(x_val)
        
        # Normalizar targets
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
    else:
        x_train_scaled = x_train
        x_val_scaled = x_val
        y_train_scaled = y_train
        y_val_scaled = y_val
    
    # Convertir a tensores
    x_train_tensor = torch.FloatTensor(x_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    x_val_tensor = torch.FloatTensor(x_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val_scaled).to(device)
    
    # Crear dataset y dataloader
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Crear modelo (con o sin dropout)
    input_size = x_train.shape[1]
    output_size = y_train.shape[1]
    dropout_for_model = dropout_rate if dropout_rate is not None else 0.0
    model = MultiOutputMLP(input_size, hidden_sizes, output_size, dropout_for_model).to(device)
    
    # Definir loss y optimizer
    criterion = nn.MSELoss()
    
    # Optimizer con o sin L2 regularization
    if use_adam:
        if l2_regularization is not None:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_regularization)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        if l2_regularization is not None:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_regularization)
        else:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Scheduler opcional
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Early stopping opcional
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    use_early_stopping = early_stopping_patience is not None
    
    # Lista para almacenar el historial de pérdidas
    train_losses = []
    val_losses = []
    
    if verbose:
        print(f"🚀 Iniciando entrenamiento...")
    
    # Entrenamiento
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_x, batch_y in train_loader:
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        # Calcular pérdida promedio de la época
        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        
        # Calcular validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_losses.append(val_loss)
        model.train()  # Volver a modo entrenamiento
        
        # Actualizar scheduler si está activado (usar validation loss)
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Early stopping si está activado (usar validation loss)
        if use_early_stopping:
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Guardar mejor modelo
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Comprobar early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"⏹️  Early stopping en época {epoch+1}")
                break
        
        # Verbose cada 10 épocas
        if verbose and (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            if use_early_stopping:
                print(f"Época {epoch+1}/{epochs} - Train Loss: {avg_loss:.6f} - Val Loss: {val_loss:.6f} - LR: {current_lr:.2e} - Best Val: {best_loss:.6f}")
            else:
                print(f"Época {epoch+1}/{epochs} - Train Loss: {avg_loss:.6f} - Val Loss: {val_loss:.6f} - LR: {current_lr:.2e}")
    
    # Cargar el mejor modelo si se usó early stopping
    if use_early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluación
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(x_val_tensor)
        y_pred_scaled = y_pred_tensor.cpu().numpy()
    
    # Desnormalizar predicciones si es necesario
    if normalize and scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
    else:
        y_pred = y_pred_scaled
    
    # Aplicar constrains (no negativos, redondear)
    y_pred = np.round(np.maximum(y_pred, 0))
    
    # Crear gráfico de pérdidas si se solicita
    if plot_losses:
        plt.figure(figsize=(10, 6))
        epochs_range = range(1, len(train_losses) + 1)
        
        plt.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.title('Training vs Validation Loss', fontsize=16, fontweight='bold')
        plt.xlabel('Época', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Marcar early stopping si se usó
        if use_early_stopping and len(train_losses) < epochs:
            plt.axvline(x=len(train_losses), color='orange', linestyle='--', 
                       label=f'Early Stopping (época {len(train_losses)})', alpha=0.7)
            plt.legend(fontsize=12)
        
        # Marcar el punto de mejor loss si se usó early stopping
        if use_early_stopping:
            best_epoch = val_losses.index(min(val_losses)) + 1
            plt.axvline(x=best_epoch, color='green', linestyle=':', 
                       label=f'Mejor modelo (época {best_epoch})', alpha=0.7)
            plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        if verbose:
            print(f"📊 Gráfico de pérdidas generado")
            print(f"📈 Training loss final: {train_losses[-1]:.6f}")
            print(f"📉 Validation loss final: {val_losses[-1]:.6f}")
            if use_early_stopping:
                print(f"🏆 Mejor validation loss: {min(val_losses):.6f} en época {val_losses.index(min(val_losses)) + 1}")
    
    if verbose:
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        print(f"✅ Entrenamiento completado!")
        print(f"📊 Train Loss final: {final_train_loss:.6f}")
        print(f"📊 Val Loss final: {final_val_loss:.6f}")
        if use_early_stopping:
            print(f"🏆 Mejor val loss: {best_loss:.6f}")
        print(f"🎯 Predicciones generadas: {y_pred.shape}")
        print(f"📈 Rango predicciones: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    
    return y_pred, train_losses, val_losses

def plot_multiple_loss_histories(loss_histories, model_names=None, subplot_layout='together', figsize=(12, 8)):
    """
    Grafica múltiples historiales de training y validation loss para comparar modelos.
    
    Args:
        loss_histories: lista de tuplas (train_losses, val_losses) de cada modelo
        model_names: lista de nombres para cada modelo (opcional)
        subplot_layout: 'together' (un gráfico) o 'separate' (subplots separados)
        figsize: tupla con tamaño de figura (ancho, alto)
    
    Returns:
        None: Muestra el gráfico
        
    Ejemplo de uso:
        # Entrenar múltiples modelos
        y_pred1, train1, val1 = neural_network_multioutput(x_train, y_train, x_val, y_val, hidden_sizes=[64, 32])
        y_pred2, train2, val2 = neural_network_multioutput(x_train, y_train, x_val, y_val, hidden_sizes=[128, 64])
        y_pred3, train3, val3 = neural_network_multioutput(x_train, y_train, x_val, y_val, hidden_sizes=[256, 128, 64])
        
        # Comparar en un solo gráfico
        plot_multiple_loss_histories(
            [(train1, val1), (train2, val2), (train3, val3)],
            ['Modelo 64-32', 'Modelo 128-64', 'Modelo 256-128-64'],
            subplot_layout='together'
        )
        
        # Comparar en subplots separados
        plot_multiple_loss_histories(
            [(train1, val1), (train2, val2), (train3, val3)],
            ['Modelo 64-32', 'Modelo 128-64', 'Modelo 256-128-64'],
            subplot_layout='separate'
        )
    """
    
    n_models = len(loss_histories)
    
    # Generar nombres por defecto si no se proporcionan
    if model_names is None:
        model_names = [f'Modelo {i+1}' for i in range(n_models)]
    
    # Colores para cada modelo
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    if subplot_layout == 'together':
        # Un solo gráfico con todas las curvas
        plt.figure(figsize=figsize)
        
        for i, (train_losses, val_losses) in enumerate(loss_histories):
            epochs = range(1, len(train_losses) + 1)
            color = colors[i]
            
            # Training loss (línea sólida)
            plt.plot(epochs, train_losses, '-', color=color, linewidth=2, 
                    label=f'{model_names[i]} - Train', alpha=0.8)
            
            # Validation loss (línea punteada)
            plt.plot(epochs, val_losses, '--', color=color, linewidth=2, 
                    label=f'{model_names[i]} - Val', alpha=0.8)
        
        plt.title('Comparación de Loss: Training vs Validation', fontsize=16, fontweight='bold')
        plt.xlabel('Época', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    elif subplot_layout == 'separate':
        # Subplots separados para cada modelo
        if n_models <= 2:
            rows, cols = 1, n_models
        elif n_models <= 4:
            rows, cols = 2, 2
        elif n_models <= 6:
            rows, cols = 2, 3
        elif n_models <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = 4, 3  # Máximo 12 modelos
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, (train_losses, val_losses) in enumerate(loss_histories):
            if i >= len(axes):
                break
                
            epochs = range(1, len(train_losses) + 1)
            
            axes[i].plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
            axes[i].plot(epochs, val_losses, 'r--', linewidth=2, label='Validation Loss')
            
            axes[i].set_title(f'{model_names[i]}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Época', fontsize=10)
            axes[i].set_ylabel('Loss (MSE)', fontsize=10)
            axes[i].legend(fontsize=9)
            axes[i].grid(True, alpha=0.3)
        
        # Ocultar subplots vacíos
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Comparación Individual de Modelos', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    else:
        raise ValueError("subplot_layout debe ser 'together' o 'separate'")
    
    # Mostrar estadísticas finales
    print("\n📊 Estadísticas Finales:")
    print("-" * 60)
    for i, (train_losses, val_losses) in enumerate(loss_histories):
        final_train = train_losses[-1]
        final_val = val_losses[-1]
        min_val = min(val_losses)
        min_val_epoch = val_losses.index(min_val) + 1
        
        print(f"{model_names[i]}:")
        print(f"  📈 Train Loss final: {final_train:.6f}")
        print(f"  📉 Val Loss final: {final_val:.6f}")
        print(f"  🏆 Mejor Val Loss: {min_val:.6f} (época {min_val_epoch})")
        print(f"  📏 Épocas entrenadas: {len(train_losses)}")
        print()


def compare_model_architectures(x_train, y_train, x_val, y_val, architectures, **common_params):
    """
    Función de conveniencia para entrenar y comparar múltiples arquitecturas de redes neuronales.
    
    Args:
        x_train, y_train, x_val, y_val: Datos de entrenamiento y validación
        architectures: lista de listas con tamaños de capas ocultas para cada modelo
        **common_params: parámetros comunes para todos los modelos (epochs, learning_rate, etc.)
    
    Returns:
        dict: Diccionario con resultados de cada modelo
        
    Ejemplo de uso:
        architectures = [
            [64, 32],
            [128, 64, 32],
            [256, 128, 64],
            [512, 256, 128, 64]
        ]
        
        results = compare_model_architectures(
            x_train, y_train, x_val, y_val,
            architectures,
            epochs=100,
            learning_rate=0.001,
            dropout_rate=0.2,
            early_stopping_patience=10,
            normalize=True,
            verbose=False,
            plot_losses=False
        )
        
        # Graficar comparación
        loss_histories = [results[f'model_{i}']['losses'] for i in range(len(architectures))]
        model_names = [f'Arch {"-".join(map(str, arch))}' for arch in architectures]
        plot_multiple_loss_histories(loss_histories, model_names, 'together')
    """
    
    results = {}
    loss_histories = []
    model_names = []
    
    print("🔄 Entrenando múltiples arquitecturas...")
    
    for i, hidden_sizes in enumerate(architectures):
        arch_name = f'model_{i}'
        arch_display = f'Arch {"-".join(map(str, hidden_sizes))}'
        
        print(f"\Entrenando {arch_display}...")
        
        # Entrenar modelo con la arquitectura actual
        y_pred, train_losses, val_losses = neural_network_multioutput(
            x_train, y_train, x_val, y_val,
            hidden_sizes=hidden_sizes,
            learning_rate=common_params.get('learning_rate', 0.001),
            epochs=common_params.get('epochs', 100),
            batch_size=common_params.get('batch_size', 32),
            dropout_rate=common_params.get('dropout_rate', None),
            early_stopping_patience=common_params.get('early_stopping_patience', None),
            normalize=common_params.get('normalize', False),
            verbose=False,
            use_adam=common_params.get('use_adam', True),
            l2_regularization=common_params.get('l2_regularization', None),
            use_scheduler=common_params.get('use_scheduler', False),
            plot_losses=False
        )
        
        # Guardar resultados
        results[arch_name] = {
            'architecture': hidden_sizes,
            'predictions': y_pred,
            'losses': (train_losses, val_losses),
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': min(val_losses),
            'best_epoch': val_losses.index(min(val_losses)) + 1,
            'total_epochs': len(train_losses)
        }
        
        loss_histories.append((train_losses, val_losses))
        model_names.append(arch_display)
    
    print("\n✅ Entrenamiento completado!")
    
    # Graficar comparación automáticamente
    print("\n📊 Generando gráfico de comparación...")
    plot_multiple_loss_histories(loss_histories, model_names, 'together')
    
    return results

