import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import models_f2 as md2
import metrics as mt
from itertools import product
import warnings
warnings.filterwarnings('ignore')


def cross_validation_neural_network_multioutput(X_train, y_train, hidden_sizes, 
                                               learning_rate_list=[0.001, 0.01, 0.1],
                                               epochs_list=[50, 100, 200],
                                               batch_size_list=[16, 32, 64],
                                               dropout_rate_list=[None, 0.2, 0.5],
                                               l2_reg_list=[None, 0.001, 0.01],
                                               use_adam_list=[True],
                                               use_scheduler_list=[True],
                                               cv_folds=5, random_state=42,
                                               early_stopping_patience=10,
                                               target_names=None):
    """
    Cross-validation avanzado para Red Neuronal Multioutput con mejoras.
    
    Parameters:
    -----------
    X_train : array-like
        Features de entrenamiento
    y_train : array-like  
        Targets de entrenamiento
    hidden_sizes : list
        Tamaños de capas ocultas fijas, ej: [128, 64, 32]
    learning_rate_list : list
        Lista de learning rates a probar
    epochs_list : list
        Lista de épocas a probar
    batch_size_list : list
        Lista de batch sizes a probar
    dropout_rate_list : list
        Lista de dropout rates a probar (None = sin dropout)
    l2_reg_list : list
        Lista de regularización L2 a probar (None = sin L2)
    use_adam_list : list
        Lista de opciones de Adam optimizer a probar (ej: [False, True])
    use_scheduler_list : list
        Lista de opciones de learning rate scheduler a probar (ej: [False, True])
    cv_folds : int
        Número de folds
    early_stopping_patience : int
        Paciencia para early stopping
    target_names : list
        Nombres de targets (opcional)
        
    Returns:
    --------
    dict: Mejores parámetros y resultados
    """
    
    # Preparar datos
    X = np.array(X_train)
    y = np.array(y_train)
    
    if target_names is None:
        target_names = [f'target_{i}' for i in range(y.shape[1])]
    
    print(f"🧠 Cross-Validation Red Neuronal Avanzado - Capas: {hidden_sizes}")
    print(f"   Datos: {X.shape[0]} filas, {X.shape[1]} features, {y.shape[1]} targets")
    print(f"   Combinaciones: {len(learning_rate_list) * len(epochs_list) * len(batch_size_list) * len(dropout_rate_list) * len(l2_reg_list) * len(use_adam_list) * len(use_scheduler_list)}")
    print(f"   Opciones Adam: {use_adam_list}, Scheduler: {use_scheduler_list}, Early Stopping: {early_stopping_patience}")
    
    # Cross-validation
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Grid de parámetros ampliado con Adam y Scheduler
    param_combinations = list(product(learning_rate_list, epochs_list, batch_size_list, 
                                    dropout_rate_list, l2_reg_list, use_adam_list, use_scheduler_list))
    
    resultados = []
    mejor_score = np.inf
    mejores_params = None
    
    for i, (lr, epochs, batch_size, dropout_rate, l2_reg, use_adam, use_scheduler) in enumerate(param_combinations):
        print(f"🔄 {i+1}/{len(param_combinations)}: lr={lr}, epochs={epochs}, batch={batch_size}, "
              f"dropout={dropout_rate}, l2={l2_reg}, adam={use_adam}, scheduler={use_scheduler}")
        
        scores_fold = []
        
        # Cross-validation para esta combinación
        for train_idx, val_idx in kfold.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Entrenar red neuronal con todas las mejoras
            try:
                y_pred, _, _ = md2.neural_network_multioutput(
                    X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                    hidden_sizes=hidden_sizes,
                    learning_rate=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    dropout_rate=dropout_rate,
                    early_stopping_patience=early_stopping_patience,
                    normalize=True,
                    verbose=False,
                    use_adam=use_adam,
                    l2_regularization=l2_reg,
                    use_scheduler=use_scheduler,
                    plot_losses=False
                )
                
                # Calcular MAE
                mae = np.mean(np.abs(y_val_fold - y_pred))
                scores_fold.append(mae)
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                scores_fold.append(np.inf)
        
        # Promedio de MAE
        mae_promedio = np.mean(scores_fold)
        mae_std = np.std(scores_fold)
        
        resultados.append({
            'learning_rate': lr,
            'epochs': epochs,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate,
            'l2_regularization': l2_reg,
            'use_adam': use_adam,
            'use_scheduler': use_scheduler,
            'mae_promedio': mae_promedio,
            'mae_std': mae_std
        })
        
        # Actualizar mejor resultado
        if mae_promedio < mejor_score:
            mejor_score = mae_promedio
            mejores_params = {
                'learning_rate': lr,
                'epochs': epochs,
                'batch_size': batch_size,
                'dropout_rate': dropout_rate,
                'l2_regularization': l2_reg,
                'use_adam': use_adam,
                'use_scheduler': use_scheduler,
                'early_stopping_patience': early_stopping_patience
            }
        
        print(f"   MAE: {mae_promedio:.4f} ± {mae_std:.4f}")
    
    # Crear DataFrame de resultados
    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values('mae_promedio')
    
    print(f"\n🏆 MEJOR RESULTADO:")
    print(f"   Parámetros: {mejores_params}")
    print(f"   MAE: {mejor_score:.4f}")
    
    print(f"\n📊 TOP 3:")
    for i, (_, row) in enumerate(df_resultados.head(3).iterrows()):
        print(f"   #{i+1}: lr={row['learning_rate']}, epochs={row['epochs']}, "
              f"batch={row['batch_size']}, dropout={row['dropout_rate']}, "
              f"l2={row['l2_regularization']}, adam={row['use_adam']}, "
              f"scheduler={row['use_scheduler']}, MAE={row['mae_promedio']:.4f}")
    
    return {
        'mejores_params': mejores_params,
        'mejor_score': mejor_score,
        'resultados_detallados': df_resultados,
        'hidden_sizes': hidden_sizes
    }


# Función de ejemplo/prueba
if __name__ == "__main__":
    """
    Ejemplo de uso
    """
    # Datos de ejemplo
    np.random.seed(42)
    X_ejemplo = np.random.randn(200, 5)
    y_ejemplo = np.random.randn(200, 3)
    
    # Cross-validation avanzado
    resultado = cross_validation_neural_network_multioutput(
        X_train=X_ejemplo,
        y_train=y_ejemplo,
        hidden_sizes=[64, 32],
        learning_rate_list=[0.001, 0.01],
        epochs_list=[20, 50],
        batch_size_list=[16, 32],
        dropout_rate_list=[None, 0.2],
        l2_reg_list=[None, 0.001],
        use_adam_list=[False, True],
        use_scheduler_list=[False, True],
        cv_folds=3,
        early_stopping_patience=5
    )
    
    print(f"\n✅ Ejemplo completado!")
    print(f"Mejores parámetros: {resultado['mejores_params']}")