# Regresion lineal
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

ds_14 = pd.read_csv('data/processed/features1/train/dataset_train_11.csv')



#Elimino las columnas fecha y fecha y hora
ds_14 = ds_14.drop(columns=['fecha', 'fecha_hora'])

#Hago PCA y grafico las dos features mas significativas
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(ds_14.drop(columns=['target']))
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=ds_14['target'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Target')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA de las dos primeras componentes principales')
plt.show()

#Le hago one hot encoding a la columna hora
#ds_14 = pd.get_dummies(ds_14, columns=['hora'], prefix='hora', drop_first=True, dtype='int64')

print(ds_14.head())

# Separar features (X) y target (y)
ds_y = ds_14['target']
ds_X = ds_14.drop(columns=['target'])

# Dividir en train y validation (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    ds_X, ds_y, 
    test_size=0.2, 
    random_state=13,
    shuffle=False
)

print(f"Dataset completo: {len(ds_14)} muestras")
print(f"Train: {len(X_train)} muestras")
print(f"Validation: {len(X_val)} muestras")
print(f"Features: {X_train.shape[1]}")

#ENTRENAMOS EL MODELO CON TRAIN
model = LinearRegression()
model.fit(X_train, y_train)

# Ridge Regression
ridge_model = Ridge(alpha=1000, random_state=13)
ridge_model.fit(X_train, y_train)

# Lasso Regression
lasso_model = Lasso(alpha=0.1, random_state=13)
lasso_model.fit(X_train, y_train)

#Realizamos las predicciones con el set de validación
y_pred = model.predict(X_val)

#Realizamos las predicciones con el modelo de Ridge
ridge_y_pred = ridge_model.predict(X_val)

#Realizamos las predicciones con el modelo de Lasso
lasso_y_pred = lasso_model.predict(X_val)

#Evaluamos el modelo
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
rmse = root_mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
mape = mean_absolute_percentage_error(y_val, y_pred)

#Evaluamos el modelo de Ridge
ridge_mse = mean_squared_error(y_val, ridge_y_pred)
ridge_r2 = r2_score(y_val, ridge_y_pred)
ridge_rmse = root_mean_squared_error(y_val, ridge_y_pred)
ridge_mae = mean_absolute_error(y_val, ridge_y_pred)
ridge_mape = mean_absolute_percentage_error(y_val, ridge_y_pred)

#Evaluamos el modelo de Lasso
lasso_mse = mean_squared_error(y_val, lasso_y_pred)
lasso_r2 = r2_score(y_val, lasso_y_pred)
lasso_rmse = root_mean_squared_error(y_val, lasso_y_pred)
lasso_mae = mean_absolute_error(y_val, lasso_y_pred)
lasso_mape = mean_absolute_percentage_error(y_val, lasso_y_pred)

print("Resultados del modelo de regresión lineal:")
print(f"Error cuadrático medio (MSE): {mse}")
print(f"R^2: {r2}")
print(f"Raíz del error cuadrático medio (RMSE): {rmse}")
print(f"Error absoluto medio (MAE): {mae}")
print(f"Error porcentual absoluto medio (MAPE): {mape}")

print("\nResultados del modelo de Ridge Regression:")
print(f"Error cuadrático medio (MSE): {ridge_mse}")
print(f"R^2: {ridge_r2}")
print(f"Raíz del error cuadrático medio (RMSE): {ridge_rmse}")
print(f"Error absoluto medio (MAE): {ridge_mae}")
print(f"Error porcentual absoluto medio (MAPE): {ridge_mape}")

print("\nResultados del modelo de Lasso Regression:")
print(f"Error cuadrático medio (MSE): {lasso_mse}")
print(f"R^2: {lasso_r2}")
print(f"Raíz del error cuadrático medio (RMSE): {lasso_rmse}")
print(f"Error absoluto medio (MAE): {lasso_mae}")
print(f"Error porcentual absoluto medio (MAPE): {lasso_mape}")

# Importar más modelos
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
import numpy as np

# Entrenar modelos adicionales
print("\n" + "="*50)
print("ENTRENANDO MODELOS ADICIONALES...")
print("="*50)

# ElasticNet (combina Ridge + Lasso)
elastic_model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=13)
elastic_model.fit(X_train, y_train)

# Decision Tree
tree_model = DecisionTreeRegressor(random_state=13, max_depth=10)
tree_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=13, max_depth=10)
rf_model.fit(X_train, y_train)

# Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=13, max_depth=6)
gb_model.fit(X_train, y_train)

# K-Nearest Neighbors
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Support Vector Regression (cuidado: puede ser lento)
svr_model = SVR(kernel='rbf', C=1.0, gamma='scale')
svr_model.fit(X_train, y_train)

# Predicciones
elastic_pred = elastic_model.predict(X_val)
tree_pred = tree_model.predict(X_val)
rf_pred = rf_model.predict(X_val)
gb_pred = gb_model.predict(X_val)
knn_pred = knn_model.predict(X_val)
svr_pred = svr_model.predict(X_val)

# Evaluaciones
models_results = [
    ("Linear Regression", y_pred),
    ("Ridge", ridge_y_pred),
    ("Lasso", lasso_y_pred),
    ("ElasticNet", elastic_pred),
    ("Decision Tree", tree_pred),
    ("Random Forest", rf_pred),
    ("Gradient Boosting", gb_pred),
    ("K-NN", knn_pred),
    ("SVR", svr_pred)
]

print("\n" + "="*80)
print("COMPARACIÓN DE TODOS LOS MODELOS")
print("="*80)
print("Modelo                R²        MSE        RMSE       MAE        MAPE")
print("-" * 75)

best_r2 = -float('inf')
best_model_name = ""

for model_name, predictions in models_results:
    mse_temp = mean_squared_error(y_val, predictions)
    r2_temp = r2_score(y_val, predictions)
    rmse_temp = root_mean_squared_error(y_val, predictions)
    mae_temp = mean_absolute_error(y_val, predictions)
    mape_temp = mean_absolute_percentage_error(y_val, predictions)
    
    print(f"{model_name:<18} {r2_temp:>8.4f} {mse_temp:>10.4f} {rmse_temp:>10.4f} {mae_temp:>10.4f} {mape_temp:>10.4f}")
    
    if r2_temp > best_r2:
        best_r2 = r2_temp
        best_model_name = model_name

print("-" * 75)
print(f"🏆 MEJOR MODELO: {best_model_name} (R² = {best_r2:.4f})")

# Información adicional sobre algunos modelos
print("\n" + "="*50)
print("INFORMACIÓN ADICIONAL")
print("="*50)

# Feature importance para Random Forest
print("\nTop 10 features más importantes (Random Forest):")
feature_names = X_train.columns
rf_importances = rf_model.feature_importances_
top_features = sorted(zip(feature_names, rf_importances), key=lambda x: x[1], reverse=True)[:10]
for feature, importance in top_features:
    print(f"  {feature}: {importance:.4f}")

# Features eliminadas por Lasso
zero_coef_count = (lasso_model.coef_ == 0).sum()
total_features = len(lasso_model.coef_)
print(f"\nFeatures eliminadas por Lasso: {zero_coef_count}/{total_features} ({zero_coef_count/total_features*100:.1f}%)")

# Features eliminadas por ElasticNet
elastic_zero_coef = (elastic_model.coef_ == 0).sum()
print(f"Features eliminadas por ElasticNet: {elastic_zero_coef}/{total_features} ({elastic_zero_coef/total_features*100:.1f}%)")


#Ahora probamos una red neuronal
from sklearn.neural_network import MLPRegressor
# Entrenar una red neuronal
nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=13)
nn_model.fit(X_train, y_train)

# Predicciones con la red neuronal
nn_y_pred = nn_model.predict(X_val)
# Evaluación de la red neuronal
nn_mse = mean_squared_error(y_val, nn_y_pred)
nn_r2 = r2_score(y_val, nn_y_pred)
nn_rmse = root_mean_squared_error(y_val, nn_y_pred)
nn_mae = mean_absolute_error(y_val, nn_y_pred)
nn_mape = mean_absolute_percentage_error(y_val, nn_y_pred)

print("\nResultados del modelo de Red Neuronal:")
print(f"Error cuadrático medio (MSE): {nn_mse}")
print(f"R^2: {nn_r2}")
print(f"Raíz del error cuadrático medio (RMSE): {nn_rmse}")
print(f"Error absoluto medio (MAE): {nn_mae}")
print(f"Error porcentual absoluto medio (MAPE): {nn_mape}")

# Comparación de la red neuronal con los otros modelos
print("\n" + "="*80)
print("COMPARACIÓN DE LA RED NEURONAL CON LOS OTROS MODELOS")
print("="*80)
print("Modelo                R²        MSE        RMSE       MAE        MAPE")
print("-" * 75)
print(f"Red Neuronal         {nn_r2:>8.4f} {nn_mse:>10.4f} {nn_rmse:>10.4f} {nn_mae:>10.4f} {nn_mape:>10.4f}")
# Añadir la red neuronal a los resultados
models_results.append(("Red Neuronal", nn_y_pred))
# Recalcular el mejor modelo
best_r2 = -float('inf')
best_model_name = ""

for model_name, predictions in models_results:
    mse_temp = mean_squared_error(y_val, predictions)
    r2_temp = r2_score(y_val, predictions)
    rmse_temp = root_mean_squared_error(y_val, predictions)
    mae_temp = mean_absolute_error(y_val, predictions)
    mape_temp = mean_absolute_percentage_error(y_val, predictions)
    
    if r2_temp > best_r2:
        best_r2 = r2_temp
        best_model_name = model_name
        
print("-" * 75)
print(f"🏆 MEJOR MODELO: {best_model_name} (R² = {best_r2:.4f})")


# Múltiples configuraciones de redes neuronales
print("\n" + "="*80)
print("PROBANDO 10 CONFIGURACIONES DIFERENTES DE REDES NEURONALES")
print("="*80)

# Configuraciones de redes neuronales
nn_configs = [
    # (nombre, hidden_layers, max_iter, learning_rate, alpha, solver)
    ("NN-1 Simple", (50,), 1000, 'constant', 0.0001, 'adam'),
    ("NN-2 Profunda", (100, 50, 25), 1000, 'constant', 0.0001, 'adam'),
    ("NN-3 Ancha", (200, 150), 1000, 'constant', 0.0001, 'adam'),
    ("NN-4 Muy Profunda", (100, 80, 60, 40, 20), 1500, 'constant', 0.0001, 'adam'),
    ("NN-5 Alta Reg", (100, 50), 1000, 'constant', 0.01, 'adam'),
    ("NN-6 Baja Reg", (100, 50), 1000, 'constant', 0.00001, 'adam'),
    ("NN-7 LR Alto", (100, 50), 1000, 'constant', 0.0001, 'sgd'),
    ("NN-8 Adaptativo", (80, 60, 40), 1000, 'adaptive', 0.001, 'adam'),
    #("NN-9 LBFGS", (100, 50), 1000, 'constant', 0.0001, 'lbfgs'),
    ("NN-10 Mega", (150, 100, 75, 50), 2000, 'invscaling', 0.0005, 'adam')
]

nn_results = []
nn_models = []

for i, (name, hidden_layers, max_iter, learning_rate, alpha, solver) in enumerate(nn_configs):
    print(f"Entrenando {name}... ", end="")
    
    try:
        # Configurar learning rate inicial
        learning_rate_init = 0.001 if solver == 'adam' else 0.01
        if solver == 'sgd':
            learning_rate_init = 0.1
        elif solver == 'lbfgs':
            learning_rate_init = 0.001
            
        nn_temp = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            alpha=alpha,
            solver=solver,
            random_state=13,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        nn_temp.fit(X_train, y_train)
        nn_pred_temp = nn_temp.predict(X_val)
        
        # Métricas
        nn_mse_temp = mean_squared_error(y_val, nn_pred_temp)
        nn_r2_temp = r2_score(y_val, nn_pred_temp)
        nn_rmse_temp = root_mean_squared_error(y_val, nn_pred_temp)
        nn_mae_temp = mean_absolute_error(y_val, nn_pred_temp)
        nn_mape_temp = mean_absolute_percentage_error(y_val, nn_pred_temp)
        
        nn_results.append({
            'name': name,
            'model': nn_temp,
            'predictions': nn_pred_temp,
            'r2': nn_r2_temp,
            'mse': nn_mse_temp,
            'rmse': nn_rmse_temp,
            'mae': nn_mae_temp,
            'mape': nn_mape_temp,
            'layers': hidden_layers,
            'iterations': nn_temp.n_iter_,
            'solver': solver,
            'alpha': alpha
        })
        
        print(f"✅ R² = {nn_r2_temp:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        continue

# Mostrar resultados de todas las redes neuronales
print("\n" + "="*90)
print("RESULTADOS DE TODAS LAS REDES NEURONALES")
print("="*90)
print("Modelo            Capas              R²        MSE        RMSE       MAE        MAPE      Iter")
print("-" * 85)

for result in nn_results:
    layers_str = str(result['layers']).replace(' ', '')
    print(f"{result['name']:<13} {layers_str:<18} {result['r2']:>8.4f} {result['mse']:>10.4f} "
          f"{result['rmse']:>10.4f} {result['mae']:>10.4f} {result['mape']:>10.4f} {result['iterations']:>6}")

# Encontrar la mejor red neuronal
best_nn = max(nn_results, key=lambda x: x['r2'])
print("-" * 85)
print(f"🏆 MEJOR RED NEURONAL: {best_nn['name']} (R² = {best_nn['r2']:.4f})")
print(f"   Arquitectura: {best_nn['layers']}")
print(f"   Solver: {best_nn['solver']}, Alpha: {best_nn['alpha']}")
print(f"   Iteraciones: {best_nn['iterations']}")

# Agregar todas las redes neuronales a los resultados generales
for result in nn_results:
    models_results.append((result['name'], result['predictions']))

# COMPARACIÓN FINAL DE TODOS LOS MODELOS
print("\n" + "="*90)
print("COMPARACIÓN FINAL DE TODOS LOS MODELOS (Lineales + Ensemble + Redes Neuronales)")
print("="*90)
print("Modelo                R²        MSE        RMSE       MAE        MAPE")
print("-" * 75)

final_results = []
for model_name, predictions in models_results:
    mse_temp = mean_squared_error(y_val, predictions)
    r2_temp = r2_score(y_val, predictions)
    rmse_temp = root_mean_squared_error(y_val, predictions)
    mae_temp = mean_absolute_error(y_val, predictions)
    mape_temp = mean_absolute_percentage_error(y_val, predictions)
    
    final_results.append({
        'name': model_name,
        'r2': r2_temp,
        'mse': mse_temp,
        'rmse': rmse_temp,
        'mae': mae_temp,
        'mape': mape_temp
    })

# Ordenar por R² (mejor a peor)
final_results.sort(key=lambda x: x['r2'], reverse=True)

for i, result in enumerate(final_results):
    medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
    print(f"{medal} {result['name']:<16} {result['r2']:>8.4f} {result['mse']:>10.4f} "
          f"{result['rmse']:>10.4f} {result['mae']:>10.4f} {result['mape']:>10.4f}")

print("-" * 75)
print(f"🏆 GANADOR ABSOLUTO: {final_results[0]['name']} (R² = {final_results[0]['r2']:.4f})")

# Análisis adicional de la mejor red neuronal
if best_nn['name'] == final_results[0]['name']:
    print(f"\n🎯 LA MEJOR RED NEURONAL ES TAMBIÉN EL MEJOR MODELO GENERAL!")
    print(f"   Detalles de la arquitectura ganadora:")
    print(f"   - Capas ocultas: {best_nn['layers']}")
    print(f"   - Número de parámetros estimados: {sum([h1*h2 for h1,h2 in zip([X_train.shape[1]] + list(best_nn['layers']), list(best_nn['layers']) + [1])])}")
    print(f"   - Solver: {best_nn['solver']}")
    print(f"   - Regularización (alpha): {best_nn['alpha']}")
    print(f"   - Convergencia en: {best_nn['iterations']} iteraciones")

# Top 5 modelos
print(f"\n📊 TOP 5 MODELOS:")
for i in range(min(5, len(final_results))):
    result = final_results[i]
    print(f"   {i+1}. {result['name']}: R² = {result['r2']:.4f}")

# Diferencia entre el mejor y peor modelo
worst_r2 = final_results[-1]['r2']
best_r2 = final_results[0]['r2']
improvement = ((best_r2 - worst_r2) / abs(worst_r2)) * 100 if worst_r2 != 0 else 0
print(f"\n📈 Mejora del mejor vs peor modelo: {improvement:.1f}%")
print(f"   Mejor R²: {best_r2:.4f}")
print(f"   Peor R²: {worst_r2:.4f}")

# Agregar después del código existente
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.linear_model import HuberRegressor, TheilSenRegressor, RANSACRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid

print("\n" + "="*80)
print("🔍 AGREGANDO MÁS MODELOS Y OPTIMIZACIÓN DE HIPERPARÁMETROS")
print("="*80)

# Modelos adicionales para probar
additional_models = {}

# 1. AdaBoost
print("1️⃣  Entrenando AdaBoost Regressor...")
ada_model = AdaBoostRegressor(n_estimators=100, random_state=13)
ada_model.fit(X_train, y_train)
ada_pred = ada_model.predict(X_val)
additional_models["AdaBoost"] = ada_pred
print("   ✅ AdaBoost completado")

# 2. Extra Trees
print("2️⃣  Entrenando Extra Trees Regressor...")
extra_model = ExtraTreesRegressor(n_estimators=100, random_state=13, max_depth=10)
extra_model.fit(X_train, y_train)
extra_pred = extra_model.predict(X_val)
additional_models["Extra Trees"] = extra_pred
print("   ✅ Extra Trees completado")

# 3. Bagging
print("3️⃣  Entrenando Bagging Regressor...")
bag_model = BaggingRegressor(n_estimators=100, random_state=13)
bag_model.fit(X_train, y_train)
bag_pred = bag_model.predict(X_val)
additional_models["Bagging"] = bag_pred
print("   ✅ Bagging completado")

# 4. Huber Regressor (robusto a outliers)
print("4️⃣  Entrenando Huber Regressor...")
huber_model = HuberRegressor(max_iter=1000)
huber_model.fit(X_train, y_train)
huber_pred = huber_model.predict(X_val)
additional_models["Huber"] = huber_pred
print("   ✅ Huber completado")

# 5. RANSAC (robusto a outliers)
print("5️⃣  Entrenando RANSAC Regressor...")
try:
    ransac_model = RANSACRegressor(random_state=13, max_trials=100)
    ransac_model.fit(X_train, y_train)
    ransac_pred = ransac_model.predict(X_val)
    additional_models["RANSAC"] = ransac_pred
    print("   ✅ RANSAC completado")
except:
    print("   ❌ RANSAC falló, continuando...")

# 6. Theil-Sen (robusto a outliers)
print("6️⃣  Entrenando Theil-Sen Regressor...")
try:
    # Usar una muestra más pequeña si el dataset es muy grande
    sample_size = min(1000, len(X_train))
    theil_model = TheilSenRegressor(random_state=13, max_subpopulation=sample_size)
    theil_model.fit(X_train, y_train)
    theil_pred = theil_model.predict(X_val)
    additional_models["Theil-Sen"] = theil_pred
    print("   ✅ Theil-Sen completado")
except:
    print("   ❌ Theil-Sen falló (dataset muy grande), continuando...")

print(f"\n✨ Modelos adicionales entrenados: {len(additional_models)}")

# CROSS-VALIDATION Y OPTIMIZACIÓN DE HIPERPARÁMETROS
print("\n" + "="*80)
print("🎯 OPTIMIZACIÓN DE HIPERPARÁMETROS CON CROSS-VALIDATION")
print("="*80)

# Función para evaluar con CV
def evaluate_with_cv(model, params, name, cv_folds=5):
    print(f"\n🔍 Optimizando {name}...")
    print(f"   Parámetros a probar: {len(list(params.values())[0] if params else 1)} combinaciones")
    
    start_time = time.time()
    
    if params:
        # Grid Search con CV
        grid_search = GridSearchCV(
            model, 
            params, 
            cv=cv_folds, 
            scoring='r2', 
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        # Mejor modelo
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_
    else:
        # Solo CV sin optimización
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
        cv_score = cv_scores.mean()
        best_model = model
        best_params = {}
    
    # Predecir en validation
    predictions = best_model.predict(X_val)
    val_r2 = r2_score(y_val, predictions)
    
    elapsed_time = time.time() - start_time
    
    print(f"   ✅ {name} completado en {elapsed_time:.1f}s")
    print(f"   📊 CV R²: {cv_score:.4f}, Val R²: {val_r2:.4f}")
    if best_params:
        print(f"   🎯 Mejores parámetros: {best_params}")
    
    return {
        'name': name,
        'model': best_model,
        'predictions': predictions,
        'cv_score': cv_score,
        'val_r2': val_r2,
        'best_params': best_params,
        'time': elapsed_time
    }

# Configuraciones para optimizar
optimization_configs = [
    # Random Forest
    (
        RandomForestRegressor(random_state=13),
        {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        },
        "Random Forest Optimizado"
    ),
    
    # Gradient Boosting
    (
        GradientBoostingRegressor(random_state=13),
        {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        "Gradient Boosting Optimizado"
    ),
    
    # SVR
    # (
    #     SVR(),
    #     {
    #         'kernel': ['rbf', 'poly'],
    #         'C': [0.1, 1, 10],
    #         'gamma': ['scale', 'auto']
    #     },
    #     "SVR Optimizado"
    # ),
    
    # KNN
    (
        KNeighborsRegressor(),
        {
            'n_neighbors': [3, 5, 7, 10, 15],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        },
        "KNN Optimizado"
    ),
    
    # Ridge
    (
        Ridge(random_state=13),
        {
            'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
        },
        "Ridge Optimizado"
    ),
    
    # Lasso
    (
        Lasso(random_state=13, max_iter=2000),
        {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        },
        "Lasso Optimizado"
    ),
    
    # ElasticNet
    (
        ElasticNet(random_state=13, max_iter=2000),
        {
            'alpha': [0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9]
        },
        "ElasticNet Optimizado"
    )
]

# Ejecutar optimizaciones
optimized_results = []
total_configs = len(optimization_configs)

for i, (model, params, name) in enumerate(optimization_configs):
    print(f"\n{'='*20} {i+1}/{total_configs}: {name} {'='*20}")
    
    try:
        result = evaluate_with_cv(model, params, name, cv_folds=5)
        optimized_results.append(result)
    except Exception as e:
        print(f"   ❌ Error en {name}: {e}")
        continue

print(f"\n🎉 Optimización completada para {len(optimized_results)} modelos")

# ENSEMBLE METHODS
print("\n" + "="*80)
print("🤝 CREANDO MODELOS ENSEMBLE")
print("="*80)

# Seleccionar los mejores modelos para ensemble
print("1️⃣  Preparando modelos base para ensemble...")
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=13)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=13)),
    ('svr', SVR(kernel='rbf', C=1.0))
]

# Voting Regressor
print("2️⃣  Entrenando Voting Regressor...")
voting_model = VotingRegressor(base_models)
voting_model.fit(X_train, y_train)
voting_pred = voting_model.predict(X_val)
print("   ✅ Voting Regressor completado")

# Stacking Regressor
print("3️⃣  Entrenando Stacking Regressor...")
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=LinearRegression(),
    cv=5
)
stacking_model.fit(X_train, y_train)
stacking_pred = stacking_model.predict(X_val)
print("   ✅ Stacking Regressor completado")

# RESULTADOS FINALES COMPLETOS
print("\n" + "="*100)
print("🏆 RESULTADOS FINALES - TODOS LOS MODELOS")
print("="*100)

# Combinar todos los resultados
all_final_results = []

# Modelos originales
for model_name, predictions in models_results:
    r2_temp = r2_score(y_val, predictions)
    mse_temp = mean_squared_error(y_val, predictions)
    rmse_temp = root_mean_squared_error(y_val, predictions)
    mae_temp = mean_absolute_error(y_val, predictions)
    mape_temp = mean_absolute_percentage_error(y_val, predictions)
    
    all_final_results.append({
        'name': model_name,
        'type': 'Original',
        'r2': r2_temp,
        'mse': mse_temp,
        'rmse': rmse_temp,
        'mae': mae_temp,
        'mape': mape_temp
    })

# Modelos adicionales
for model_name, predictions in additional_models.items():
    r2_temp = r2_score(y_val, predictions)
    mse_temp = mean_squared_error(y_val, predictions)
    rmse_temp = root_mean_squared_error(y_val, predictions)
    mae_temp = mean_absolute_error(y_val, predictions)
    mape_temp = mean_absolute_percentage_error(y_val, predictions)
    
    all_final_results.append({
        'name': model_name,
        'type': 'Adicional',
        'r2': r2_temp,
        'mse': mse_temp,
        'rmse': rmse_temp,
        'mae': mae_temp,
        'mape': mape_temp
    })

# Modelos optimizados
for result in optimized_results:
    all_final_results.append({
        'name': result['name'],
        'type': 'Optimizado',
        'r2': result['val_r2'],
        'mse': mean_squared_error(y_val, result['predictions']),
        'rmse': root_mean_squared_error(y_val, result['predictions']),
        'mae': mean_absolute_error(y_val, result['predictions']),
        'mape': mean_absolute_percentage_error(y_val, result['predictions'])
    })

# Redes neuronales
for result in nn_results:
    all_final_results.append({
        'name': result['name'],
        'type': 'Red Neuronal',
        'r2': result['r2'],
        'mse': result['mse'],
        'rmse': result['rmse'],
        'mae': result['mae'],
        'mape': result['mape']
    })

# Ensemble models
ensemble_models = [
    ("Voting Regressor", voting_pred),
    ("Stacking Regressor", stacking_pred)
]

for model_name, predictions in ensemble_models:
    r2_temp = r2_score(y_val, predictions)
    mse_temp = mean_squared_error(y_val, predictions)
    rmse_temp = root_mean_squared_error(y_val, predictions)
    mae_temp = mean_absolute_error(y_val, predictions)
    mape_temp = mean_absolute_percentage_error(y_val, predictions)
    
    all_final_results.append({
        'name': model_name,
        'type': 'Ensemble',
        'r2': r2_temp,
        'mse': mse_temp,
        'rmse': rmse_temp,
        'mae': mae_temp,
        'mape': mape_temp
    })

# Ordenar por R² (mejor a peor)
all_final_results.sort(key=lambda x: x['r2'], reverse=True)

print("Pos Tipo           Modelo                    R²        MSE        RMSE       MAE        MAPE")
print("-" * 100)

for i, result in enumerate(all_final_results[:20]):  # Top 20
    medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
    print(f"{medal} {result['type']:<12} {result['name']:<22} {result['r2']:>8.4f} {result['mse']:>10.4f} "
          f"{result['rmse']:>10.4f} {result['mae']:>10.4f} {result['mape']:>10.4f}")

print("-" * 100)
print(f"🏆 CAMPEÓN ABSOLUTO: {all_final_results[0]['name']} ({all_final_results[0]['type']})")
print(f"   R² = {all_final_results[0]['r2']:.6f}")

# Estadísticas por tipo
print(f"\n📈 ESTADÍSTICAS POR TIPO DE MODELO:")
model_types = {}
for result in all_final_results:
    model_type = result['type']
    if model_type not in model_types:
        model_types[model_type] = []
    model_types[model_type].append(result['r2'])

for model_type, r2_scores in model_types.items():
    avg_r2 = sum(r2_scores) / len(r2_scores)
    max_r2 = max(r2_scores)
    print(f"   {model_type}: Promedio R² = {avg_r2:.4f}, Mejor R² = {max_r2:.4f}, Modelos: {len(r2_scores)}")

print(f"\n🎯 TOTAL DE MODELOS PROBADOS: {len(all_final_results)}")
print(f"✨ Proceso completado exitosamente!")



#El mejor modelo fue gradient boosting, probamos muchos hiperparametros por ahi
print("\n" + "="*80)
print("🔍 GRID SEARCH EXHAUSTIVO PARA GRADIENT BOOSTING")
print("="*80)

# Definir una grilla muy completa de hiperparámetros
gb_param_grid = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_features': ['sqrt', 'log2', None, 0.5, 0.7],
    'subsample': [0.8, 0.85, 0.9, 0.95, 1.0],
    'loss': ['squared_error', 'absolute_error', 'huber']
}

total_combinations = (len(gb_param_grid['n_estimators']) * len(gb_param_grid['learning_rate']) * 
                     len(gb_param_grid['max_depth']) * len(gb_param_grid['min_samples_split']) * 
                     len(gb_param_grid['min_samples_leaf']) * len(gb_param_grid['max_features']) * 
                     len(gb_param_grid['subsample']) * len(gb_param_grid['loss']))

print(f"🔢 Combinaciones totales a probar: {total_combinations:,}")
print("⚠️  Esto puede tomar mucho tiempo. Usando paralelización...")

# Crear el modelo base
gb_base = GradientBoostingRegressor(random_state=13, validation_fraction=0.1, n_iter_no_change=10)

# Clase personalizada para mostrar progreso
class VerboseGridSearchCV(GridSearchCV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.combination_count = 0
        self.total_combinations = 0
        
    def fit(self, X, y=None):
        # Calcular total de combinaciones
        self.total_combinations = len(list(ParameterGrid(self.param_grid)))
        print(f"📊 Total de combinaciones a evaluar: {self.total_combinations}")
        print(f"📊 Con {self.cv} folds de CV = {self.total_combinations * self.cv} entrenamientos")
        print("\n🚀 Iniciando evaluación de hiperparámetros...")
        print("-" * 80)
        
        return super().fit(X, y)

# Grid Search con CV y progreso personalizado
print("\n🚀 Iniciando Grid Search...")
start_time = time.time()

# Crear un callback para mostrar progreso
class ProgressCallback:
    def __init__(self, total_combinations):
        self.total_combinations = total_combinations
        self.completed = 0
        self.best_score = -float('inf')
        self.best_params = None
        
    def update(self, score, params):
        self.completed += 1
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
            
        if self.completed % 50 == 0 or score > self.best_score:
            percentage = (self.completed / self.total_combinations) * 100
            print(f"📈 Progreso: {self.completed}/{self.total_combinations} ({percentage:.1f}%) - "
                  f"Mejor R²: {self.best_score:.4f}")
            if score > self.best_score - 0.001:  # Mostrar parámetros prometedores
                print(f"   Parámetros actuales: {params}")

gb_grid_search = GridSearchCV(
    estimator=gb_base,
    param_grid=gb_param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,  # Usar todos los procesadores disponibles
    verbose=2,  # Nivel máximo de verbosidad
    return_train_score=True
)

# Mostrar parámetros que se van a probar
print(f"\n📋 RANGO DE HIPERPARÁMETROS A PROBAR:")
for param, values in gb_param_grid.items():
    print(f"   {param}: {values}")

print(f"\n⏰ Hora de inicio: {time.strftime('%H:%M:%S')}")
print("🔄 Iniciando búsqueda exhaustiva...")

# Entrenar con monitoreo de progreso
gb_grid_search.fit(X_train, y_train)

elapsed_time = time.time() - start_time
print(f"\n⏱️  Grid Search completado en {elapsed_time/60:.1f} minutos")
print(f"⏰ Hora de finalización: {time.strftime('%H:%M:%S')}")

# Obtener el mejor modelo
best_gb_model = gb_grid_search.best_estimator_
best_gb_params = gb_grid_search.best_params_
best_cv_score = gb_grid_search.best_score_

print(f"\n🏆 MEJORES HIPERPARÁMETROS ENCONTRADOS:")
for param, value in best_gb_params.items():
    print(f"   {param}: {value}")

print(f"\n📊 MEJOR SCORE CV: {best_cv_score:.6f}")

# Mostrar progreso de los mejores resultados durante la búsqueda
print(f"\n📈 HISTÓRICO DE MEJORES RESULTADOS:")
results_df = pd.DataFrame(gb_grid_search.cv_results_)
results_df_sorted = results_df.sort_values('mean_test_score', ascending=False)

print("Rank  R² Score    Parámetros")
print("-" * 100)
for i, (idx, row) in enumerate(results_df_sorted.head(10).iterrows()):
    params_short = {k.replace('param_', ''): v for k, v in row.items() if k.startswith('param_')}
    params_str = ", ".join([f"{k}:{v}" for k, v in list(params_short.items())[:4]])  # Primeros 4 parámetros
    print(f"{i+1:4d}  {row['mean_test_score']:8.4f}    {params_str}...")

# Evaluar en el conjunto de validación
best_gb_pred = best_gb_model.predict(X_val)
best_gb_r2 = r2_score(y_val, best_gb_pred)
best_gb_mse = mean_squared_error(y_val, best_gb_pred)
best_gb_rmse = root_mean_squared_error(y_val, best_gb_pred)
best_gb_mae = mean_absolute_error(y_val, best_gb_pred)
best_gb_mape = mean_absolute_percentage_error(y_val, best_gb_pred)

print(f"\n🎯 RESULTADOS EN CONJUNTO DE VALIDACIÓN:")
print(f"   R²: {best_gb_r2:.6f}")
print(f"   MSE: {best_gb_mse:.6f}")
print(f"   RMSE: {best_gb_rmse:.6f}")
print(f"   MAE: {best_gb_mae:.6f}")
print(f"   MAPE: {best_gb_mape:.6f}")

# Comparar con el modelo original
original_gb = GradientBoostingRegressor(n_estimators=100, random_state=13, max_depth=6)
original_gb.fit(X_train, y_train)
original_gb_pred = original_gb.predict(X_val)
original_gb_r2 = r2_score(y_val, original_gb_pred)

improvement = ((best_gb_r2 - original_gb_r2) / abs(original_gb_r2)) * 100
print(f"\n📈 MEJORA RESPECTO AL MODELO ORIGINAL:")
print(f"   Modelo original R²: {original_gb_r2:.6f}")
print(f"   Modelo optimizado R²: {best_gb_r2:.6f}")
print(f"   Mejora: {improvement:.2f}%")

# Información adicional del modelo
print(f"\n🔍 INFORMACIÓN ADICIONAL DEL MEJOR MODELO:")
print(f"   Número de árboles: {best_gb_model.n_estimators}")
print(f"   Tasa de aprendizaje: {best_gb_model.learning_rate}")
print(f"   Profundidad máxima: {best_gb_model.max_depth}")
print(f"   Función de pérdida: {best_gb_model.loss}")
print(f"   Submuestreo: {best_gb_model.subsample}")
print(f"   Min samples split: {best_gb_model.min_samples_split}")
print(f"   Min samples leaf: {best_gb_model.min_samples_leaf}")
print(f"   Max features: {best_gb_model.max_features}")

# Feature importance del mejor modelo
print(f"\n🎯 TOP 15 FEATURES MÁS IMPORTANTES (Mejor Gradient Boosting):")
feature_names = X_train.columns
gb_importances = best_gb_model.feature_importances_
top_features = sorted(zip(feature_names, gb_importances), key=lambda x: x[1], reverse=True)[:15]
for i, (feature, importance) in enumerate(top_features):
    print(f"   {i+1:2d}. {feature}: {importance:.4f}")

# Análisis de los resultados del Grid Search
print(f"\n📊 ANÁLISIS DE RESULTADOS DEL GRID SEARCH:")
results_df = pd.DataFrame(gb_grid_search.cv_results_)

print(f"   Total de combinaciones probadas: {len(results_df)}")
print(f"   Mejor ranking: {results_df['rank_test_score'].min()}")
print(f"   Score promedio: {results_df['mean_test_score'].mean():.4f}")
print(f"   Desviación estándar promedio: {results_df['std_test_score'].mean():.4f}")
print(f"   Tiempo promedio por combinación: {elapsed_time/len(results_df):.2f} segundos")

# Top 10 combinaciones con más detalle
print(f"\n🏅 TOP 10 MEJORES COMBINACIONES:")
top_10_results = results_df.nlargest(10, 'mean_test_score')
for i, (idx, row) in enumerate(top_10_results.iterrows()):
    print(f"\n   {i+1:2d}. R² = {row['mean_test_score']:.6f} (±{row['std_test_score']:.4f})")
    print(f"       Tiempo: {row['mean_fit_time']:.2f}s")
    
    # Mostrar todos los parámetros
    params = {k.replace('param_', ''): v for k, v in row.items() if k.startswith('param_')}
    for param, value in params.items():
        print(f"       {param}: {value}")

# Análisis de sensibilidad de parámetros
print(f"\n🔍 ANÁLISIS DE SENSIBILIDAD DE PARÁMETROS:")
for param in gb_param_grid.keys():
    param_col = f'param_{param}'
    if param_col in results_df.columns:
        param_analysis = results_df.groupby(param_col)['mean_test_score'].agg(['mean', 'std', 'count'])
        print(f"\n   {param}:")
        for value, stats in param_analysis.iterrows():
            print(f"      {value}: R² = {stats['mean']:.4f} (±{stats['std']:.4f}, n={stats['count']})")

# Guardar el mejor modelo para uso futuro
print(f"\n💾 El mejor modelo Gradient Boosting ha sido entrenado y está listo para usar")
print(f"   Acceder al modelo: best_gb_model")
print(f"   Predicciones: best_gb_pred")

# Comparación final con todos los modelos anteriores
print(f"\n🏆 COMPARACIÓN CON TODOS LOS MODELOS ANTERIORES:")
print(f"   Mejor modelo anterior: {all_final_results[0]['name']} (R² = {all_final_results[0]['r2']:.6f})")
print(f"   Gradient Boosting optimizado: R² = {best_gb_r2:.6f}")

if best_gb_r2 > all_final_results[0]['r2']:
    final_improvement = ((best_gb_r2 - all_final_results[0]['r2']) / abs(all_final_results[0]['r2'])) * 100
    print(f"   🎉 ¡NUEVO CAMPEÓN! Mejora de {final_improvement:.2f}%")
else:
    print(f"   📊 El modelo anterior sigue siendo mejor")

print(f"\n✨ Grid Search exhaustivo completado exitosamente!")
print(f"📊 Estadísticas finales:")
print(f"   - Tiempo total: {elapsed_time/60:.1f} minutos")
print(f"   - Combinaciones probadas: {len(results_df):,}")
print(f"   - Entrenamientos realizados: {len(results_df) * 5:,}")
print(f"   - Mejor R² encontrado: {best_cv_score:.6f}")
