# Regresion lineal
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
# Importar más modelos
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
import numpy as np


# Agregar después del código existente
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.linear_model import HuberRegressor, TheilSenRegressor, RANSACRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid


ds_14 = pd.read_csv('data/processed/dataset_14.csv')



#Elimino las columnas fecha y fecha y hora
ds_14 = ds_14.drop(columns=['fecha', 'fecha_hora'])


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

#El mejor modelo fue gradient boosting, probamos muchos hiperparametros por ahi
print("\n" + "="*80)
print("🔍 GRID SEARCH EXHAUSTIVO PARA GRADIENT BOOSTING")
print("="*80)

# Definir una grilla muy completa de hiperparámetros
gb_param_grid = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'learning_rate': [0.01, 0.05],
    'max_depth': [3, 4, 5],
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


# Combinar todos los resultados
all_final_results = []


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