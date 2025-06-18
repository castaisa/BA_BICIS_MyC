# Clasificaci贸n con m煤ltiples modelos
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np

ds_14 = pd.read_csv('data/processed/dataset_14.csv')

# Elimino las columnas fecha y fecha y hora
ds_14 = ds_14.drop(columns=['fecha', 'fecha_hora'])

# Convertir target a clases de clasificaci贸n (0-11, y 12+ = clase 12)
ds_14['target'] = ds_14['target'].astype(int)
ds_14['target'] = ds_14['target'].apply(lambda x: min(x, 12))  # Valores >11 se convierten en clase 12

print(f"Distribuci贸n de clases:")
print(ds_14['target'].value_counts().sort_index())

# Hago PCA y grafico las dos features mas significativas
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(ds_14.drop(columns=['target']))
plt.figure(figsize=(10, 6))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=ds_14['target'], cmap='tab20', alpha=0.5)
plt.colorbar(scatter, label='Target Class')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA de las dos primeras componentes principales (Clasificaci贸n)')
plt.show()

# Le hago one hot encoding a la columna hora
ds_14 = pd.get_dummies(ds_14, columns=['hora'], prefix='hora', drop_first=True, dtype='int64')

print(ds_14.head())

# Separar features (X) y target (y)
ds_y = ds_14['target']
ds_X = ds_14.drop(columns=['target'])

# Dividir en train y validation (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    ds_X, ds_y, 
    test_size=0.2, 
    random_state=13,
    shuffle=False,
    stratify=None  # No stratify para mantener orden temporal
)

print(f"Dataset completo: {len(ds_14)} muestras")
print(f"Train: {len(X_train)} muestras")
print(f"Validation: {len(X_val)} muestras")
print(f"Features: {X_train.shape[1]}")
print(f"Clases 煤nicas: {sorted(ds_y.unique())}")

# # ENTRENAMOS LOS MODELOS CON TRAIN
# model = LogisticRegression(random_state=13, max_iter=1000)
# model.fit(X_train, y_train)

# Ridge Classifier
ridge_model = RidgeClassifier(alpha=1000, random_state=13)
ridge_model.fit(X_train, y_train)

# # Realizamos las predicciones con el set de validaci贸n
# y_pred = model.predict(X_val)

# Realizamos las predicciones con el modelo de Ridge
ridge_y_pred = ridge_model.predict(X_val)

# # Evaluamos el modelo de Logistic Regression
# accuracy = accuracy_score(y_val, y_pred)
# f1 = f1_score(y_val, y_pred, average='weighted')
# precision = precision_score(y_val, y_pred, average='weighted')
# recall = recall_score(y_val, y_pred, average='weighted')

# Evaluamos el modelo de Ridge Classifier
ridge_accuracy = accuracy_score(y_val, ridge_y_pred)
ridge_f1 = f1_score(y_val, ridge_y_pred, average='weighted')
ridge_precision = precision_score(y_val, ridge_y_pred, average='weighted')
ridge_recall = recall_score(y_val, ridge_y_pred, average='weighted')

# print("Resultados del modelo de Logistic Regression:")
# print(f"Accuracy: {accuracy:.4f}")
# print(f"F1-Score (weighted): {f1:.4f}")
# print(f"Precision (weighted): {precision:.4f}")
# print(f"Recall (weighted): {recall:.4f}")

print("\nResultados del modelo de Ridge Classifier:")
print(f"Accuracy: {ridge_accuracy:.4f}")
print(f"F1-Score (weighted): {ridge_f1:.4f}")
print(f"Precision (weighted): {ridge_precision:.4f}")
print(f"Recall (weighted): {ridge_recall:.4f}")

# Importar m谩s modelos de clasificaci贸n
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Entrenar modelos adicionales
print("\n" + "="*50)
print("ENTRENANDO MODELOS ADICIONALES...")
print("="*50)

# Decision Tree
tree_model = DecisionTreeClassifier(random_state=13, max_depth=10)
tree_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=13, max_depth=10)
rf_model.fit(X_train, y_train)

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=13, max_depth=6)
gb_model.fit(X_train, y_train)

# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Support Vector Classifier
svc_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=13)
svc_model.fit(X_train, y_train)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Linear Discriminant Analysis
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)

# Quadratic Discriminant Analysis
try:
    qda_model = QuadraticDiscriminantAnalysis()
    qda_model.fit(X_train, y_train)
    qda_available = True
except:
    print("QDA no disponible (problemas de covarianza)")
    qda_available = False

# Predicciones
tree_pred = tree_model.predict(X_val)
rf_pred = rf_model.predict(X_val)
gb_pred = gb_model.predict(X_val)
knn_pred = knn_model.predict(X_val)
svc_pred = svc_model.predict(X_val)
nb_pred = nb_model.predict(X_val)
lda_pred = lda_model.predict(X_val)

if qda_available:
    qda_pred = qda_model.predict(X_val)

# Evaluaciones
models_results = [
    #("Logistic Regression", y_pred),
    ("Ridge Classifier", ridge_y_pred),
    ("Decision Tree", tree_pred),
    ("Random Forest", rf_pred),
    ("Gradient Boosting", gb_pred),
    ("K-NN", knn_pred),
    ("SVC", svc_pred),
    ("Naive Bayes", nb_pred),
    ("LDA", lda_pred)
]

if qda_available:
    models_results.append(("QDA", qda_pred))

print("\n" + "="*90)
print("COMPARACI脫N DE TODOS LOS MODELOS")
print("="*90)
print("Modelo                Accuracy   F1-Score   Precision  Recall")
print("-" * 70)

best_accuracy = -float('inf')
best_model_name = ""

for model_name, predictions in models_results:
    acc_temp = accuracy_score(y_val, predictions)
    f1_temp = f1_score(y_val, predictions, average='weighted')
    prec_temp = precision_score(y_val, predictions, average='weighted')
    rec_temp = recall_score(y_val, predictions, average='weighted')
    
    print(f"{model_name:<18} {acc_temp:>8.4f} {f1_temp:>10.4f} {prec_temp:>10.4f} {rec_temp:>10.4f}")
    
    if acc_temp > best_accuracy:
        best_accuracy = acc_temp
        best_model_name = model_name

print("-" * 70)
print(f"馃弳 MEJOR MODELO: {best_model_name} (Accuracy = {best_accuracy:.4f})")

# Informaci贸n adicional sobre algunos modelos
print("\n" + "="*50)
print("INFORMACI脫N ADICIONAL")
print("="*50)

# Feature importance para Random Forest
print("\nTop 10 features m谩s importantes (Random Forest):")
feature_names = X_train.columns
rf_importances = rf_model.feature_importances_
top_features = sorted(zip(feature_names, rf_importances), key=lambda x: x[1], reverse=True)[:10]
for feature, importance in top_features:
    print(f"  {feature}: {importance:.4f}")

# Ahora probamos una red neuronal
from sklearn.neural_network import MLPClassifier

# Entrenar una red neuronal
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=13)
nn_model.fit(X_train, y_train)

# Predicciones con la red neuronal
nn_y_pred = nn_model.predict(X_val)

# Evaluaci贸n de la red neuronal
nn_accuracy = accuracy_score(y_val, nn_y_pred)
nn_f1 = f1_score(y_val, nn_y_pred, average='weighted')
nn_precision = precision_score(y_val, nn_y_pred, average='weighted')
nn_recall = recall_score(y_val, nn_y_pred, average='weighted')

print("\nResultados del modelo de Red Neuronal:")
print(f"Accuracy: {nn_accuracy:.4f}")
print(f"F1-Score (weighted): {nn_f1:.4f}")
print(f"Precision (weighted): {nn_precision:.4f}")
print(f"Recall (weighted): {nn_recall:.4f}")

# Agregar la red neuronal a los resultados
models_results.append(("Red Neuronal", nn_y_pred))

# Recalcular el mejor modelo
best_accuracy = -float('inf')
best_model_name = ""

for model_name, predictions in models_results:
    acc_temp = accuracy_score(y_val, predictions)
    
    if acc_temp > best_accuracy:
        best_accuracy = acc_temp
        best_model_name = model_name
        
print(f"\n馃弳 MEJOR MODELO: {best_model_name} (Accuracy = {best_accuracy:.4f})")

# M煤ltiples configuraciones de redes neuronales
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
    ("NN-7 SGD", (100, 50), 1000, 'constant', 0.0001, 'sgd'),
    ("NN-8 Adaptativo", (80, 60, 40), 1000, 'adaptive', 0.001, 'adam'),
    ("NN-9 Peque帽a", (30, 20), 800, 'constant', 0.0001, 'adam'),
    ("NN-10 Mega", (150, 100, 75, 50), 2000, 'invscaling', 0.0005, 'adam')
]

nn_results = []

for i, (name, hidden_layers, max_iter, learning_rate, alpha, solver) in enumerate(nn_configs):
    print(f"Entrenando {name}... ", end="")
    
    try:
        # Configurar learning rate inicial
        learning_rate_init = 0.001 if solver == 'adam' else 0.01
        if solver == 'sgd':
            learning_rate_init = 0.1
            
        nn_temp = MLPClassifier(
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
        
        # M茅tricas
        nn_acc_temp = accuracy_score(y_val, nn_pred_temp)
        nn_f1_temp = f1_score(y_val, nn_pred_temp, average='weighted')
        nn_prec_temp = precision_score(y_val, nn_pred_temp, average='weighted')
        nn_rec_temp = recall_score(y_val, nn_pred_temp, average='weighted')
        
        nn_results.append({
            'name': name,
            'model': nn_temp,
            'predictions': nn_pred_temp,
            'accuracy': nn_acc_temp,
            'f1': nn_f1_temp,
            'precision': nn_prec_temp,
            'recall': nn_rec_temp,
            'layers': hidden_layers,
            'iterations': nn_temp.n_iter_,
            'solver': solver,
            'alpha': alpha
        })
        
        print(f"鉁� Accuracy = {nn_acc_temp:.4f}")
        
    except Exception as e:
        print(f"鉂� Error: {e}")
        continue

# Mostrar resultados de todas las redes neuronales
print("\n" + "="*100)
print("RESULTADOS DE TODAS LAS REDES NEURONALES")
print("="*100)
print("Modelo            Capas              Accuracy   F1-Score   Precision  Recall     Iter")
print("-" * 95)

for result in nn_results:
    layers_str = str(result['layers']).replace(' ', '')
    print(f"{result['name']:<13} {layers_str:<18} {result['accuracy']:>8.4f} "
          f"{result['f1']:>10.4f} {result['precision']:>10.4f} {result['recall']:>10.4f} {result['iterations']:>6}")

# Encontrar la mejor red neuronal
if nn_results:
    best_nn = max(nn_results, key=lambda x: x['accuracy'])
    print("-" * 95)
    print(f"馃弳 MEJOR RED NEURONAL: {best_nn['name']} (Accuracy = {best_nn['accuracy']:.4f})")
    print(f"   Arquitectura: {best_nn['layers']}")
    print(f"   Solver: {best_nn['solver']}, Alpha: {best_nn['alpha']}")
    print(f"   Iteraciones: {best_nn['iterations']}")

# Agregar todas las redes neuronales a los resultados generales
for result in nn_results:
    models_results.append((result['name'], result['predictions']))

# Modelos adicionales
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, Perceptron
from sklearn.gaussian_process import GaussianProcessClassifier

print("\n" + "="*80)
print("馃攳 AGREGANDO M脕S MODELOS DE CLASIFICACI脫N")
print("="*80)

additional_models = {}

# 1. Extra Trees
print("1锔忊儯  Entrenando Extra Trees Classifier...")
extra_model = ExtraTreesClassifier(n_estimators=100, random_state=13, max_depth=10)
extra_model.fit(X_train, y_train)
extra_pred = extra_model.predict(X_val)
additional_models["Extra Trees"] = extra_pred
print("   鉁� Extra Trees completado")

# 2. AdaBoost
print("2锔忊儯  Entrenando AdaBoost Classifier...")
ada_model = AdaBoostClassifier(n_estimators=100, random_state=13)
ada_model.fit(X_train, y_train)
ada_pred = ada_model.predict(X_val)
additional_models["AdaBoost"] = ada_pred
print("   鉁� AdaBoost completado")

# 3. Bagging
print("3锔忊儯  Entrenando Bagging Classifier...")
bag_model = BaggingClassifier(n_estimators=100, random_state=13)
bag_model.fit(X_train, y_train)
bag_pred = bag_model.predict(X_val)
additional_models["Bagging"] = bag_pred
print("   鉁� Bagging completado")

# 4. Passive Aggressive
print("4锔忊儯  Entrenando Passive Aggressive Classifier...")
pa_model = PassiveAggressiveClassifier(random_state=13, max_iter=1000)
pa_model.fit(X_train, y_train)
pa_pred = pa_model.predict(X_val)
additional_models["Passive Aggressive"] = pa_pred
print("   鉁� Passive Aggressive completado")

# 5. Perceptron
print("5锔忊儯  Entrenando Perceptron...")
perc_model = Perceptron(random_state=13, max_iter=1000)
perc_model.fit(X_train, y_train)
perc_pred = perc_model.predict(X_val)
additional_models["Perceptron"] = perc_pred
print("   鉁� Perceptron completado")

print(f"\n鉁� Modelos adicionales entrenados: {len(additional_models)}")

# Cross-validation y optimizaci贸n
from sklearn.model_selection import GridSearchCV, cross_val_score
import time

print("\n" + "="*80)
print("馃幆 OPTIMIZACI脫N DE HIPERPAR脕METROS CON CROSS-VALIDATION")
print("="*80)

def evaluate_with_cv(model, params, name, cv_folds=5):
    print(f"\n馃攳 Optimizando {name}...")
    
    start_time = time.time()
    
    if params:
        grid_search = GridSearchCV(
            model, 
            params, 
            cv=cv_folds, 
            scoring='accuracy', 
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
        cv_score = cv_scores.mean()
        best_model = model
        best_params = {}
    
    predictions = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, predictions)
    
    elapsed_time = time.time() - start_time
    
    print(f"   鉁� {name} completado en {elapsed_time:.1f}s")
    print(f"   馃搳 CV Accuracy: {cv_score:.4f}, Val Accuracy: {val_accuracy:.4f}")
    if best_params:
        print(f"   馃幆 Mejores par谩metros: {best_params}")
    
    return {
        'name': name,
        'model': best_model,
        'predictions': predictions,
        'cv_score': cv_score,
        'val_accuracy': val_accuracy,
        'best_params': best_params,
        'time': elapsed_time
    }

# Configuraciones para optimizar
optimization_configs = [
    (
        RandomForestClassifier(random_state=13),
        {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        },
        "Random Forest Optimizado"
    ),
    
    (
        GradientBoostingClassifier(random_state=13),
        {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        "Gradient Boosting Optimizado"
    ),
    
    (
        KNeighborsClassifier(),
        {
            'n_neighbors': [3, 5, 7, 10, 15],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        },
        "KNN Optimizado"
    ),
    
    (
        LogisticRegression(random_state=13, max_iter=2000),
        {
            'C': [0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        },
        "Logistic Regression Optimizado"
    )
]

# Ejecutar optimizaciones
optimized_results = []

for i, (model, params, name) in enumerate(optimization_configs):
    print(f"\n{'='*20} {i+1}/{len(optimization_configs)}: {name} {'='*20}")
    
    try:
        result = evaluate_with_cv(model, params, name, cv_folds=5)
        optimized_results.append(result)
    except Exception as e:
        print(f"   鉂� Error en {name}: {e}")
        continue

# Ensemble methods
from sklearn.ensemble import VotingClassifier, StackingClassifier

print("\n" + "="*80)
print("馃 CREANDO MODELOS ENSEMBLE")
print("="*80)

print("1锔忊儯  Preparando modelos base para ensemble...")
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=13)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=13)),
    ('svc', SVC(kernel='rbf', C=1.0, probability=True, random_state=13))
]

# Voting Classifier
print("2锔忊儯  Entrenando Voting Classifier...")
voting_model = VotingClassifier(base_models, voting='soft')
voting_model.fit(X_train, y_train)
voting_pred = voting_model.predict(X_val)
print("   鉁� Voting Classifier completado")

# Stacking Classifier
print("3锔忊儯  Entrenando Stacking Classifier...")
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(random_state=13),
    cv=5
)
stacking_model.fit(X_train, y_train)
stacking_pred = stacking_model.predict(X_val)
print("   鉁� Stacking Classifier completado")

# RESULTADOS FINALES COMPLETOS
print("\n" + "="*100)
print("馃弳 RESULTADOS FINALES - TODOS LOS MODELOS")
print("="*100)

all_final_results = []

# Modelos originales
for model_name, predictions in models_results:
    acc_temp = accuracy_score(y_val, predictions)
    f1_temp = f1_score(y_val, predictions, average='weighted')
    prec_temp = precision_score(y_val, predictions, average='weighted')
    rec_temp = recall_score(y_val, predictions, average='weighted')
    
    all_final_results.append({
        'name': model_name,
        'type': 'Original',
        'accuracy': acc_temp,
        'f1': f1_temp,
        'precision': prec_temp,
        'recall': rec_temp
    })

# Modelos adicionales
for model_name, predictions in additional_models.items():
    acc_temp = accuracy_score(y_val, predictions)
    f1_temp = f1_score(y_val, predictions, average='weighted')
    prec_temp = precision_score(y_val, predictions, average='weighted')
    rec_temp = recall_score(y_val, predictions, average='weighted')
    
    all_final_results.append({
        'name': model_name,
        'type': 'Adicional',
        'accuracy': acc_temp,
        'f1': f1_temp,
        'precision': prec_temp,
        'recall': rec_temp
    })

# Modelos optimizados
for result in optimized_results:
    all_final_results.append({
        'name': result['name'],
        'type': 'Optimizado',
        'accuracy': result['val_accuracy'],
        'f1': f1_score(y_val, result['predictions'], average='weighted'),
        'precision': precision_score(y_val, result['predictions'], average='weighted'),
        'recall': recall_score(y_val, result['predictions'], average='weighted')
    })

# Redes neuronales
for result in nn_results:
    all_final_results.append({
        'name': result['name'],
        'type': 'Red Neuronal',
        'accuracy': result['accuracy'],
        'f1': result['f1'],
        'precision': result['precision'],
        'recall': result['recall']
    })

# Ensemble models
ensemble_models = [
    ("Voting Classifier", voting_pred),
    ("Stacking Classifier", stacking_pred)
]

for model_name, predictions in ensemble_models:
    acc_temp = accuracy_score(y_val, predictions)
    f1_temp = f1_score(y_val, predictions, average='weighted')
    prec_temp = precision_score(y_val, predictions, average='weighted')
    rec_temp = recall_score(y_val, predictions, average='weighted')
    
    all_final_results.append({
        'name': model_name,
        'type': 'Ensemble',
        'accuracy': acc_temp,
        'f1': f1_temp,
        'precision': prec_temp,
        'recall': rec_temp
    })

# Ordenar por accuracy (mejor a peor)
all_final_results.sort(key=lambda x: x['accuracy'], reverse=True)

print("Pos Tipo           Modelo                    Accuracy   F1-Score   Precision  Recall")
print("-" * 90)

for i, result in enumerate(all_final_results[:20]):  # Top 20
    medal = "馃" if i == 0 else "馃" if i == 1 else "馃" if i == 2 else f"{i+1:2d}."
    print(f"{medal} {result['type']:<12} {result['name']:<22} {result['accuracy']:>8.4f} "
          f"{result['f1']:>10.4f} {result['precision']:>10.4f} {result['recall']:>10.4f}")

print("-" * 90)
print(f"馃弳 CAMPE脫N ABSOLUTO: {all_final_results[0]['name']} ({all_final_results[0]['type']})")
print(f"   Accuracy = {all_final_results[0]['accuracy']:.6f}")

# Estad铆sticas por tipo
print(f"\n馃搱 ESTAD脥STICAS POR TIPO DE MODELO:")
model_types = {}
for result in all_final_results:
    model_type = result['type']
    if model_type not in model_types:
        model_types[model_type] = []
    model_types[model_type].append(result['accuracy'])

for model_type, acc_scores in model_types.items():
    avg_acc = sum(acc_scores) / len(acc_scores)
    max_acc = max(acc_scores)
    print(f"   {model_type}: Promedio Accuracy = {avg_acc:.4f}, Mejor Accuracy = {max_acc:.4f}, Modelos: {len(acc_scores)}")

# Matriz de confusi贸n del mejor modelo
print(f"\n馃幆 MATRIZ DE CONFUSI脫N DEL MEJOR MODELO ({all_final_results[0]['name']}):")
best_predictions = None
for model_name, predictions in models_results + [(name, pred) for name, pred in additional_models.items()] + [(name, pred) for name, pred in ensemble_models]:
    if model_name == all_final_results[0]['name']:
        best_predictions = predictions
        break

if best_predictions is not None:
    cm = confusion_matrix(y_val, best_predictions)
    print("\nMatriz de Confusi贸n:")
    print("Clases:", sorted(y_val.unique()))
    print(cm)
    
    # Reporte de clasificaci贸n
    print(f"\nREPORTE DE CLASIFICACI脫N DEL MEJOR MODELO:")
    print(classification_report(y_val, best_predictions))

print(f"\nTOTAL DE MODELOS PROBADOS: {len(all_final_results)}")
print(f" Proceso de clasificaci贸n completado exitosamente!")