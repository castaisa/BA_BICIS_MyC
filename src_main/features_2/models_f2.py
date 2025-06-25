import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def linear_regression(x_train, y_train, x_val):
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    return y_pred

def random_forest_regressor(x_train, y_train, x_val, n_estimators=100, random_state=42):
    """
    Random Forest Regressor
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    return y_pred

def gradient_boosting_regressor(x_train, y_train, x_val, n_estimators=100, learning_rate=0.1, random_state=42):
    """
    Gradient Boosting Regressor
    """
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    return y_pred

def decision_tree_regressor(x_train, y_train, x_val, random_state=42):
    """
    Decision Tree Regressor
    """
    model = DecisionTreeRegressor(random_state=random_state)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    return y_pred

def knn_regressor(x_train, y_train, x_val, n_neighbors=5):
    """
    K-Nearest Neighbors Regressor
    """
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    return y_pred

def svr_regressor(x_train, y_train, x_val, kernel='rbf', C=1.0):
    """
    Support Vector Regressor
    """
    model = SVR(kernel=kernel, C=C)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    return y_pred

def ridge_regression(x_train, y_train, x_val, alpha=1.0):
    """
    Ridge Regression (L2 regularization)
    """
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    return y_pred

def lasso_regression(x_train, y_train, x_val, alpha=1.0):
    """
    Lasso Regression (L1 regularization)
    """
    model = Lasso(alpha=alpha)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    return y_pred


def ada_boost_regressor(x_train, y_train, x_val, n_estimators=50, learning_rate=1.0, random_state=42):
    """
    AdaBoost Regressor
    """
    model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    return y_pred


