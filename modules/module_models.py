from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
import joblib
import os

def import_models(model_list=None):
    """
    Import and define regression models to evaluate.

    Args:
        model_list (list): List of model names to include. If None, includes all.

    Returns:
        models (dict): Dictionary with model names and their instances.

    Regression Models Available:
        - "LinearRegression": Simple Linear Regression
        - "Ridge": Ridge Regression (L2 Regularization)
        - "Lasso": Lasso Regression (L1 Regularization)
        - "RandomForestRegressor": Random Forest for Regression
        - "KNNRegressor": K-Nearest Neighbors for Regression
        - "SVR": Vector Support Machine for Regression
        - "DecisionTreeRegressor": Decision Tree for Regression
    """
    available_models = {
        # Linear Models
        "LinearRegression": LinearRegression(n_jobs=-1),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.1, max_iter=1000, random_state=42),

        # Non linear/Ensemble Models
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, max_depth=15),
        "KNNRegressor": KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1),
        "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale"),
        "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=10, random_state=42)
    }

    if model_list is None:
        models = available_models
    else:
        # Filter the models based on the provided list
        models = {name: model for name, model in available_models.items() if name in model_list}

    # Check if any models were found
    if not models:
        print(f"\nWarning: None of the models in {model_list} are available.")
    else:
        print("\nRegression models to evaluate:", list(models.keys()))

    return models

# Prepare data for modeling: feature selection, train-test split, encoding.
def prepare_data_for_modeling(df: pd.DataFrame, show_info: bool = True):
    """
    Prepare data for modeling: feature selection and train-test split (Regression).
    
    Args:
        df (pd.DataFrame): The input DataFrame with features and continuous target.
        show_info (bool): If True, prints the sizes of the resulting datasets.
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test:
        Split datasets ready for modeling.
    """
    
    # 1. Feature and Target Definition
    # define your features (add all the ones you want to test).
    features = df.columns.difference(["target","random_index"])  # all except this
    X = df[features].copy()
    y = df["target"].copy()

    # 🚨 CAMBIO CLAVE: Target continuo
    # Se elimina LabelEncoder. El target (y) ya es la variable final para la división.
    print("[INFO] Target continuo. Omitiendo codificación de etiquetas (LabelEncoder).")
    y_enc = y.copy() 

    # -------------------------------------------------------------
    # 2. Split 1: Train/Validation (80%) vs. Test (20%)
    # División aleatoria simple (sin stratify) para targets continuos.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42
    )

    # -------------------------------------------------------------
    # 3. Split 2: Train (64%) vs. Validation (16%)
    # División aleatoria simple (sin stratify).
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42
    )

    # -------------------------------------------------------------
    # 4. Convert to pandas DataFrames/Series
    
    # Convert Feature arrays (X) to pandas DataFrames
    X_train_df = pd.DataFrame(X_train, columns=features)
    X_val_df = pd.DataFrame(X_val, columns=features)
    X_test_df = pd.DataFrame(X_test, columns=features)

    # Convert Target arrays (y) to pandas Series (usando el nombre original 'target')
    y_train_series = pd.Series(y_train, name='target')
    y_val_series = pd.Series(y_val, name='target')
    y_test_series = pd.Series(y_test, name='target')

    print("\nData prepared and split into train, val, test (Random Split for Regressors).")

    if show_info:
        print("\nSizes (Filas x Columnas):")
        print("Train:", X_train_df.shape, y_train_series.shape)
        print("Val:", X_val_df.shape, y_val_series.shape)
        print("Test: ", X_test_df.shape, y_test_series.shape)

    return X_train_df, X_val_df, X_test_df, y_train_series, y_val_series, y_test_series

# Run Random Forest to determine feature importances and select top-K features.
def rf_feature_selection(X_train, y_train, n_features=5, show_importance=True):
    """
    Run Random Forest to determine feature importances and select top-K features.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
    Returns:
        topK (list): List of top-K feature names based on importance.
    """
    rf_base = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1, max_depth=15
        )
    rf_base.fit(X_train, y_train)

    importances = pd.Series(rf_base.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    if show_importance:
        print("\nImportance (RF):", importances)

    # K = number of top features to select
    K = min(n_features, X_train.shape[1])
    topK = importances.index[:K].tolist()
    print("\nTop-K features:", topK)
    return topK

# Run Recursive Feature Elimination (RFE) with Random Forest to select top-K features.
def rfe_feature_selection(X_train, y_train, n_features=5):
    """
    Run Recursive Feature Elimination (RFE) with Random Forest to select top-K features.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        K (int): Number of features to select.
    Returns:
        selected_features (list): List of selected feature names.
    """
    rf_for_rfe = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1, max_depth=15
        )
    rfe = RFE(estimator=rf_for_rfe, n_features_to_select=n_features, step=1)
    rfe.fit(X_train, y_train)

    selected_mask = rfe.support_
    selected_features = X_train.columns[selected_mask].tolist()
    print("\nRFE selected:", selected_features)
    return selected_features

#compare features selected by RF and RFE
def compare_feature_selection(topK_rf, topK_rfe):
    """
    Compare features selected by Random Forest and RFE.
    Args:
        topK_rf (list): Features selected by Random Forest.
        topK_rfe (list): Features selected by RFE.
    """
    set_rf = set(topK_rf)
    set_rfe = set(topK_rfe)

    common_features = set_rf.intersection(set_rfe)
    only_rf = set_rf - set_rfe
    only_rfe = set_rfe - set_rf

    print(f"Common features ({len(common_features)}): {common_features}")
    print(f"Only RF ({len(only_rf)}): {only_rf}")
    print(f"Only RFE ({len(only_rfe)}): {only_rfe}")

# Evaluate a classifier with cross-validation and test set.
def eval_model(regressor, X_tr, y_tr, X_te, y_te, cv_folds=5):
    """
    Evaluate a regressor with cross-validation and test set.
    
    Args:
        regressor: Regressor instance (must implement fit and predict), e.g., LinearRegression.
        X_tr, y_tr: Training data (features and continuous target).
        X_te, y_te: Test data (features and continuous target).
        cv_folds (int): Number of cross-validation folds.
        
    Returns:
        r2_cv, mae_cv: Cross-validated R2 and MAE on training data.
        r2_test, rmse_test, mae_test: R2, RMSE, and MAE on test data.
        dummy_matrix: A dummy placeholder (since confusion matrix is for classification).
    """
    
    # -------------------------------------------------------------
    # CV: Usamos KFold (No Estratificado) para targets continuos.
    # Esto soluciona el ValueError de n_splits=5.
    # -------------------------------------------------------------
    # Nota: Usamos cv_folds=5 por defecto, aunque el argumento original no lo tenía,
    # el error se produjo por n_splits=5, así que lo incluimos como parámetro por buena práctica.
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # R2 (análogo a 'accuracy' en el contexto de CV, métrica principal)
    # cross_val_score con scoring="r2" retorna un valor entre -inf y 1.0. Más alto es mejor.
    r2_cv = cross_val_score(regressor, X_tr, y_tr, cv=cv, scoring="r2").mean()
    
    # MAE (Mean Absolute Error, análogo a 'f1_macro', otra métrica común de error)
    # Usamos "neg_mean_absolute_error" y lo invertimos. Cuanto más bajo el error, mejor.
    mae_cv = -cross_val_score(regressor, X_tr, y_tr, cv=cv, scoring="neg_mean_absolute_error").mean()
    
    # Fit + test
    regressor.fit(X_tr, y_tr)
    y_pred = regressor.predict(X_te)
    
    # R2 en Test (análogo a acc en el original)
    r2_test = r2_score(y_te, y_pred)
    
    # RMSE en Test (análogo a f1 en el original, una métrica de error robusta)
    mse_test = mean_squared_error(y_te, y_pred)
    rmse_test = np.sqrt(mse_test) 

    # MAE en Test (análogo a auc en el original, otra métrica de error)
    mae_test = mean_absolute_error(y_te, y_pred)
    
    # Placeholders para mantener la estructura de retorno
    # La matriz de confusión no aplica en Regresión.
    dummy_matrix = np.zeros((1, 1)) 
    
    # Retorno: Los nombres han cambiado, pero el orden es análogo a la estructura original:
    # (R2_CV, MAE_CV, R2_Test, RMSE_Test, MAE_Test, dummy_matrix)
    return r2_cv, mae_cv, r2_test, rmse_test, mae_test, dummy_matrix

#Comparison of training with all features vs. K features.
def compare_model_performance(X_train, y_train, X_val, y_val, X_test, y_test, selected_features):
    """
    Compare regressor model performance (RandomForestRegressor) 
    using all features vs. selected features (RFE/other selection).
    
    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        X_test, y_test: Test data.
        selected_features (list): List of selected feature names.
    """
    
    # Base Random Forest model (Cambiado a Regressor)
    rf_final = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1, max_depth=15
        )

    # (A) All features
    # metrics_all es la tupla: (r2_cv, mae_cv, r2_test, rmse_test, mae_test, dummy_matrix)
    metrics_all = eval_model(rf_final, X_train, y_train, X_test, y_test)

    # (B) Only features selected by RFE
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]
    metrics_sel = eval_model(rf_final, X_train_sel, y_train, X_test_sel, y_test)

    # (C) ONLY features selected by RFE - VALIDATION
    X_val_sel = X_val[selected_features]
    # En el contexto de evaluación, el conjunto de entrenamiento no cambia, solo el de prueba/validación.
    metrics_sel_val = eval_model(rf_final, X_train_sel, y_train, X_val_sel, y_val)


    # ------------------------------------------------------------------------------------------
    # IMPRESIÓN DE RESULTADOS: Reemplazando métricas de Clasificación por Regresión
    # ------------------------------------------------------------------------------------------
    
    # 0=R2_CV, 1=MAE_CV, 2=R2_Test, 3=RMSE_Test, 4=MAE_Test, 5=dummy_matrix
    
    print("\n=== RF Results (all features) ===")
    print(f"CV R2: {metrics_all[0]:.3f} | CV MAE: {metrics_all[1]:.3f} | Test R2: {metrics_all[2]:.3f} | Test RMSE: {metrics_all[3]:.3f} | Test MAE: {metrics_all[4]:.3f}")
    print("Dummy Matrix (Confusion matrix no aplica):\n", metrics_all[5])

    print("\n=== RF Results (Selected Features - RFE) ===")
    print(f"CV R2: {metrics_sel[0]:.3f} | CV MAE: {metrics_sel[1]:.3f} | Test R2: {metrics_sel[2]:.3f} | Test RMSE: {metrics_sel[3]:.3f} | Test MAE: {metrics_sel[4]:.3f}")
    print("Dummy Matrix (Confusion matrix no aplica):\n", metrics_sel[5])

    print("\n=== RF Results (Selected Features - RFE) Validation ===")
    print(f"CV R2: {metrics_sel_val[0]:.3f} | CV MAE: {metrics_sel_val[1]:.3f} | Test R2: {metrics_sel_val[2]:.3f} | Test RMSE: {metrics_sel_val[3]:.3f} | Test MAE: {metrics_sel_val[4]:.3f}")
    print("Dummy Matrix (Confusion matrix no aplica):\n", metrics_sel_val[5])

def compute_metrics(regressor, X_tr, y_tr, X_ev, y_ev, feature_names):
    """
    Train a regressor and compute performance metrics on an evaluation set.
    
    Args:
        regressor: Regressor instance (must implement fit and predict).
        X_tr, y_tr: Training data.
        X_ev, y_ev: Evaluation data.
        feature_names (list): List of feature names (analogous to class_names, but unused in metrics).
        
    Returns:
        r2, mae, rmse, dummy_metrics, dummy_report, regressor
        (R2, MAE, RMSE, Placeholder for CM/AUC, Placeholder for Report, Trained Regressor)
    """
    
    # 1. Fit the Regressor
    regressor.fit(X_tr, y_tr)
    
    # 2. Predict on the Evaluation Set
    y_pred = regressor.predict(X_ev)
    
    # -------------------------------------------------------------
    # 3. Compute Regression Metrics (Análogos a las métricas originales)
    # -------------------------------------------------------------
    
    # R2 Score (Análogo a Accuracy)
    r2 = r2_score(y_ev, y_pred)
    
    # MAE (Mean Absolute Error, Análogo a F1-score)
    mae = mean_absolute_error(y_ev, y_pred)

    # RMSE (Root Mean Squared Error, Calculado del MSE)
    mse = mean_squared_error(y_ev, y_pred)
    rmse = np.sqrt(mse)
    
    # -------------------------------------------------------------
    # 4. Placeholders para mantener la estructura de retorno
    # -------------------------------------------------------------
    
    # AUC/f1/Acc: En la regresión, no hay un análogo directo para AUC o f1.
    # Usaremos el RMSE en lugar del AUC para llenar el tercer slot de la tupla.
    # Si quieres mantener el RMSE como una métrica separada, podemos usar:
    # r2, mae, rmse, ...
    
    # Usamos un valor fijo (por ejemplo, np.nan) como análogo de la Matriz de Confusión (cm)
    dummy_metrics = np.nan 
    
    # El reporte de clasificación no aplica. Usamos un string descriptivo.
    dummy_report = (
        f"--- Regression Metrics ---\n"
        f"R2: {r2:.4f}\n"
        f"MAE: {mae:.4f}\n"
        f"RMSE: {rmse:.4f}\n"
    )
    
    # 5. Retornar resultados (Orden Análogo: acc, f1, auc, cm, report, clf)
    return r2, mae, rmse, dummy_metrics, dummy_report, regressor

def compare_in_validation(models, X_train, y_train, X_val, y_val, show_detailed_report=True):
    """
    Compare multiple regressor model performance on the validation set.
    
    Args:
        models (dict): Dictionary of {name: regressor_instance}.
        X_train, y_train: Training data (needed for fitting the models).
        X_val, y_val: Validation data for evaluation.
        show_detailed_report (bool): If True, prints the detailed report (R2, MAE, RMSE).
        
    Returns:
        str: The name of the best performing model based on the chosen metric (Validation MAE).
    """
    val_rows = []
    val_details = {}  # to store detailed reports
    
    # 🚨 NOTA: Se elimina LabelEncoder ya que el target es continuo.
    
    # Iterar sobre cada modelo regresor
    for name, regressor in models.items():
        # Usamos la función análoga de Regresión:
        # Retorna: r2, mae, rmse, dummy_metrics, dummy_report, regressor
        r2, mae, rmse, dummy_metrics, rep, _ = compute_metrics(
            regressor, X_train, y_train, X_val, y_val, X_val.columns
        )
        
        # Almacenamos las métricas de Regresión
        val_rows.append({
            "Model": name, 
            "Val_R2": r2, 
            "Val_MAE": mae, # <-- Métrica principal de error
            "Val_RMSE": rmse
        })
        val_details[name] = {"report": rep} # El 'report' ahora es el string con las métricas
        
    # Ordenar los resultados: Para métricas de error (MAE, RMSE), queremos el valor más bajo.
    df_val = pd.DataFrame(val_rows).sort_values("Val_MAE", ascending=True)
    
    print("\n=== Validation Results (Regression) ===")
    print(df_val)

    # -------------------------------------------------------------
    # Selección del Mejor Modelo
    # -------------------------------------------------------------
    best_model_name = df_val.iloc[0]["Model"]
    
    print(f"\nSelected best model on validation (Minimizing MAE): {best_model_name} 🏆")
    
    # Imprimir el reporte detallado (análogo al Classification Report)
    if show_detailed_report:
        print("\nRegression Metrics Report (VAL):\n", val_details[best_model_name]["report"])

    # 🚨 Se retorna solo el nombre, como en la función original.
    return best_model_name

def retrain_best_model(X_train, y_train, X_val, y_val, X_test, y_test, best_model_name):
    """
    Retrains the best regressor model using the combined training and validation 
    data, and evaluates its final performance on the test set.
    
    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        X_test, y_test: Test data.
        best_model_name (str): Name of the best model selected from validation.
        
    Returns:
        The final trained regressor instance.
    """
    
    # 🚨 NOTA: Se elimina LabelEncoder ya que el target es continuo.
    
    # 1. Importar e inicializar el mejor modelo regresor
    # Asumimos que import_models retorna una instancia de Regresor
    models_dict = import_models([best_model_name])
    best_model = models_dict[best_model_name]

    # 2. Combinar conjuntos de Entrenamiento y Validación
    # np.vstack y np.hstack son correctos para combinar arrays de NumPy o Series/DataFrames convertidos.
    # Usamos pd.concat() para DataFrames si X_train, X_val son DataFrames, pero nos apegaremos 
    # a la analogía de numpy/scikit-learn.
    
    # Nos aseguramos de que sean arrays de NumPy para que hstack/vstack funcione con la analogía original.
    # Si X_train, y_train, etc. son Series/DataFrames, debes usar .values para numpy:
    # X_trfin = np.vstack([X_train.values, X_val.values])
    # y_trfin = np.hstack([y_train.values, y_val.values])
    
    # Asumiendo que X_train, y_train, etc. son arrays de NumPy (como salen de train_test_split):
    X_trfin = np.vstack([X_train, X_val])
    y_trfin = np.hstack([y_train, y_val])
    
    # 3. Evaluar el modelo final en el conjunto de Prueba
    # Usamos la función de regresión:
    # Retorna: r2, mae, rmse, dummy_metrics, dummy_report, regressor
    test_r2, test_mae, test_rmse, _, test_rep, final_regressor = compute_metrics(
        best_model, X_trfin, y_trfin, X_test, y_test, X_test.columns
    )

    # 4. Imprimir resultados (Reemplazando métricas de Clasificación por Regresión)
    print("\n=== Test Results (Best Regressor) ===")
    print(f"Model: {best_model_name}")
    print(f"Test R2 Score:  {test_r2:.3f}")
    print(f"Test MAE:       {test_mae:.3f}")
    print(f"Test RMSE:      {test_rmse:.3f}") # Sustituye el slot de AUC
    print("\nRegression Metrics Report (TEST):\n", test_rep) # Sustituye Classification Report

    # 5. Retornar el modelo entrenado
    return final_regressor

def save_model(model, folder_path, filename="final_model.joblib"):
    file_path = os.path.join(folder_path, filename)
    joblib.dump(model, file_path)
    print(f"Model saved successfully in {file_path} as '{filename}'.")