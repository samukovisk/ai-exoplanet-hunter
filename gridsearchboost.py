import os, json, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# Ajuste este caminho para o CSV padronizado gerado antes
csv_path = "./datasets/selected_features_exoplanets.csv"

# Parâmetros GridSearch para XGBoost
# Grade RÁPIDA (recomendado para começar)
param_grid_fast = {
    "max_depth": [5, 7, 9],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [200, 300],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3]
}

# Grade COMPLETA (mais lenta, mas mais exaustiva)
param_grid_full = {
    "max_depth": [3, 5, 7, 9, 11],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 300, 400],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.2, 0.3],
    "reg_alpha": [0, 0.01, 0.1],  # regularização L1
    "reg_lambda": [1, 1.5, 2]      # regularização L2
}

# Escolha qual grade usar (comece com fast!)
param_grid = param_grid_fast

# Config
k = 5
random_state = 0
n_jobs = -1  # use all CPUs; ajuste se necessário

# 1) carregar
df = pd.read_csv(csv_path)
X = df.drop(columns=["koi_disposition_num"])
y = df["koi_disposition_num"].astype(int).values

# 2) k-fold estratificado
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

# 3) Criar classificador XGBoost
clf = xgb.XGBClassifier(
    random_state=random_state,
    eval_metric='mlogloss',  # para multi-classe
    use_label_encoder=False,  # evita warning
    tree_method='hist'  # mais rápido
)

# 4) GridSearch
print("Iniciando GridSearch com XGBoost...")
print(f"Total de combinações: {np.prod([len(v) for v in param_grid.values()])}")

grid = GridSearchCV(
    clf, 
    param_grid=param_grid, 
    scoring="f1_weighted", 
    cv=skf, 
    n_jobs=n_jobs, 
    verbose=2, 
    refit=True
)

grid.fit(X, y)

# Salvar melhor modelo
joblib.dump(grid.best_estimator_, "xgboost_grid_best_model.joblib")
print("\nModelo salvo em: xgboost_grid_best_model.joblib")

# 5) Resultados do melhor modelo
best_params = grid.best_params_
best_score = grid.best_score_

print("\n" + "="*60)
print("RESULTADOS DO GRID SEARCH")
print("="*60)
print("Best params:", json.dumps(best_params, indent=2))
print(f"Best CV f1_weighted: {best_score:.4f}")

# 6) Avaliação agregada (cross-val predictions com melhor config encontrada)
print("\n" + "="*60)
print("AVALIAÇÃO COM CROSS-VALIDATION")
print("="*60)

best_clf = xgb.XGBClassifier(
    random_state=random_state,
    eval_metric='mlogloss',
    use_label_encoder=False,
    tree_method='hist',
    **best_params
)

y_pred = cross_val_predict(best_clf, X, y, cv=skf, n_jobs=n_jobs)

print("\nClassification Report:")
print(classification_report(
    y, y_pred, 
    target_names=["FALSE POS (0)", "CANDIDATE (1)", "CONFIRMED (2)"]
))

cm = confusion_matrix(y, y_pred, labels=[0, 1, 2])
print("\nConfusion Matrix:")
print(cm)

# 7) Feature Importance (bônus do XGBoost!)
print("\n" + "="*60)
print("TOP 15 FEATURES MAIS IMPORTANTES")
print("="*60)

best_clf.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_clf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(15).to_string(index=False))

# 8) Salvar resultados em JSON
results = {
    "model": "XGBoost",
    "best_params": best_params,
    "best_cv_f1_weighted": float(best_score),
    "cv_folds": k,
    "random_state": random_state,
    "top_features": feature_importance.head(15).to_dict('records')
}

with open("xgboost_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResultados salvos em: xgboost_results.json")
print("\n" + "="*60)
print("CONCLUÍDO!")
print("="*60)