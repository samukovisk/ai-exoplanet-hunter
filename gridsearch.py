# gridsearch_decision_tree.py
import os, json, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Ajuste este caminho para o CSV padronizado gerado antes
csv_path = "./datasets/selected_features_exoplanets.csv"

# Parâmetros GridSearch (exaustivo; pode ficar lento)
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "class_weight": [None, "balanced"]
}

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
clf = DecisionTreeClassifier(random_state=random_state)

# 3) GridSearch
grid = GridSearchCV(clf, param_grid=param_grid, scoring="f1_weighted", cv=skf, n_jobs=n_jobs, verbose=2, refit=True)
grid.fit(X, y)  # ATENÇÃO: pode demorar muito se grade for grande


joblib.dump(grid.best_estimator_, "decision_tree_grid_best_model.joblib")
# 5) avaliação agregada (cross-val predictions com melhor config encontrada)
best_params = grid.best_params_
best_score = grid.best_score_
print("Best params:", best_params)
print("Best CV f1_weighted:", best_score)

best_clf = DecisionTreeClassifier(random_state=random_state, **best_params)
y_pred = cross_val_predict(best_clf, X, y, cv=skf, n_jobs=n_jobs)

print(classification_report(y, y_pred, target_names=["FALSE POS (0)","CANDIDATE (1)","CONFIRMED (2)"]))
cm = confusion_matrix(y, y_pred, labels=[0,1,2])
print("Confusion matrix:\n", cm)


