
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

csv_path = "./datasets/selected_features_exoplanets.csv"  # ajuste se necessário
k = 6
random_state = 0
out_importances = "./datasets/decision_tree_feature_importances_top20.csv"
out_preds = "./datasets/decision_tree_crossval_predictions.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")
df = pd.read_csv(csv_path)

# 1) separar X e y
if "koi_disposition_num" not in df.columns:
    raise KeyError("Coluna 'koi_disposition_num' não encontrada no CSV.")
X = df.drop(columns=["koi_disposition_num"])
y = df["koi_disposition_num"].astype(int).values

# 3) garantir numérico / imputação (segurança)
# Se você já padronizou/imputou, isso só age como fallback.
X = X.apply(pd.to_numeric, errors='coerce')  # força numerics
imputer = SimpleImputer(strategy="mean")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# (Opcional) se ainda quiser padronizar aqui, descomente:
# scaler = StandardScaler()
# X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns, index=X_imputed.index)
# Usaremos X_imputed assumindo que já está padronizado previamente.
X_final = X_imputed

# 4) k-fold estratificado e classificador
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
clf = DecisionTreeClassifier(random_state=random_state,
                             criterion="gini",
                             max_depth=7,
                             min_samples_leaf= 2 ,
                             # you can set max_depth to avoid overfitting (ex.: max_depth=6)
                             )

# 5) métricas por cross-validate
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
cv_results = cross_validate(clf, X_final, y, cv=skf, scoring=scoring, return_train_score=False)

# Resumo métricas
for metric in scoring:
    arr = cv_results[f"test_{metric}"]
    print(f"{metric}: mean={arr.mean():.4f}  std={arr.std():.4f}")

# 6) predições agregadas via cross_val_predict (útil para relatório e matriz de confusão)
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(clf, X_final, y, cv=skf)

print("\nClassification report (agreg. cross-val predictions):")
print(classification_report(y, y_pred, target_names=["FALSE POS (0)","CANDIDATE (1)","CONFIRMED (2)"], digits=4))

# 7) matriz de confusão agregada
cm = confusion_matrix(y, y_pred, labels=[0,1,2])
cm_df = pd.DataFrame(cm,
                     index=["true_0_falsepos","true_1_candidate","true_2_confirmed"],
                     columns=["pred_0_falsepos","pred_1_candidate","pred_2_confirmed"])
print("\nConfusion matrix (aggregate):")
print(cm_df)

# 8) treinar árvore final em todo o dataset (apenas para extrair feature importances)
final_clf = DecisionTreeClassifier(random_state=random_state)
final_clf.fit(X_final, y)
importances = pd.Series(final_clf.feature_importances_, index=X_final.columns).sort_values(ascending=False)
top20 = importances.head(20).reset_index()
top20.columns = ["feature","importance"]
# print("\nTop 20 features (importância da árvore treinada no conjunto inteiro):")
# print(top20)

# 9) salvar outputs úteis
top20.to_csv(out_importances, index=False)
pd.DataFrame({"y_true": y, "y_pred": y_pred}).to_csv(out_preds, index=False)

# print("\nSalvos:")
# print(" - importâncias:", out_importances)
# print(" - predições cross-val:", out_preds)

# Visualizar a matriz de confusão

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão Agregada')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')
plt.tight_layout()
plt.savefig('./datasets/decision_tree_confusion_matrix.png')