import pandas as pd
import numpy as np
df = pd.read_csv('./datasets/ml_ready_exoplanets.csv')



# Calculate the mutual information
from sklearn.feature_selection import mutual_info_regression
# Prepare raw data (before normalization) for mutual information calculation
X_raw = df.drop("koi_disposition_num", axis=1)
df_corr = df

# Delete all "err" columns
err_cols = [col for col in df_corr.columns if "err" in col]
df_corr = df_corr.drop(columns=err_cols)
# Select only numeric columns for correlation
numeric_df = df_corr.select_dtypes(include=[np.number])

# Drop rows with NaN values only in the columns needed for MI calculation
mi_features = numeric_df.drop("koi_disposition_num", axis=1)
mi_target = numeric_df["koi_disposition_num"]
mi_df = numeric_df.dropna(subset=mi_features.columns.tolist() + ["koi_disposition_num"])

# Calculate mutual information for all numeric features if there are samples
if len(mi_df) > 0:
	mi = mutual_info_regression(mi_df.drop("koi_disposition_num", axis=1), mi_df["koi_disposition_num"])
	mi_series = pd.Series(mi, index=mi_df.drop("koi_disposition_num", axis=1).columns)
	mi_series.sort_values(ascending=False)
else:
	print("No samples available for mutual information calculation after dropping NaNs.")
	
# Selects the top 20 features based on mutual information
# mi_series = mi_series[mi_series >= 0.1].sort_values(ascending=False).head(20)
# Features com MI >= 0.1
selected_features = mi_series.index.tolist()

# Inclui a coluna alvo de volta
selected_features.append("koi_disposition_num")

# Cria um novo dataframe só com as colunas relevantes
df_filtered = df_corr[selected_features]

# Salva o novo dataframe em um arquivo CSV
df_filtered.to_csv('./datasets/selected_features_exoplanets.csv', index=False)