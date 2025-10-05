# predictor.py
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from typing import Dict, List, Optional

class ExoplanetPredictor:
    """
    Sistema para prever classificação de exoplanetas com explicação.
    Lida com dados faltantes e fornece análise de importância.
    """
    
    def __init__(self, model_path: str, training_data_path: str):
        """
        Inicializa o preditor.
        
        Args:
            model_path: Caminho para o modelo treinado (.joblib)
            training_data_path: Caminho para dados de treino (para SHAP)
        """
        self.model = joblib.load(model_path)
        
        # Carregar dados de treino para calcular SHAP values
        df_train = pd.read_csv(training_data_path)
        self.X_train = df_train.drop(columns=["koi_disposition_num"])
        self.feature_names = self.X_train.columns.tolist()
        
        # Inicializar SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        # Labels das classes
        self.class_labels = {
            0: "FALSO POSITIVO",
            1: "CANDIDATO",
            2: "CONFIRMADO"
        }
    
    def preprocess_input(self, user_data: Dict[str, Optional[float]]) -> pd.DataFrame:
        """
        Processa dados do usuário, lidando com valores faltantes.
        
        Args:
            user_data: Dicionário com features e valores (None para faltantes)
        
        Returns:
            DataFrame processado
        """
        # Criar DataFrame com todas as features esperadas
        input_df = pd.DataFrame([user_data])
        
        # Adicionar features faltantes com None
        for feature in self.feature_names:
            if feature not in input_df.columns:
                input_df[feature] = None
        
        # Ordenar colunas na ordem correta
        input_df = input_df[self.feature_names]
        
        for col in input_df.columns:
            # Tentar converter para numérico, transformando erros em NaN
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Preencher valores faltantes com a mediana do treino
        for col in input_df.columns:
            if input_df[col].isna().any():
                median_value = self.X_train[col].median()
                input_df[col].fillna(median_value, inplace=True)
        
        return input_df
    
    def predict_with_explanation(self, user_data: Dict[str, Optional[float]]) -> Dict:
        """
        Faz predição e retorna explicação detalhada.
        
        Args:
            user_data: Dicionário com dados do usuário
        
        Returns:
            Dicionário com predição, probabilidades e explicação
        """
        # Processar entrada
        X_input = self.preprocess_input(user_data)
        
        # Fazer predição
        prediction = self.model.predict(X_input)[0]
        probabilities = self.model.predict_proba(X_input)[0]
        
        # Calcular SHAP values (explicação)
        shap_values = self.explainer.shap_values(X_input)
        
        # Tratar diferentes formatos de SHAP values
        if isinstance(shap_values, list):
            # Multi-classe: lista de arrays
            shap_values_class = shap_values[int(prediction)]
        else:
            # Binário ou regressão: array único
            shap_values_class = shap_values
        
        # Achatar o array se necessário
        if isinstance(shap_values_class, np.ndarray):
            if len(shap_values_class.shape) > 1:
                shap_values_class = shap_values_class.flatten()
        
        # Obter features mais importantes para a predição
        feature_importance = []
        for i, feature in enumerate(self.feature_names):
            # Extrair valor do SHAP de forma segura
            if isinstance(shap_values_class, np.ndarray):
                importance_val = shap_values_class[i]
            else:
                importance_val = shap_values_class[0][i]
            
            # Converter para float nativo do Python
            importance = float(np.asarray(importance_val).item())
            value = float(X_input.iloc[0][feature])
            was_missing = feature not in user_data or user_data[feature] is None
            
            feature_importance.append({
                'feature': feature,
                'value': value,
                'importance': importance,
                'was_missing': was_missing,
                'abs_importance': abs(importance)
            })
        
        # Ordenar por importância absoluta
        feature_importance.sort(key=lambda x: float(x['abs_importance']), reverse=True)
        
        # Criar explicação em texto

        
        return {
            'prediction': int(prediction),
            'prediction_label': self.class_labels[prediction],
            'probabilities': {
                self.class_labels[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            },
            'confidence': float(probabilities[prediction]),
            'top_features': feature_importance[:10],
            'missing_features': [f for f in user_data if user_data[f] is None]
        }
    
    def predict_batch(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        """
        Faz predições em lote a partir de um arquivo CSV ou Excel.
        
        Args:
            input_file: Caminho para arquivo CSV ou Excel com os dados
            output_file: Caminho para salvar resultados (opcional)
        
        Returns:
            DataFrame com predições e explicações
        """
        # Ler arquivo (detecta automaticamente CSV ou Excel)
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(input_file)
        else:
            raise ValueError("Arquivo deve ser CSV ou Excel (.xlsx, .xls)")
        
        # Limpar dados: converter tudo para numérico onde possível
        print("Limpando dados não numéricos...")
        original_shape = df.shape
        
        for col in df.columns:
            if col in self.feature_names:
                # Converter para numérico, transformando erros em NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Contar quantos valores foram convertidos para NaN
        nan_count = df[self.feature_names].isna().sum().sum()
        if nan_count > 0:
            print(f" {nan_count} valores não numéricos encontrados e serão estimados")
        
        results = []
        
        print(f"Processando {len(df)} exoplanetas...")
        
        for idx, row in df.iterrows():
            # Converter linha para dicionário
            user_data = row.to_dict()
            
            # Fazer predição
            try:
                result = self.predict_with_explanation(user_data)
                
                # Adicionar informações básicas
                result_row = {
                    'id': idx,
                    'prediction': result['prediction'],
                    'prediction_label': result['prediction_label'],
                    'confidence': result['confidence'],
                    'prob_false_positive': result['probabilities']['FALSO POSITIVO'],
                    'prob_candidate': result['probabilities']['CANDIDATO'],
                    'prob_confirmed': result['probabilities']['CONFIRMADO'],
                    'top_feature_1': result['top_features'][0]['feature'],
                    'top_feature_1_importance': result['top_features'][0]['importance'],
                    'top_feature_2': result['top_features'][1]['feature'],
                    'top_feature_2_importance': result['top_features'][1]['importance'],
                    'top_feature_3': result['top_features'][2]['feature'],
                    'top_feature_3_importance': result['top_features'][2]['importance'],
                    'missing_features_count': len(result['missing_features'])
                }
                
                results.append(result_row)
                
                if (idx + 1) % 100 == 0:
                    print(f"Processados {idx + 1}/{len(df)}...")
                    
            except Exception as e:
                print(f"Erro na linha {idx}: {e}")
                results.append({
                    'id': idx,
                    'prediction': None,
                    'prediction_label': 'ERRO',
                    'confidence': 0,
                    'error': str(e)
                })
        
        # Converter para DataFrame
        results_df = pd.DataFrame(results)
        
        # Adicionar dados originais
        results_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        
        # Salvar se especificado
        if output_file:
            if output_file.endswith('.csv'):
                results_df.to_csv(output_file, index=False)
            else:
                results_df.to_excel(output_file, index=False)
            print(f"\nResultados salvos em: {output_file}")
        
        print(f"\n✓ Processamento concluído!")
        print(f"Total: {len(df)} | Sucesso: {results_df['prediction'].notna().sum()} | Erros: {results_df['prediction'].isna().sum()}")
        
        return results_df
    
    def generate_summary_report(self, results_df: pd.DataFrame) -> str:
        """
        Gera relatório resumido das predições em lote.
        
        Args:
            results_df: DataFrame retornado por predict_batch
        
        Returns:
            String com relatório formatado
        """
        total = len(results_df)
        
        # Contar por classe
        counts = results_df['prediction_label'].value_counts()
        
        report = "=" * 70 + "\n"
        report += "RELATÓRIO DE CLASSIFICAÇÃO EM LOTE\n"
        report += "=" * 70 + "\n\n"
        
        report += f"📊 RESUMO GERAL\n"
        report += f"Total de exoplanetas analisados: {total}\n\n"
        
        report += "📈 DISTRIBUIÇÃO POR CLASSE\n"
        for label, count in counts.items():
            percentage = (count / total) * 100
            report += f"  • {label}: {count} ({percentage:.1f}%)\n"
        
        report += f"\n🎯 CONFIANÇA MÉDIA\n"
        avg_confidence = results_df['confidence'].mean() * 100
        report += f"  Confiança média: {avg_confidence:.1f}%\n"
        
        # Top features mais influentes
        report += f"\n🔍 TOP 5 FEATURES MAIS INFLUENTES (geral)\n"
        top_features = results_df['top_feature_1'].value_counts().head(5)
        for i, (feature, count) in enumerate(top_features.items(), 1):
            report += f"  {i}. {feature} (apareceu {count}x como mais importante)\n"
        
        # Estatísticas por classe
        report += f"\n📋 ESTATÍSTICAS POR CLASSE\n"
        for label in counts.index:
            if label != 'ERRO':
                subset = results_df[results_df['prediction_label'] == label]
                avg_conf = subset['confidence'].mean() * 100
                report += f"\n  {label}:\n"
                report += f"    - Confiança média: {avg_conf:.1f}%\n"
                report += f"    - Feature mais importante: {subset['top_feature_1'].mode().values[0]}\n"
        
        return report 
   


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Inicializar preditor
    predictor = ExoplanetPredictor(
        model_path="xgboost_grid_best_model.joblib",
        training_data_path="./datasets/selected_features_exoplanets.csv"
    )
    
    # Exemplo 1: Usuário com todos os dados (exoplaneta confirmado típico)
    print("="*70)
    print("EXEMPLO 1: DADOS COMPLETOS - Exoplaneta tipo Terra")
    print("="*70)
    
    user_data_complete = {
        'koi_dikco_msky': 0.125,
        'koi_dicco_msky': 0.098,
        'koi_max_mult_ev': 8.5,
        'koi_fwm_srao': 0.245,
        'koi_fwm_sdeco': 0.189,
        'koi_dikco_mra': 0.156,
        'koi_model_snr': 45.3,
        'koi_dikco_mdec': 0.134,
        'koi_dicco_mdec': 0.112,
        'koi_ror': 0.0145,
        'koi_dicco_mra': 0.167,
        'koi_prad': 1.2,
        'koi_fpflag_ss': 0,
        'koi_dor': 15.8,
        'koi_fpflag_co': 0,
        'koi_max_sngle_ev': 12.7,
        'koi_period': 365.25,
        'koi_fwm_prao': 0.289,
        'koi_num_transits': 4,
        'koi_ldm_coeff1': 0.456,
        'koi_incl': 89.5,
        'koi_fwm_stat_sig': 156.4,
        'koi_depth': 125.8,
        'koi_fwm_pdeco': 0.234,
        'koi_ldm_coeff2': 0.289,
        'koi_bin_oedp_sig': 3.45,
        'koi_count': 1,
        'koi_fpflag_nt': 0,
        'koi_teq': 288.0,
        'koi_insol': 1.0,
        'koi_impact': 0.23,
        'koi_steff': 5778.0,
        'koi_fwm_sra': 0.267
    }
    
    result = predictor.predict_with_explanation(user_data_complete)
    
    # Exemplo 2: Usuário com dados faltantes (júpiter quente típico)
    print("\n" + "="*70)
    print("EXEMPLO 2: DADOS INCOMPLETOS - Júpiter Quente")
    print("="*70)
    
    user_data_incomplete = {
        'koi_dikco_msky': 0.234,
        'koi_dicco_msky': None,  # Dado faltante
        'koi_max_mult_ev': 15.2,
        'koi_fwm_srao': None,  # Dado faltante
        'koi_fwm_sdeco': 0.345,
        'koi_dikco_mra': 0.289,
        'koi_model_snr': 78.9,
        'koi_dikco_mdec': None,  # Dado faltante
        'koi_dicco_mdec': 0.267,
        'koi_ror': 0.089,
        'koi_dicco_mra': 0.312,
        'koi_prad': 11.5,
        'koi_fpflag_ss': 0,
        'koi_dor': 3.2,
        'koi_fpflag_co': 0,
        'koi_max_sngle_ev': 25.6,
        'koi_period': 3.5,
        'koi_fwm_prao': 0.456,
        'koi_num_transits': 12,
        'koi_ldm_coeff1': None,  # Dado faltante
        'koi_incl': 88.7,
        'koi_fwm_stat_sig': 345.7,
        'koi_depth': 8500.0,
        'koi_fwm_pdeco': None,  # Dado faltante
        'koi_ldm_coeff2': 0.312,
        'koi_bin_oedp_sig': 8.9,
        'koi_count': 1,
        'koi_fpflag_nt': 0,
        'koi_teq': 1500.0,
        'koi_insol': 850.0,
        'koi_impact': 0.15,
        'koi_steff': 6200.0,
        'koi_fwm_sra': 0.389
    }
    
    result = predictor.predict_with_explanation(user_data_incomplete)
    print(result['explanation'])
    print(f"\nFeatures que foram estimadas: {result['missing_features']}")
    
    # Exemplo 3: Apenas algumas features conhecidas (Super-Terra)
    print("\n" + "="*70)
    print("EXEMPLO 3: POUCOS DADOS DISPONÍVEIS - Super-Terra")
    print("="*70)
    
    user_data_minimal = {
        'koi_period': 45.2,
        'koi_prad': 2.5,
        'koi_steff': 5200.0,
        'koi_depth': 450.0,
        'koi_teq': 380.0,
        'koi_insol': 3.2,
        'koi_num_transits': 6,
        'koi_model_snr': 32.5
        # Muitas features faltando - serão estimadas
    }
    
    result = predictor.predict_with_explanation(user_data_minimal)
    print(f"\nConfiança da predição: {result['confidence']*100:.1f}%")
    
    # ========================================================================
    # PROCESSAMENTO EM LOTE (múltiplos exoplanetas de uma vez)
    # ========================================================================
    
    print("\n" + "="*70)
    print("EXEMPLO 4: PROCESSAMENTO EM LOTE")
    print("="*70)
    
    # Criar arquivo CSV de exemplo para testar
    import pandas as pd
    
    # Criar dataset de exemplo com múltiplos exoplanetas
    batch_data = pd.DataFrame([
        user_data_complete,
        user_data_incomplete,
        user_data_minimal,
        {  # Mais um exemplo
            'koi_period': 88.0,
            'koi_prad': 0.9,
            'koi_steff': 5200.0,
            'koi_depth': 95.0,
            'koi_teq': 320.0
        }
    ])
    
    # Salvar CSV temporário
    batch_data.to_csv('temp_batch_input.csv', index=False)
    
    # Processar em lote
    results_df = predictor.predict_batch(
        input_file='temp_batch_input.csv',
        output_file='batch_results.csv'
    )
    
    # Gerar relatório
    report = predictor.generate_summary_report(results_df)
    print(report)
    
    # Mostrar primeiras linhas dos resultados
    print("\n📋 PRIMEIRAS LINHAS DOS RESULTADOS:")
    print(results_df[['prediction_label', 'confidence', 'top_feature_1']].head())
    
    # Limpar arquivo temporário
    import os
    if os.path.exists('temp_batch_input.csv'):
        os.remove('temp_batch_input.csv')