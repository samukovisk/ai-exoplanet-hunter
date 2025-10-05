# exoplanet_predictor.py
# Script para processar múltiplos exoplanetas de uma planilha

import sys
import argparse
from predictor import ExoplanetPredictor
import pandas as pd

def main():
    """
    Script principal para processar planilhas de exoplanetas.
    
    Uso:
        python exoplanet_predictor.py input.csv output.csv
        python exoplanet_predictor.py input.xlsx output.xlsx
        python exoplanet_predictor.py input.csv output.csv --report
    """
    
    parser = argparse.ArgumentParser(
        description='Classifica múltiplos exoplanetas a partir de uma planilha'
    )
    parser.add_argument(
        'input_file',
        help='Arquivo de entrada (CSV ou Excel) com dados dos exoplanetas'
    )
    parser.add_argument(
        'output_file',
        help='Arquivo de saída para salvar resultados'
    )
    parser.add_argument(
        '--model',
        default='xgboost_grid_best_model1.joblib',
        help='Caminho para o modelo treinado'
    )
    parser.add_argument(
        '--training-data',
        default='./datasets/selected_features_exoplanets.csv',
        help='Caminho para dados de treinamento'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Gerar relatório resumido'
    )
    parser.add_argument(
        '--report-file',
        default='batch_report.txt',
        help='Arquivo para salvar o relatório'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("CLASSIFICADOR DE EXOPLANETAS EM LOTE")
    print("="*70)
    print(f"\n Arquivo de entrada: {args.input_file}")
    print(f" Arquivo de saída: {args.output_file}")
    print(f" Modelo: {args.model}\n")
    
    try:
        # Inicializar preditor
        print("Carregando modelo...")
        predictor = ExoplanetPredictor(
            model_path=args.model,
            training_data_path=args.training_data
        )
        print(" Modelo carregado com sucesso!\n")
        
        # Processar em lote
        results_df = predictor.predict_batch(
            input_file=args.input_file,
            output_file=args.output_file
        )
        
        # Gerar relatório se solicitado
        if args.report:
            print("\n" + "="*70)
            print("GERANDO RELATÓRIO")
            print("="*70)
            
            report = predictor.generate_summary_report(results_df)
            print(report)
            
            # Salvar relatório em arquivo
            with open(args.report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n Relatório salvo em: {args.report_file}")
        
        # Mostrar resumo rápido
        print("\n" + "="*70)
        print("RESUMO RÁPIDO")
        print("="*70)
        counts = results_df['prediction_label'].value_counts()
        for label, count in counts.items():
            print(f"{label}: {count}")
        
        print("\n Processamento concluído com sucesso!")
        
    except FileNotFoundError as e:
        print(f"\n ERRO: Arquivo não encontrado - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# EXEMPLOS DE USO
# ============================================================================

"""
# 1. Processar CSV simples
python exoplanet_predictor.py meus_exoplanetas.csv resultados.csv

# 2. Processar Excel
python exoplanet_predictor.py meus_exoplanetas.xlsx resultados.xlsx

# 3. Processar com relatório
python exoplanet_predictor.py meus_exoplanetas.csv resultados.csv --report

# 4. Especificar modelo customizado
python exoplanet_predictor.py input.csv output.csv --model meu_modelo.joblib

# 5. Relatório customizado
python exoplanet_predictor.py input.csv output.csv --report --report-file relatorio.txt
"""

# ============================================================================
# FORMATO DO ARQUIVO DE ENTRADA
# ============================================================================

"""
O arquivo CSV ou Excel deve ter colunas com os nomes das features:

Exemplo CSV (mínimo necessário):
koi_period,koi_prad,koi_steff,koi_depth
365.25,1.0,5778.0,120.5
3.5,11.2,6200.0,8500.0
45.0,2.5,5200.0,450.0

Exemplo CSV (completo - todas as 33 features):
koi_dikco_msky,koi_dicco_msky,koi_max_mult_ev,koi_fwm_srao,...
0.125,0.098,8.5,0.245,...
0.234,0.198,15.2,0.345,...

⚠️ IMPORTANTE:
- Colunas não fornecidas serão preenchidas automaticamente com valores estimados
- Valores vazios (NA, NaN, em branco) também serão estimados
- Não incluir a coluna 'koi_disposition_num' (é o target, não entrada)
"""

# ============================================================================
# FORMATO DO ARQUIVO DE SAÍDA
# ============================================================================

"""
O arquivo de saída conterá TODAS as colunas do arquivo de entrada MAIS:

- id: Número da linha
- prediction: Classe numérica (0, 1, 2)
- prediction_label: Label da classe (FALSO POSITIVO, CANDIDATO, CONFIRMADO)
- confidence: Confiança da predição (0-1)
- prob_false_positive: Probabilidade de ser falso positivo
- prob_candidate: Probabilidade de ser candidato
- prob_confirmed: Probabilidade de ser confirmado
- top_feature_1: Feature mais importante
- top_feature_1_importance: Importância da feature 1
- top_feature_2: Segunda feature mais importante
- top_feature_2_importance: Importância da feature 2
- top_feature_3: Terceira feature mais importante
- top_feature_3_importance: Importância da feature 3
- missing_features_count: Quantas features foram estimadas

Você pode abrir este arquivo no Excel, Google Sheets, ou qualquer ferramenta
de análise de dados!
"""

if __name__ == "__main__":
    main()