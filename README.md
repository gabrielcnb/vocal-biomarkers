# Biomarcadores Vocais para Detecção de Parkinson

## Resumo

Sistema de machine learning para detecção de indicadores da doença de Parkinson a partir de biomarcadores vocais. Utiliza o dataset UCI Parkinson's (Little et al., 2007) com 195 gravações de 31 sujeitos, extraindo 22 medidas acústicas de fonações sustentadas. Implementa três classificadores (SVM, Random Forest, XGBoost) com **split group-aware** para evitar data leakage entre sujeitos.

**AVISO: Esta ferramenta é exclusivamente para fins de pesquisa e educação. NÃO é um instrumento de diagnóstico médico.**

## Contexto Científico

A disartria e disfonia são sintomas presentes em aproximadamente 90% dos pacientes com doença de Parkinson (Logemann et al., 1978). As alterações vocais incluem:

- **Jitter**: variações involuntárias na frequência fundamental (F0), indicando instabilidade no controle das pregas vocais
- **Shimmer**: variações na amplitude do sinal, refletindo irregularidades na vibração glótica
- **HNR/NHR**: razão harmônico-ruído, medindo a presença de ruído na voz
- **RPDE, DFA, D2**: medidas não-lineares que capturam a complexidade dinâmica do sinal vocal
- **PPE, spread1, spread2**: medidas de variação de pitch que detectam monotonia vocal (característica de PD)

Essas alterações podem ser detectadas antes dos sintomas motores clássicos (tremor, rigidez), tornando a análise vocal um potencial biomarcador precoce.

## Dataset

**Fonte**: [UCI Machine Learning Repository - Parkinsons Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)

| Propriedade | Valor |
|---|---|
| Amostras | 195 gravações de fonação sustentada (/a/) |
| Sujeitos | 31 (23 com Parkinson, 8 saudáveis) |
| Features | 22 biomarcadores vocais |
| Distribuição | 147 PD (75.4%) / 48 saudáveis (24.6%) |

### Features Extraídas

| Grupo | Features | Descrição |
|---|---|---|
| Frequência | Fo, Fhi, Flo | Frequência fundamental (média, máxima, mínima) |
| Jitter | Jitter(%), Jitter(Abs), RAP, PPQ, DDP | Variações ciclo-a-ciclo na frequência |
| Shimmer | Shimmer, Shimmer(dB), APQ3, APQ5, APQ, DDA | Variações ciclo-a-ciclo na amplitude |
| Ruído | NHR, HNR | Razão ruído-harmônico e harmônico-ruído |
| Não-lineares | RPDE, DFA, D2, spread1, spread2, PPE | Complexidade dinâmica e variação de pitch |

## Metodologia

### Split Group-Aware (CRÍTICO)

O dataset contém múltiplas gravações por sujeito (~6 cada). Um split aleatório convencional (`train_test_split`) causaria **data leakage**: gravações do mesmo sujeito apareceriam tanto no treino quanto no teste. O modelo aprenderia a reconhecer a identidade vocal do indivíduo em vez dos biomarcadores de Parkinson.

**Solução**: `GroupShuffleSplit` do scikit-learn, que garante que **todas** as gravações de cada sujeito fiquem exclusivamente no conjunto de treino OU de teste. Os resultados reportados representam a capacidade real de generalização para sujeitos nunca vistos.

### Classificadores

1. **SVM (RBF)**: GridSearchCV sobre C=[0.1, 1, 10, 100], gamma=['scale', 'auto'], class_weight='balanced'
2. **Random Forest**: n_estimators=[100, 200, 500], max_depth=[5, 10, None], class_weight='balanced'
3. **XGBoost**: n_estimators=[100, 200], max_depth=[3, 5, 7], learning_rate=[0.01, 0.1], scale_pos_weight

Todos otimizados com StratifiedKFold(10) e scoring='roc_auc'. StandardScaler ajustado apenas no treino.

## Resultados (Group-Aware Split)

| Modelo | Acurácia | Precisão | Recall | F1-Score | AUC (teste) | AUC (CV) |
|---|---|---|---|---|---|---|
| **SVM** | 0.9070 | 0.9444 | 0.9444 | 0.9444 | **0.9325** | 0.9955 |
| Random Forest | 0.9302 | 0.9459 | 0.9722 | 0.9589 | 0.9246 | 0.9705 |
| XGBoost | 0.9302 | 0.9459 | 0.9722 | 0.9589 | 0.9286 | 0.9636 |

**Nota**: Os resultados com group-aware split são tipicamente inferiores aos reportados com split aleatório na literatura (~95-97% acurácia), justamente porque não há vazamento de informação entre sujeitos. Estes resultados são mais honestos e realistas.

## Análise de Features

As features mais discriminativas (por importância nos modelos):
- **spread1**: medida de variação fundamental de frequência — consistentemente a feature mais importante
- **PPE**: entropia de período de pitch — captura monotonia vocal
- **MDVP:Fo(Hz)**: frequência fundamental média
- **HNR**: razão harmônico-ruído
- **DFA**: análise de flutuação destendenciada
- **RPDE**: entropia de recorrência

## Instalação e Uso

```bash
# Instalar dependências
pip install -r requirements.txt

# Treinar modelos (download automático do dataset)
python train.py

# Iniciar aplicação web
python app.py
# Abrir http://127.0.0.1:5000
```

### Extração de Features de Áudio

A aplicação aceita arquivos WAV de vogal sustentada (/a/) e extrai automaticamente os 22 biomarcadores usando Parselmouth/Praat. Para as medidas não-lineares (RPDE, DFA, D2), utiliza a biblioteca nolds quando disponível.

## Estrutura do Projeto

```
vocal-biomarkers/
├── app.py                  # Servidor Flask
├── train.py                # Pipeline de treinamento
├── config.py               # Configurações centrais
├── ml/
│   ├── dataset.py          # Carregamento e split group-aware
│   ├── train_models.py     # Treinamento SVM, RF, XGBoost
│   ├── evaluate.py         # Avaliação e geração de plots
│   └── predict.py          # Predição com modelo salvo
├── audio/
│   └── extractor.py        # Extração de features via Parselmouth
├── templates/              # HTML (Jinja2)
├── static/                 # CSS, JS, plots gerados
├── data/raw/               # Dataset UCI
└── models/                 # Modelos treinados (.joblib)
```

## Referências

1. **Little, M.A., McSharry, P.E., Roberts, S.J., Costello, D.A.E., Moroz, I.M.** (2007). Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection. *BioMedical Engineering OnLine*, 6:23.

2. **Little, M.A., McSharry, P.E., Hunter, E.J., Spielman, J., Ramig, L.O.** (2009). Suitability of Dysphonia Measurements for Telemonitoring of Parkinson's Disease. *IEEE Transactions on Biomedical Engineering*, 56(4):1015-1022.

3. **Tsanas, A., Little, M.A., McSharry, P.E., Ramig, L.O.** (2010). Accurate Telemonitoring of Parkinson's Disease Progression by Noninvasive Speech Tests. *IEEE Transactions on Biomedical Engineering*, 57(4):884-893.

## Licença

MIT License. Ver arquivo [LICENSE](LICENSE).
