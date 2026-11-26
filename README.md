# Classificação de websites de phishing – Aprendizado de Máquina

Trabalho prático de Inteligência Artificial (Feevale): comparação de 3 algoritmos de
classificação supervisionada no dataset **Phishing Websites** (UCI).

- Dataset: https://archive.ics.uci.edu/dataset/327/phishing+websites
  (Mohammad, R. & McCluskey, L., 2012 – DOI 10.24432/C51W2X, licença CC BY 4.0)
- Atributo classe: `Result` (-1 = phishing, 1 = legítimo)

## Estrutura

```
data/original/phishing.arff        dataset original (11.055 instâncias, 30 atributos + classe)
data/processed/v1_original.csv      V1: atributos -1/0/1, classe nominal
data/processed/v2_onehot.csv        V2: one-hot encoding (68 colunas binárias)
data/processed/v3_sem_duplicatas.csv V3: V1 sem linhas duplicadas (5.849 instâncias)
src/01_caracterizacao.py            item 3: exploração do dataset
src/02_preprocessamento.py          gera V1, V2 e V3
src/03_experimentos.py              itens 4 e 5: Árvore de Decisão, Naive Bayes, KNN
results/tabelas/                    tabelas em CSV usadas no relatório
results/figuras/                    gráficos usados no relatório
results/logs/                       logs de cada execução
```

## Como executar

```bash
pip install -r requirements.txt
cd src
python 01_caracterizacao.py
python 02_preprocessamento.py
python 03_experimentos.py
```

Validação cruzada estratificada de 10 folds (semente 42). Métricas em média macro.
