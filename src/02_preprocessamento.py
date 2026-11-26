"""Gera as variações de pré-processamento usadas nos experimentos (item 1.d da entrega).

V1 original      : atributos como estão (-1/0/1), classe convertida para nominal.
V2 one-hot       : cada atributo nominal vira colunas binárias (0/1), uma por valor.
V3 sem_duplicatas: V1 sem as linhas repetidas (47% do dataset são cópias exatas).
"""
import pandas as pd

from comum import ALVO, PROCESSADOS, ROTULOS, carregar_original, configurar_log

log = configurar_log("02_preprocessamento")
PROCESSADOS.mkdir(parents=True, exist_ok=True)

df = carregar_original()
df[ALVO] = df[ALVO].map(ROTULOS)  # -1/1 -> phishing/legitimo (classe nominal)

v1 = df
v2 = pd.concat([pd.get_dummies(df.drop(columns=ALVO).astype(str), dtype=int), df[ALVO]], axis=1)
v3 = df.drop_duplicates().reset_index(drop=True)

for nome, d in [("v1_original", v1), ("v2_onehot", v2), ("v3_sem_duplicatas", v3)]:
    d.to_csv(PROCESSADOS / f"{nome}.csv", index=False)
    log.info("%s: %d instâncias x %d atributos preditores | classe: %s",
             nome, len(d), d.shape[1] - 1, d[ALVO].value_counts().to_dict())
