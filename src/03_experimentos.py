"""Itens 4 e 5: compara 3 algoritmos em cada variação de pré-processamento.

Validação cruzada estratificada de 10 folds. Métricas: acurácia, precisão, recall e
F1 (média macro entre as duas classes), tempo médio de treino por fold e métricas por classe.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from comum import (ALVO, AZUL, FIGURAS, LARANJA, PROCESSADOS, TABELAS, TINTA_2,
                   VERDE_AGUA, configurar_log, estilo_graficos)

log = configurar_log("03_experimentos")
estilo_graficos()
SEMENTE = 42
CLASSES = ["phishing", "legitimo"]
VARIACOES = {"v1_original": "V1 – Original", "v2_onehot": "V2 – One-hot",
             "v3_sem_duplicatas": "V3 – Sem duplicatas"}


def algoritmos():
    return {
        "Árvore de Decisão": DecisionTreeClassifier(random_state=SEMENTE),
        # CategoricalNB exige categorias >= 0: desloca -1/0/1 para 0/1/2
        "Naive Bayes (categórico)": make_pipeline(FunctionTransformer(lambda x: x + 1),
                                                  CategoricalNB(min_categories=3)),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEMENTE)
metricas = {"acuracia": "accuracy", "precisao": "precision_macro",
            "recall": "recall_macro", "f1": "f1_macro"}
geral, por_classe = [], []

for var, rotulo in VARIACOES.items():
    d = pd.read_csv(PROCESSADOS / f"{var}.csv")
    X, y = d.drop(columns=ALVO).to_numpy(), d[ALVO].to_numpy()
    log.info("=== %s: %d instâncias, %d atributos ===", rotulo, *X.shape)
    for nome, modelo in algoritmos().items():
        r = cross_validate(modelo, X, y, cv=cv, scoring=metricas)
        linha = {"variacao": rotulo, "algoritmo": nome}
        for m in metricas:
            linha[m] = r[f"test_{m}"].mean()
            linha[f"{m}_dp"] = r[f"test_{m}"].std()
        linha["tempo_treino_s"] = r["fit_time"].mean()
        geral.append(linha)
        log.info("%-26s acc=%.4f prec=%.4f rec=%.4f f1=%.4f treino=%.4fs (média/fold)",
                 nome, linha["acuracia"], linha["precisao"], linha["recall"],
                 linha["f1"], linha["tempo_treino_s"])

        pred = cross_val_predict(modelo, X, y, cv=cv)
        rep = classification_report(y, pred, labels=CLASSES, output_dict=True)
        cm = confusion_matrix(y, pred, labels=CLASSES)
        for i, c in enumerate(CLASSES):
            por_classe.append({"variacao": rotulo, "algoritmo": nome, "classe": c,
                               "precisao": rep[c]["precision"], "recall": rep[c]["recall"],
                               "f1": rep[c]["f1-score"], "suporte": int(rep[c]["support"]),
                               "erros": int(cm[i].sum() - cm[i, i])})
        log.info("   matriz de confusão (linhas=real %s): %s", CLASSES, cm.tolist())

geral = pd.DataFrame(geral)
por_classe = pd.DataFrame(por_classe)
geral.round(4).to_csv(TABELAS / "resultados_gerais.csv", index=False)
por_classe.round(4).to_csv(TABELAS / "resultados_por_classe.csv", index=False)

melhor = geral.loc[geral.f1.idxmax()]
pior = geral.loc[geral.f1.idxmin()]
log.info("Melhor: %s / %s (F1=%.4f) | Pior: %s / %s (F1=%.4f)",
         melhor.algoritmo, melhor.variacao, melhor.f1, pior.algoritmo, pior.variacao, pior.f1)

# Figura: F1 por algoritmo e variação (barras agrupadas, eixo único)
cores = {"Árvore de Decisão": AZUL, "Naive Bayes (categórico)": LARANJA, "KNN (k=5)": VERDE_AGUA}
fig, ax = plt.subplots(figsize=(7.5, 3.8))
largura = 0.26
xs = np.arange(len(VARIACOES))
for i, (alg, cor) in enumerate(cores.items()):
    s = geral[geral.algoritmo == alg].set_index("variacao").loc[list(VARIACOES.values())]
    # pontos (não barras): o eixo não começa em zero para mostrar as diferenças
    ax.scatter(xs + (i - 1) * largura, s.f1 * 100, s=70, color=cor, label=alg, zorder=3,
               edgecolors="#fcfcfb", linewidths=2)
    for x, v in zip(xs + (i - 1) * largura, s.f1 * 100):
        ax.text(x, v + 0.5, f"{v:.1f}", ha="center", fontsize=8, color=TINTA_2)
ax.set_xticks(xs, list(VARIACOES.values()))
ax.set_ylabel("F1 macro (%)")
ax.set_ylim(90, 98)
ax.set_title("F1 médio (10 folds) por algoritmo e pré-processamento", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(FIGURAS / "f1_por_variacao.png")
plt.close(fig)
log.info("Resultados salvos em results/tabelas e results/figuras")
