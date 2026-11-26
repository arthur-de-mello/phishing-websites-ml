"""Item 3 dos requisitos: caracterização / exploração do dataset."""
import matplotlib.pyplot as plt
import pandas as pd

from comum import (ALVO, AZUL, FIGURAS, LARANJA, ROTULOS, TABELAS, TINTA_2,
                   carregar_original, configurar_log, estilo_graficos)

log = configurar_log("01_caracterizacao")
estilo_graficos()
FIGURAS.mkdir(parents=True, exist_ok=True)
TABELAS.mkdir(parents=True, exist_ok=True)

df = carregar_original()
n_inst, n_attr = df.shape
log.info("Instâncias: %d | Atributos: %d (30 preditores + 1 classe)", n_inst, n_attr)

# Tabela por atributo: tipo, conjunto de valores, faltantes, desbalanceamento
linhas = []
for col in df.columns:
    cont = df[col].value_counts()
    linhas.append({
        "atributo": col,
        "tipo": "nominal (classe)" if col == ALVO else
                ("nominal binário" if cont.size == 2 else "nominal ordinal (3 níveis)"),
        "valores": "{" + ", ".join(str(v) for v in sorted(cont.index)) + "}",
        "faltantes": int(df[col].isna().sum()),
        "valor_dominante": int(cont.idxmax()),
        "pct_dominante": round(100 * cont.max() / n_inst, 2),
        "distribuicao": "; ".join(f"{v}: {c}" for v, c in sorted(cont.items())),
    })
tab = pd.DataFrame(linhas)
tab.to_csv(TABELAS / "caracterizacao_atributos.csv", index=False)

faltantes = tab[tab.faltantes > 0]
log.info("Atributos com valores faltando: %d", len(faltantes))
preditores = tab[tab.atributo != ALVO].sort_values("pct_dominante", ascending=False)
mais_desb = preditores.iloc[0]
log.info("Atributo mais desbalanceado: %s (valor %d em %.2f%% das instâncias)",
         mais_desb.atributo, mais_desb.valor_dominante, mais_desb.pct_dominante)

classe = df[ALVO].map(ROTULOS).value_counts()
log.info("Distribuição da classe: %s", classe.to_dict())

dup = int(df.duplicated().sum())
log.info("Linhas duplicadas (idênticas a outra): %d (%.1f%%)", dup, 100 * dup / n_inst)

# Correlação de cada atributo com a classe (para o relatório)
corr = df.corr()[ALVO].drop(ALVO).sort_values(key=abs, ascending=False)
corr.rename("correlacao_com_classe").to_csv(TABELAS / "correlacao_com_classe.csv")
log.info("Top 5 correlações com a classe: %s", corr.head(5).round(3).to_dict())

resumo = pd.DataFrame([
    ("Instâncias", n_inst),
    ("Atributos preditores", n_attr - 1),
    ("Atributo classe", f"{ALVO} (-1 = phishing, 1 = legítimo)"),
    ("Atributos com valores faltando", len(faltantes)),
    ("Atributo mais desbalanceado", f"{mais_desb.atributo} ({mais_desb.pct_dominante}% = {mais_desb.valor_dominante})"),
    ("Classe phishing", int(classe["phishing"])),
    ("Classe legítimo", int(classe["legitimo"])),
    ("Linhas duplicadas", dup),
], columns=["item", "valor"])
resumo.to_csv(TABELAS / "resumo_dataset.csv", index=False)

# Figura 1: distribuição da classe
fig, ax = plt.subplots(figsize=(5, 3))
ordem = ["phishing", "legitimo"]
barras = ax.bar(["Phishing (-1)", "Legítimo (1)"], [classe[o] for o in ordem],
                color=[LARANJA, AZUL], width=0.5)
for b, o in zip(barras, ordem):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 80,
            f"{classe[o]:,} ({100 * classe[o] / n_inst:.1f}%)".replace(",", "."),
            ha="center", color=TINTA_2)
ax.set_ylabel("Instâncias")
ax.set_title("Distribuição do atributo classe (Result)", loc="left")
ax.grid(axis="x", visible=False)
ax.set_ylim(0, classe.max() * 1.15)
fig.tight_layout()
fig.savefig(FIGURAS / "distribuicao_classe.png")
plt.close(fig)

# Figura 2: % do valor dominante por atributo (desbalanceamento)
p = preditores.sort_values("pct_dominante")
fig, ax = plt.subplots(figsize=(7, 7))
ax.barh(p.atributo, p.pct_dominante, color=AZUL, height=0.7)
ax.axvline(50, color=TINTA_2, lw=1, ls="--")
ax.set_xlabel("% das instâncias com o valor mais frequente")
ax.set_title("Desbalanceamento dos atributos preditores", loc="left")
ax.set_xlim(0, 100)
ax.grid(axis="y", visible=False)
top = p.iloc[-1]
ax.text(top.pct_dominante - 1, len(p) - 1, f"{top.pct_dominante:.1f}%", va="center",
        ha="right", color="white", fontsize=9)
fig.tight_layout()
fig.savefig(FIGURAS / "desbalanceamento_atributos.png")
plt.close(fig)

# Figura 3: correlação com a classe
c = corr.sort_values()
fig, ax = plt.subplots(figsize=(7, 7))
ax.barh(c.index, c.values, color=[LARANJA if v < 0 else AZUL for v in c.values], height=0.7)
ax.axvline(0, color=TINTA_2, lw=1)
ax.set_xlabel("Correlação de Pearson com Result (positivo = associado a legítimo)")
ax.set_title("Correlação de cada atributo com a classe", loc="left")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(FIGURAS / "correlacao_classe.png")
plt.close(fig)

log.info("Tabelas e figuras salvas em results/")
