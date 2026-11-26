"""Funções compartilhadas: caminhos, leitura do ARFF e configuração de log/gráficos."""
import logging
from pathlib import Path

import pandas as pd
from scipy.io import arff

RAIZ = Path(__file__).resolve().parents[1]
ARFF = RAIZ / "data" / "original" / "phishing.arff"
PROCESSADOS = RAIZ / "data" / "processed"
FIGURAS = RAIZ / "results" / "figuras"
TABELAS = RAIZ / "results" / "tabelas"
LOGS = RAIZ / "results" / "logs"

ALVO = "Result"
ROTULOS = {-1: "phishing", 1: "legitimo"}  # convenção do dataset (Mohammad et al.)

# Paleta (slots 1-3 da paleta categórica de referência)
AZUL, LARANJA, VERDE_AGUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_2 = "#0b0b0b", "#52514e"


def carregar_original() -> pd.DataFrame:
    """Lê o ARFF original e converte os atributos nominais ('-1','0','1') para inteiros."""
    dados, _ = arff.loadarff(ARFF)
    df = pd.DataFrame(dados)
    for col in df.columns:
        df[col] = df[col].str.decode("utf-8").astype(int)
    return df


def configurar_log(nome: str) -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(nome)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.FileHandler(LOGS / f"{nome}.log", mode="w", encoding="utf-8"),
              logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def estilo_graficos():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#d6d5d0", "axes.labelcolor": TINTA_2,
        "xtick.color": TINTA_2, "ytick.color": TINTA_2, "text.color": TINTA,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 10, "figure.dpi": 150,
    })
