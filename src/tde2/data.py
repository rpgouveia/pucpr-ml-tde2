"""Carga da base, divisão treino/teste e validação cruzada.

Uso típico nos notebooks de modelagem:

    from tde2.data import load_train, get_cv
    X_train, y_train = load_train()
    cv = get_cv()

`load_test()` só deve ser chamado UMA vez, na seção final de avaliação.
"""

from __future__ import annotations

import hashlib
import json
import platform
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config as C

INDEX_COL = "row_id"  # índice original da linha em heart.csv (rastreabilidade, não é atributo)


def sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def validate(df: pd.DataFrame) -> None:
    """Falha cedo se a base não tiver o formato esperado."""
    expected = set(C.FEATURES + [C.TARGET])
    missing = expected - set(df.columns)
    extra = set(df.columns) - expected
    if missing or extra:
        raise ValueError(f"Colunas inesperadas. Faltando: {missing}. Sobrando: {extra}.")
    if df[C.FEATURES + [C.TARGET]].isna().any().any():
        raise ValueError("A base bruta contém NaN; o esperado era nenhum.")
    if not set(df[C.TARGET].unique()) <= {0, 1}:
        raise ValueError(f"Alvo {C.TARGET} deveria ser binário 0/1.")


def load_raw() -> pd.DataFrame:
    """Lê a base completa (918 linhas). Use apenas para caracterização, nunca para modelar."""
    df = pd.read_csv(C.RAW_CSV)
    df.index.name = INDEX_COL
    validate(df)
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[C.FEATURES], df[C.TARGET]


def make_split(seed: int = C.SEED, test_size: float = C.TEST_SIZE) -> dict:
    """Gera train.csv, test.csv e split_report.json em dataset/splits/.

    Divisão estratificada pelo alvo, com semente fixa: rodar de novo produz
    exatamente os mesmos arquivos (mesmo hash).
    """
    df = load_raw()
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[C.TARGET], random_state=seed
    )
    train, test = train.sort_index(), test.sort_index()

    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(C.TRAIN_CSV, index_label=INDEX_COL)
    test.to_csv(C.TEST_CSV, index_label=INDEX_COL)

    def dist(s: pd.Series) -> dict:
        counts = s.value_counts().sort_index()
        return {
            str(k): {"n": int(v), "prop": round(v / len(s), 4)} for k, v in counts.items()
        }

    report = {
        "seed": seed,
        "test_size": test_size,
        "stratified_by": C.TARGET,
        "n_total": len(df),
        "n_train": len(train),
        "n_test": len(test),
        "class_distribution": {
            "total": dist(df[C.TARGET]),
            "train": dist(train[C.TARGET]),
            "test": dist(test[C.TARGET]),
        },
        "zeros_as_missing": {
            col: {
                "train": int((train[col] == 0).sum()),
                "test": int((test[col] == 0).sum()),
            }
            for col in C.ZERO_AS_MISSING
        },
        "sha256": {
            "heart.csv": sha256(C.RAW_CSV),
            "train.csv": sha256(C.TRAIN_CSV),
            "test.csv": sha256(C.TEST_CSV),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    C.SPLIT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def verify_split() -> None:
    """Confere se os arquivos de divisão estão íntegros e consistentes com a base bruta."""
    report = json.loads(C.SPLIT_REPORT.read_text())
    for name, path in [("heart.csv", C.RAW_CSV), ("train.csv", C.TRAIN_CSV), ("test.csv", C.TEST_CSV)]:
        if sha256(path) != report["sha256"][name]:
            raise AssertionError(
                f"{name} difere do registrado em split_report.json. "
                "Rode `uv run python scripts/make_split.py` e faça commit do resultado."
            )
    train = pd.read_csv(C.TRAIN_CSV, index_col=INDEX_COL)
    test = pd.read_csv(C.TEST_CSV, index_col=INDEX_COL)
    raw = load_raw()
    assert train.index.intersection(test.index).empty, "Linhas repetidas entre treino e teste."
    assert len(train) + len(test) == len(raw), "Treino + teste não somam a base inteira."
    assert raw.loc[train.index].equals(train), "Conteúdo do treino diverge da base bruta."
    assert raw.loc[test.index].equals(test), "Conteúdo do teste diverge da base bruta."


def load_train() -> tuple[pd.DataFrame, pd.Series]:
    """X_train, y_train. Toda seleção de modelo e ajuste acontece aqui dentro."""
    df = pd.read_csv(C.TRAIN_CSV, index_col=INDEX_COL)
    validate(df)
    return split_xy(df)


def load_test() -> tuple[pd.DataFrame, pd.Series]:
    """X_test, y_test. Chamar UMA única vez, ao final, com as configurações já congeladas."""
    warnings.warn(
        "Carregando o conjunto de TESTE. Ele deve ser usado uma única vez, ao final, "
        "depois que todas as configurações estiverem escolhidas.",
        stacklevel=2,
    )
    df = pd.read_csv(C.TEST_CSV, index_col=INDEX_COL)
    validate(df)
    return split_xy(df)


def get_cv(n_splits: int = C.N_SPLITS, seed: int = C.SEED) -> StratifiedKFold:
    """Mesmo objeto de validação cruzada para TODAS as alternativas (membros e comitês).

    Com shuffle=True e random_state fixo, as dobras são idênticas a cada chamada,
    desde que o X_train seja o mesmo (garantido pelo train.csv versionado).
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
