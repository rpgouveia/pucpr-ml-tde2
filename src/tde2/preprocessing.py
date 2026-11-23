"""Pré-processamento NÃO ajustado, para ser colocado dentro de um Pipeline.

Nunca chame .fit() deste objeto fora de um Pipeline avaliado por validação
cruzada: a mediana de imputação e a média/desvio do StandardScaler têm de ser
aprendidas só com os dados de treino de cada dobra.

    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from tde2.preprocessing import build_preprocessor

    model = make_pipeline(build_preprocessor(), LogisticRegression())
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Cria o ColumnTransformer padrão do projeto.

    - RestingBP e Cholesterol: 0 é tratado como ausente, imputado pela mediana
      do treino da dobra. `add_indicator=True` cria a coluna
      `missingindicator_<col>`, porque o fato de o colesterol estar zerado é
      informativo (88% dessas linhas são positivas).
    - Demais numéricos: só padronização.
    - FastingBS: já é 0/1, passa direto.
    - Categóricos: one-hot (binários viram uma coluna só).

    scale=False desliga a padronização (árvores e comitês de árvores não
    precisam). Manter scale=True para todos também é válido e simplifica a
    comparação "mesmas condições".
    """
    scaler = [("scale", StandardScaler())] if scale else []

    zero_missing = Pipeline(
        [("impute", SimpleImputer(missing_values=0, strategy="median", add_indicator=True))]
        + scaler
    )
    numeric = Pipeline(scaler) if scale else "passthrough"
    categorical = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        [
            ("zero_missing", zero_missing, C.ZERO_AS_MISSING),
            ("numeric", numeric, C.OTHER_NUMERIC),
            ("binary", "passthrough", C.BINARY),
            ("categorical", categorical, C.CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
