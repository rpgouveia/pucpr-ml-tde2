# TDE 2: Comparação de comitês de classificadores

Aprendizagem de Máquina (PCO103A), PUCPR, Prof. Carlos Costa Cedro. Entrega: **23/11/2026**, antes do início da Aula 14.
Enunciado completo: [docs/AM TDE 2.pdf](docs/AM%20TDE%202.pdf).

| Parte | Responsável | Entrega |
|---|---|---|
| Dados | Renato Gouveia | base, divisão treino/teste, pré-processamento, caracterização (`notebooks/01_*`) |
| Código | Fernando Alonso | membros, comitês, ajuste, diversidade, erros, custo (`notebooks/02_*`) |
| Relatório | Ângelo Piovezan | PDF de 6 a 10 páginas |

## Estrutura

```
dataset/
  heart.csv                 base bruta (Kaggle), NÃO editar
  splits/
    train.csv               612 linhas (2/3): usar para tudo
    test.csv                306 linhas (1/3): usar UMA vez, no final
    split_report.json       semente, distribuição das classes, hashes SHA-256, versões
src/tde2/
  config.py                 SEED, caminhos, colunas, métricas: único lugar para constantes
  data.py                   load_train(), load_test(), get_cv(), make_split(), verify_split()
  preprocessing.py          build_preprocessor(): ColumnTransformer NÃO ajustado
scripts/make_split.py       gera/verifica a divisão
notebooks/
  01_caracterizacao_dados.ipynb   seção 4.1 do relatório (pronto)
  02_modelagem.ipynb              esqueleto com TODOs para a parte de código
reports/figures/            figuras geradas pelos notebooks (PNG, para o relatório)
```

## Como rodar

**Com [uv](https://docs.astral.sh/uv/) (recomendado):**

```bash
uv sync                                          # cria .venv com Python 3.12 e versões fixadas (uv.lock)
uv run python scripts/make_split.py --check      # confere se a divisão está íntegra
uv run jupyter lab                               # abre os notebooks
```

**Com pip:** `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

**No Google Colab:** na primeira célula, `!git clone <url-do-repo> && %cd pucpr-ml-tde2/notebooks`. A célula de bootstrap de cada notebook encontra `src/` sozinha.

Para regerar a divisão (não deveria ser necessário): `uv run python scripts/make_split.py`. O resultado sai idêntico, byte a byte, e o script confere os hashes.

## A base

**Heart Failure Prediction Dataset** ([Kaggle, fedesoriano, 2021](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)), que junta 5 bases de doença cardíaca do UCI (Cleveland, Hungarian, Switzerland, Long Beach VA, Statlog) sem duplicatas.

- **918 exemplos, 11 atributos preditivos** (5 numéricos, 1 binário, 5 categóricos), alvo binário `HeartDisease`.
- **Classes:** 508 com doença (55,3%) e 410 normais (44,7%), levemente desbalanceada.
- **Ausentes disfarçados:** `Cholesterol = 0` em 172 linhas (88% delas positivas, então o "zero" é informativo) e `RestingBP = 0` em 1 linha. Sem `NaN` nem duplicatas.
- **Não satura:** a regressão logística, sem ajuste, fica em ~0,86 ± 0,04 de acurácia balanceada (CV 10 dobras), o que deixa espaço para os comitês se diferenciarem.

Detalhes, tabelas e figuras: [notebooks/01_caracterizacao_dados.ipynb](notebooks/01_caracterizacao_dados.ipynb).

## Protocolo já fixado (seção 3 do enunciado)

| Regra | Implementação |
|---|---|
| Teste separado no início e usado uma vez | `train_test_split` estratificado, 2/3 treino e 1/3 teste, `random_state=42`, gravado em `dataset/splits/`. `load_test()` emite um aviso ao ser chamado. |
| Pré-processamento dentro de pipeline | `build_preprocessor()` devolve um transformador **não ajustado**. Sempre usar `make_pipeline(build_preprocessor(), modelo)`. |
| Mesmas divisões para todas as alternativas | `get_cv()` = `StratifiedKFold(10, shuffle=True, random_state=42)`, com as mesmas dobras a cada chamada. |
| Sementes fixas e documentadas | `config.SEED = 42` para divisão, dobras e `random_state` dos modelos. |
| Versões das bibliotecas | Python 3.12.3, numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1 (também em `split_report.json`). |

**Pipeline de pré-processamento** (`src/tde2/preprocessing.py`):

| Colunas | Transformação |
|---|---|
| `RestingBP`, `Cholesterol` | 0 → ausente, imputado pela mediana do treino da dobra, mais a coluna `missingindicator_Cholesterol` e padronização |
| `Age`, `MaxHR`, `Oldpeak` | padronização (`StandardScaler`) |
| `FastingBS` | passa direto (já é 0/1) |
| `Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope` | one-hot (`drop="if_binary"`, `handle_unknown="ignore"`) |

Saída: 19 colunas. `build_preprocessor(scale=False)` desliga a padronização para modelos de árvore. Manter `scale=True` para todos também é válido e simplifica o argumento de "mesmas condições".

## Uso no código

```python
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

from tde2 import config as C
from tde2.data import load_train, get_cv
from tde2.preprocessing import build_preprocessor

X_train, y_train = load_train()
cv = get_cv()

rf = make_pipeline(build_preprocessor(scale=False),
                   RandomForestClassifier(random_state=C.SEED))
r = cross_validate(rf, X_train, y_train, cv=cv, scoring=C.SCORING)
```

O esqueleto [notebooks/02_modelagem.ipynb](notebooks/02_modelagem.ipynb) já tem uma função `evaluate()`, que aplica as mesmas dobras e métricas e mede o tempo, e a linha de base da classe majoritária.

## Decisões em aberto para a parte de código

1. **Métrica principal.** A sugestão em `config.PRIMARY_METRIC` é `balanced_accuracy`: funciona com predições duras (votação *hard*) e não favorece o "chuta sempre doença". Reportar junto `f1_macro`, `recall` da classe doença (falso negativo é o erro caro) e `roc_auc`. Com a métrica F1 da classe positiva, a linha de base majoritária chega a 0,71 e engana.
2. **Variabilidade.** Com 10 dobras, o desvio da regressão logística já é ±0,04. As diferenças entre comitês provavelmente serão menores que isso. Duas saídas: `RepeatedStratifiedKFold` (mais execuções) ou comparar as alternativas dobra a dobra, já que as dobras são as mesmas para todas. Se trocar o CV, troque em `get_cv()` para valer para todos.
3. **Ajuste sem contaminar a estimativa.** Fazer o `GridSearchCV`/`RandomizedSearchCV` com o mesmo `cv` e reportar o score da melhor configuração é levemente otimista. CV aninhada resolve, mas custa mais. O importante é usar o mesmo orçamento de busca para todas as alternativas.
4. **Entrega do notebook.** O notebook final depende de `src/tde2`. Entregar o link do Colab (com `git clone`) ou um .zip do repositório. A outra opção é copiar o conteúdo de `src/tde2` para as primeiras células antes da entrega.

## Checklist do enunciado (seção 8)

- [x] Pré-processamento dentro de pipeline
- [x] Sementes fixadas e documentadas
- [x] Teste isolado para uso único, ao final
- [x] Mesmas divisões para todas as alternativas
- [x] Linha de base (classe majoritária) no esqueleto
- [ ] Membros individuais avaliados nas mesmas condições dos comitês
- [ ] Variabilidade reportada, com as diferenças afirmadas excedendo-a
- [ ] Custo computacional (treino e predição) de cada alternativa
- [ ] Matriz de confusão e análise de erros por classe
- [ ] Diversidade medida e relacionada ao ganho
- [ ] Notebook executa do início ao fim sem intervenção

## Uso de IA generativa

Conforme a seção 7 do enunciado, o relatório deve terminar com a declaração de uso de IA. A estrutura do repositório e o código de dados foram feitos com apoio do **Claude Code (Claude Opus 5.5)**. Os resultados experimentais e a redação da análise devem ser da equipe.
