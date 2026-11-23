# Pendências para decidir em grupo

**Entrega: hoje, 23/11/2026**

A parte de dados está pronta: base, divisão treino/teste, pré-processamento e caracterização. Como usar está no [README](../README.md). Abaixo estão as decisões que faltam, na ordem em que travam o trabalho. Anotem a decisão em cada item e façam commit deste arquivo, para ficar registrado o que foi combinado.

---

## 1. Antes de programar (travam o código)

### 1.1 Métrica principal
A métrica define o que o ajuste de hiperparâmetros otimiza. Não dá para trocar depois sem refazer tudo.

- **Sugestão:** acurácia balanceada (já em `config.PRIMARY_METRIC`), reportando também F1-macro, recall da classe doença e ROC-AUC.
- **Por quê:** a base é levemente desbalanceada (55% positivos), e a acurácia balanceada funciona com votação dura. Com a métrica F1 da classe positiva, o classificador que sempre responde "doença" marca 0,71 e engana a leitura.

> **Decisão (proposta do Fernando, implementada):** acurácia balanceada como métrica principal (é a que a busca otimiza), mais F1-macro, recall da classe doença e ROC-AUC.

### 1.2 Comitê heterogêneo: membros e regra de fusão
O enunciado exige que a regra de fusão seja explicitada.

- **Sugestão de membros:** regressão logística, KNN, SVM, árvore de decisão e Naive Bayes, que têm vieses indutivos diferentes e por isso aumentam a diversidade.
- **Regra de fusão:** votação dura, votação suave ou empilhamento (*stacking*).

> **Decisão (proposta do Fernando, implementada):** regressão logística, KNN, SVM RBF (probabilidades por `CalibratedClassifierCV(SVC(), ensemble=False)`, porque `SVC(probability=True)` está obsoleto no scikit-learn 1.9), árvore de decisão e Naive Bayes gaussiano. Fusão principal: **votação suave** (média simples das probabilidades). A votação dura também é avaliada, para comparação.

### 1.3 Comitês homogêneos
O membro-base de cada um também precisa ser avaliado sozinho, nas mesmas condições.

- **Redução de variância:** bagging ou Random Forest.
- **Redução de viés:** AdaBoost, `GradientBoostingClassifier` ou `HistGradientBoostingClassifier`.

> **Decisão (proposta do Fernando, implementada):** variância: Random Forest e Bagging de árvores (membro-base: árvore de decisão). Viés: AdaBoost com tocos (membro-base: toco de decisão, avaliado sozinho) e Gradient Boosting (referência: árvore de decisão).

### 1.4 Orçamento de ajuste
A rubrica penaliza quando uma alternativa recebe ajuste visivelmente mais cuidadoso que as outras.

- Usar o mesmo número de configurações testadas para cada alternativa (ex.: `RandomizedSearchCV` com o mesmo `n_iter`, `cv=get_cv()` e `random_state=C.SEED`).
- Validação cruzada aninhada: é mais correta, porque o score da melhor configuração na busca é levemente otimista, mas custa mais tempo.

> **Decisão (proposta do Fernando, implementada):** CV aninhada. Laço externo = `get_cv()` (10 dobras, as mesmas para todos); laço interno = 5 dobras. `RandomizedSearchCV(n_iter=20, random_state=42)` para cada membro e cada comitê homogêneo; no heterogêneo, cada membro é a sua própria busca com o mesmo orçamento. A configuração final (congelada) é a melhor de uma busca com o mesmo orçamento sobre o treino inteiro.

### 1.5 Padronização para modelos de árvore
- `build_preprocessor(scale=True)` para todos simplifica o argumento de "mesmas condições".
- `scale=False` nas árvores dá o mesmo resultado, mas o relatório precisa explicar a diferença.

> **Decisão (proposta do Fernando, implementada):** `scale=True` para todos.

---

## 2. Durante a análise

### 2.1 Como afirmar que uma alternativa supera outra
Com 10 dobras, o desvio da regressão logística já é ±0,04. As diferenças entre comitês provavelmente serão menores que isso.

- **Opção A:** repetir a validação cruzada (`RepeatedStratifiedKFold`). Se for essa, troquem dentro de `get_cv()` para valer para todos.
- **Opção B:** comparar as alternativas dobra a dobra, já que as dobras são as mesmas para todas.
- Combinem um critério do tipo "só dizemos que A supera B se...".

> **Decisão (proposta do Fernando, implementada):** opção B, comparação dobra a dobra. Só se afirma que A supera B se o teste t pareado corrigido de Nadeau e Bengio der p < 0,05 **e** a diferença média for maior que o desvio-padrão das diferenças entre dobras.

### 2.2 Medida de diversidade
O enunciado pede relacionar a diversidade ao ganho e discutir se o comitê que ganhou mais é também o mais diverso.

- **Opções:** discordância par a par, estatística Q de Yule ou kappa, calculadas sobre as predições fora da dobra (`cross_val_predict(..., cv=get_cv())`).

> **Decisão (proposta do Fernando, implementada):** discordância e Q de Yule sobre as saídas oráculo (acertou/errou), nas dobras de validação externas. Heterogêneo: entre os 5 membros. RF, Bagging e AdaBoost: entre os estimadores internos. Gradient Boosting: não se aplica (estimadores internos são regressores).

### 2.3 Congelar configurações e rodar o teste
O conjunto de teste (`load_test()`) é usado **uma única vez**, depois que todos concordarem com as configurações finais. Rodar de novo após ver o resultado invalida o protocolo.

> **Quem roda:** Fernando · **Quando:** 23/11/2026, na execução final de `notebooks/02_modelagem.ipynb` (seção 13). As configurações congeladas estão em `reports/tables/02_hiperparametros_congelados.csv`.

---

## 3. Entrega

### 3.1 Formato do notebook
O notebook depende da pasta `src/`. O professor precisa conseguir executá-lo do início ao fim sem intervenção.

- **Link do Colab com `git clone`:** exige repositório público ou com acesso dado ao professor.
- **.zip do repositório:** funciona sem depender do acesso ao GitHub.
- **Notebook autocontido:** copiar o código de `src/tde2` para as primeiras células antes de entregar.

> **Decisão:**

### 3.2 Pontos a registrar no relatório
- A justificativa da divisão 2/3 treino e 1/3 teste: mais precisão na nota final (306 exemplos de teste), ao custo de menos dados de treino.
- A partição desigual dos zeros de colesterol: 17,8% no treino contra 20,6% no teste, e o único `RestingBP = 0` caiu no teste. O pipeline lida com isso, mas é característica da partição.
- O material da seção 4.1 está em `notebooks/01_caracterizacao_dados.ipynb` e as figuras em `reports/figures/`.

### 3.3 Declaração de uso de IA (todos)
Cada integrante informa quais ferramentas usou e para quê, para montar a declaração única do final do relatório. O enunciado só permite IA para apoio à escrita, depuração de código e explicação de conceitos. Os resultados experimentais e a redação da análise precisam ser da equipe.

- **Renato:** Claude Code (Claude Opus 5.5), para estruturação do repositório e depuração do código da parte de dados.
- **Fernando:** Claude Code (Claude Opus 5.5), para a depuração do código da parte de modelagem (`src/tde2/models.py`, `src/tde2/evaluation.py`, `notebooks/02_modelagem.ipynb`) e explicação de conceitos.
- **Ângelo:**
