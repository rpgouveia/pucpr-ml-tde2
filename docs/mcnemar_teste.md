# Escolha do modelo pela VC e teste de McNemar no conjunto de teste

Apoio à revisão da Seção 6.1 do relatório ([reports/relatorio_tde2_v2.pdf](../reports/relatorio_tde2_v2.pdf)). A ideia é escolher o modelo pela validação cruzada aninhada (VC) e usar o teste apenas como confirmação.

> **Situação (23/11/2026).** Os itens 2 e 3 estão concluídos, com base nas tabelas exportadas. Os itens 4 a 7 dependem de [scripts/mcnemar_teste.py](../scripts/mcnemar_teste.py), que precisa rodar no Mac onde o notebook foi executado (§7). Nesta máquina (Windows), o Random Forest não reproduz a matriz de confusão do relatório (§5). Mesmo assim, as respostas (ii) e (iii) já podem ser dadas: elas dependem só das contagens do relatório (§4). Depois da execução no Mac, as seções marcadas como **pendente** serão preenchidas com `docs/mcnemar_teste/resultados.md`.

> **Inconsistência na Tabela 2 do relatório (§6).** Os espaços de busca de todos os modelos e os hiperparâmetros finais de 6 dos 10 modelos ajustados não correspondem ao código nem aos valores congelados que geraram os resultados. Isso inclui o Random Forest citado na Seção 6.1. Os resultados do relatório estão corretos; a tabela, não.

## 1. Itens 2 e 3: a VC, com as tabelas exportadas

Fontes: [02_resultados_cv.csv](../reports/tables/02_resultados_cv.csv), com média ± DP nas 10 dobras externas, e [02_erros_por_classe_cv.csv](../reports/tables/02_erros_por_classe_cv.csv), com os FN das predições fora da dobra (N = 612; 339 doentes). Não foi preciso rodar nada.

Ordenado por ROC-AUC:

| # AUC | # Sens. | Modelo | Família | Acurácia balanceada | Sensibilidade | ROC-AUC | FN (de 339) |
|---:|---:|---|---|---|---|---|---:|
| 1 | 2 | **Random Forest** | homogêneo (variância) | 0,8474 ± 0,0296 | 0,8998 ± 0,0606 | **0,9292 ± 0,0183** | 34 |
| 2 | 6 | Votação suave | heterogêneo | 0,8477 ± 0,0387 | 0,8822 ± 0,0535 | 0,9283 ± 0,0149 | 40 |
| 3 | 7 | Regressão logística | membro | 0,8510 ± 0,0332 | 0,8703 ± 0,0461 | 0,9281 ± 0,0193 | 44 |
| 4 | 3 | KNN | membro | **0,8532 ± 0,0236** | 0,8968 ± 0,0421 | 0,9223 ± 0,0196 | 35 |
| 5 | 8 | Gradient Boosting | homogêneo (viés) | 0,8458 ± 0,0319 | 0,8672 ± 0,0506 | 0,9211 ± 0,0255 | 45 |
| 6 | 1 | **Bagging de árvores** | homogêneo (variância) | 0,8505 ± 0,0323 | **0,9057 ± 0,0567** | 0,9204 ± 0,0177 | **32** |
| 7 | 9 | AdaBoost | homogêneo (viés) | 0,8469 ± 0,0392 | 0,8583 ± 0,0461 | 0,9198 ± 0,0183 | 48 |
| 8 | 4 | SVM (RBF) | membro | 0,8408 ± 0,0337 | 0,8939 ± 0,0395 | 0,9186 ± 0,0192 | 36 |
| 9 | 10 | Naive Bayes | membro | 0,8418 ± 0,0397 | 0,8556 ± 0,0593 | 0,9139 ± 0,0235 | 49 |
| 10 | 12 | Árvore de decisão | membro | 0,8211 ± 0,0201 | 0,8437 ± 0,0636 | 0,8962 ± 0,0226 | 53 |
| 11 | 11 | Toco de decisão | membro | 0,8106 ± 0,0266 | 0,8554 ± 0,0686 | 0,8106 ± 0,0266 | 49 |
| — | 5 | Votação dura | heterogêneo | 0,8487 ± 0,0365 | 0,8880 ± 0,0410 | — (sem probabilidades) | 38 |

Ordenado por sensibilidade: Bagging 0,9057 > **RF 0,8998** > KNN 0,8968 > SVM 0,8939 > Votação dura 0,8880 > Votação suave 0,8822 > Regressão logística 0,8703 > Gradient Boosting 0,8672 > AdaBoost 0,8583 > Naive Bayes 0,8556 > Toco 0,8554 > Árvore 0,8437.

- **(a) Sim, o RF tem a maior ROC-AUC média**: 0,9292, contra 0,9283 da Votação suave, que fica em segundo. A diferença é de **+0,0010**, cerca de 1/20 do DP entre dobras (0,018). A Regressão logística vem logo depois, com 0,9281.
- **(b) A maior sensibilidade média é a do Bagging**: 0,9057, contra 0,8998 do RF. A diferença é de **0,59 ponto percentual**. Nas predições fora da dobra, isso equivale a **2 falsos negativos**: 32 do Bagging contra 34 do RF, em 339 doentes.
- **Contexto.** Na acurácia balanceada, a métrica primária usada na busca de hiperparâmetros, o RF fica em 6º de 12 (0,8474; o líder é o KNN, com 0,8532). A tabela exportada [02_comparacoes_pareadas.csv](../reports/tables/02_comparacoes_pareadas.csv), com o teste de Nadeau-Bengio sobre essa métrica, não aponta diferença significativa: RF × KNN tem diferença −0,0058 e p = 0,58; RF × Bagging, −0,0031 e p = 0,73; Votação suave × RF, +0,0003 e p = 0,98.
- **O argumento que a VC sustenta.** O RF é o **único modelo entre os dois primeiros nos dois critérios**: 1º em ROC-AUC e 2º em sensibilidade. O Bagging é 1º em sensibilidade, mas 6º em ROC-AUC. A Votação suave é 2ª em ROC-AUC, mas 6ª em sensibilidade.

## 2. Item 4: RF dobra a dobra, com Nadeau-Bengio (pendente)

**Pendente: execução no Mac.** Os valores de ROC-AUC e sensibilidade por dobra não foram exportados pelo notebook. O script refaz a VC aninhada do RF, da Votação suave, do Bagging e do KNN e só calcula o teste se as médias, os DPs, os FN e as diferenças por dobra da acurácia balanceada reproduzirem as tabelas exportadas.

| A | B | Métrica | Diferença média | DP da diferença | t corrigido | p (9 g.l.) |
|---|---|---|---|---|---|---|
| Random Forest | Votação suave (2º em ROC-AUC) | ROC-AUC | pendente | pendente | pendente | pendente |
| Random Forest | Bagging (1º em sensibilidade) | sensibilidade | pendente | pendente | pendente | pendente |

O teste é `evaluation.paired_comparison()`. Ele usa a variância corrigida (1/k + n_teste/n_treino) · s², com k = 10 e n_teste/n_treino = 1/9, e um t com 9 graus de liberdade.

## 3. Item 5: reprodução das matrizes de confusão no teste

| Modelo | Relatório (VN/FP/FN/VP) | Windows (esta máquina) | Mac |
|---|---|---|---|
| Random Forest | 115 / 22 / 12 / 157 | **116 / 21** / 12 / 157 | pendente |
| KNN | 119 / 18 / 15 / 154 | 119 / 18 / 15 / 154 | pendente |
| Bagging | 117 / 20 / 18 / 151 | 117 / 20 / 18 / 151 (ROC-AUC difere em 6,5 × 10⁻⁵) | pendente |
| Regressão logística | 116 / 21 / 17 / 152 | 116 / 21 / 17 / 152 | pendente |
| Votação suave | 118 / 19 / 17 / 152 | 118 / 19 / 17 / 152 | pendente |

Por isso os testes não foram calculados nesta máquina. Ver §5.

## 4. Itens 6 e 7: McNemar exato

### O que já se sabe pelas contagens do relatório

Seja A = Random Forest, b = pacientes que A acerta e B erra, e c = pacientes que A erra e B acerta. Então **b − c = (erros de B) − (erros de A)**, e esse valor sai direto das matrizes de confusão. O teste exato bilateral tem p = min(1, 2 · P[Bin(b + c, ½) ≤ min(b, c)]). Para uma diferença |b − c| fixa, esse p é não decrescente em min(b, c); isso foi verificado numericamente até min(b, c) = 300. Portanto, **o menor p possível é o de c = 0**, que vale 2 · 0,5^|b−c|.

| Versão | B | Erros de A × de B | b − c | Menor p possível | Pode dar p < 0,05? |
|---|---|---|---:|---:|---|
| (a) acerto geral, N = 306 | KNN | 34 × 33 | −1 | 1,000 | não |
| (a) | Bagging | 34 × 38 | +4 | 0,125 | não |
| (a) | Regressão logística | 34 × 38 | +4 | 0,125 | não |
| (a) | Votação suave | 34 × 36 | +2 | 0,500 | não |
| (b) doentes, N = 169 | KNN | 12 × 15 FN | +3 | **0,250** | **não** |
| (b) | Bagging | 12 × 18 FN | +6 | 0,031 | só se c = 0 |
| (b) | Regressão logística | 12 × 17 FN | +5 | 0,0625 | não |
| (b) | Votação suave | 12 × 17 FN | +5 | 0,0625 | não |
| item 7, fora da dobra (339 doentes) | Bagging | 34 × 32 FN | −2 | 0,500 | não |
| item 7 | KNN | 34 × 35 FN | +1 | 1,000 | não |

Com Holm, o menor p-valor de uma família é multiplicado por 4. Assim, o menor p ajustado possível é 4 × 0,125 = **0,5** na versão (a) e 4 × 0,031 = **0,125** na versão (b). **Nenhum par pode ser significativo depois da correção, seja qual for o pareamento.** A ressalva é que isso vale para as matrizes do relatório, que a execução no Mac precisa confirmar.

### Versão (a): acerto geral, N = 306 (pendente)

| B | Ambos acertam | b (RF acerta, B erra) | c (RF erra, B acerta) | Ambos erram | p exato | p Holm |
|---|---:|---:|---:|---:|---:|---:|
| KNN | pendente | pendente | pendente | pendente | pendente | pendente |
| Bagging | pendente | pendente | pendente | pendente | pendente | pendente |
| Regressão logística | pendente | pendente | pendente | pendente | pendente | pendente |
| Votação suave | pendente | pendente | pendente | pendente | pendente | pendente |

### Versão (b): sensibilidade, 169 doentes (pendente)

Acerto = detectar a doença (predição 1).

| B | Ambos detectam | b (RF detecta, B não) | c (RF não, B detecta) | Nenhum detecta | p exato | p Holm |
|---|---:|---:|---:|---:|---:|---:|
| KNN | pendente | pendente | pendente | pendente | pendente | pendente |
| Bagging | pendente | pendente | pendente | pendente | pendente | pendente |
| Regressão logística | pendente | pendente | pendente | pendente | pendente | pendente |
| Votação suave | pendente | pendente | pendente | pendente | pendente | pendente |

### Item 7: versão (b) nas predições fora da dobra, só descritiva (pendente)

**Esta análise é descritiva.** Cada predição fora da dobra vem de um modelo diferente: em cada dobra, a busca interna escolhe hiperparâmetros próprios. Por isso, as predições não são 612 observações pareadas de um mesmo par de classificadores, como o teste supõe.

| B | Ambos detectam | b | c | Nenhum detecta | p exato |
|---|---:|---:|---:|---:|---:|
| Bagging | pendente | pendente | pendente | pendente | pendente |
| KNN | pendente | pendente | pendente | pendente | pendente |

## 5. Reprodutibilidade entre máquinas

Os resultados do notebook foram gerados num Mac (a saída cita `/var/folders/...`). Os mesmos modelos, ajustados no Windows com as mesmas versões de bibliotecas, se comportam assim:

| Modelo | Usa reamostragem aleatória? | Reproduz no Windows? |
|---|---|---|
| KNN, Regressão logística | não | exato (matriz e ROC-AUC) |
| Votação suave (inclui uma árvore de decisão) | não | exato |
| AdaBoost (hiperparâmetros do `freeze()`) | não | exato |
| Random Forest | sim (bootstrap) | probabilidades diferem; 1 paciente muda de lado (row_id 21, p = 0,49993, ou row_id 798, p = 0,4989); ROC-AUC difere em 8,6 × 10⁻⁵ |
| Bagging | sim (bootstrap) | mesma matriz, ROC-AUC difere em 6,5 × 10⁻⁵ |
| Gradient Boosting | sim (`subsample`) | o `freeze()` escolhe outros hiperparâmetros (§6c) |

O padrão aponta para os modelos com reamostragem aleatória. A causa exata não foi isolada: pode ser a geração dos índices de bootstrap ou diferenças de ponto flutuante entre sistemas e CPUs. Como as versões das bibliotecas são idênticas (§6d), a diferença vem da plataforma, e não de versões. Por isso o script roda no Mac e para se algo não bater.

## 6. Tabela 2 do relatório × código

Esta seção foi feita só com leitura de arquivos.

**(a) Espaço de busca do Random Forest.** Em [src/tde2/models.py](../src/tde2/models.py), `search_spaces()`, e em [02_espacos_busca.csv](../reports/tables/02_espacos_busca.csv):

| Hiperparâmetro | Código | Tabela 2 |
|---|---|---|
| n_estimators | inteiro em [50, 500] | [50; 400] |
| max_depth | {None, 2, 3, 4, 5, 6, 8, 10, 12} | [3; 12] |
| min_samples_leaf | inteiro em [1, 20] | (ausente) |
| max_features | {sqrt, log2, 0,5, None} | [0,2; 0,8] |

Não bate. E não é só o RF: **nenhum dos 10 espaços de busca da Tabela 2 corresponde ao código**. Exemplos:
- KNN: o código busca n_neighbors em [1, 50], `weights` e `p`; a tabela diz "ímpar em [3; 49]" e `metric`.
- Árvore: o código usa `min_samples_leaf` e `criterion`; a tabela cita `min_samples_split`.
- Naive Bayes: o código vai de 10⁻¹² a 10⁻¹; a tabela, de 10⁻¹¹ a 10⁻⁵.
- Toco: o código busca 3 hiperparâmetros; a tabela diz "fixo".
- Bagging, Gradient Boosting e AdaBoost têm intervalos diferentes, e no Bagging falta `estimator__max_depth`.

**(b) Origem de 250 / 10 / 0,387.** Esses valores não estão em nenhum arquivo do projeto: código, CSVs exportados, saídas do notebook, README, `docs/PENDENCIAS.md` e histórico do git (`git log -S`; o `models.py` tem um único commit). Eles aparecem apenas nos PDFs: o relatório v1 (commit `4fcbede`) já trazia a mesma Tabela 2, e o v2 a herdou. Os valores congelados de fato saem de `freeze()`, estão na saída da célula 34 do notebook e em [02_hiperparametros_congelados.csv](../reports/tables/02_hiperparametros_congelados.csv), e o `freeze()` nesta máquina os reproduz:

| Modelo | Tabela 2 (relatório) | Congelado (notebook/CSV) | Confere? |
|---|---|---|---|
| Regressão logística | C = 0,0885 | C = 0,1695; class_weight = None | não |
| KNN | n_neighbors = 19; euclidean | n_neighbors = 44; p = 1 (manhattan); weights = distance | não |
| SVM | C = 1,185; gamma = 0,0061 | C = 13,15; gamma = 0,0004208; class_weight = None | não |
| Árvore | max_depth = 4; min_samples_split = 2 | max_depth = 10; min_samples_leaf = 12; entropy | não |
| Naive Bayes | var_smoothing = 1,22 × 10⁻⁸ | var_smoothing = 0,003372 | não |
| Toco | max_depth = 1 | min_samples_leaf = 29; entropy; class_weight = None | parcial |
| **Random Forest** | **250; max_depth = 10; max_features = 0,387** | **363; max_depth = 6; max_features = log2; min_samples_leaf = 1** | **não** |
| Bagging | 50; 0,497; 0,550 | 50; 0,4967; 0,5497; estimator__max_depth = 10 | sim (falta max_depth) |
| Gradient Boosting | 320; 0,0433; 5; 0,866 | 320; 0,04328; 5; 0,866 | sim |
| AdaBoost | 264; 0,1061 | 264; 0,1061 | sim |

Os resultados do relatório foram gerados com os valores congelados, não com os da Tabela 2. Com eles, KNN, Bagging, Regressão logística e Votação suave reproduzem exatamente as matrizes do relatório (§3). Ou seja, **a Tabela 2 e a Seção 6.1, que cita "n_estimators = 250, max_depth = 10, max_features = 0,387", precisam ser corrigidas**; os números de desempenho não. A origem dos valores errados não é rastreável no repositório.

**(c) Gradient Boosting.** A Tabela 2 (0,0433 / 5), o CSV e a saída do notebook coincidem: 320 / 0,04328 / 5 / 0,866. Nesta máquina, o `freeze()` dá 393 / 0,01084 / 2 / 0,9162. A diferença é de plataforma (§5), não do relatório. Os 20 candidatos sorteados pela busca deveriam ser os mesmos nas duas máquinas, porque usam `random_state = 42`; o que muda é o desempenho estimado de cada candidato, e portanto o vencedor. Isso não foi verificado aqui, porque exigiria rodar a busca. O GB não entra nos testes, e o script registra essa comparação apenas como informação. O espaço de busca do GB na Tabela 2 também não bate com o código: n_estimators [50; 350] × [50, 500]; learning_rate [0,01; 0,2] × [0,01; 0,5]; max_depth [2; 5] × [1, 5]; e falta `subsample`.

**(d) Versões fixadas.**

| | uv.lock | requirements.txt | Notebook (Mac) | .venv (Windows) |
|---|---|---|---|---|
| Python | ≥ 3.12, < 3.14 (`.python-version`: 3.12) | — | 3.12.15 | 3.12 (patch não registrado) |
| numpy | 2.5.5 | 2.5.5 | 2.5.5 | 2.5.5 |
| pandas | 3.1.1 | 3.1.1 | 3.1.1 | 3.1.1 |
| scikit-learn | 1.9.1 | 1.9.1 | 1.9.1 | 1.9.1 |
| scipy | 1.18.1 | — | 1.18.1 | 1.18.1 |
| joblib / threadpoolctl | 1.6.0 / 3.8.0 | — | (não registrado) | 1.6.0 / 3.8.0 |
| BLAS | — | — | (não registrado) | OpenBLAS 0.3.34, Haswell |

O script registra Python, as bibliotecas, o sistema operacional, a CPU e o BLAS em `docs/mcnemar_teste/ambiente.json`.

## 7. Como executar no Mac

```bash
git switch <branch com o script>
uv sync --frozen        # só se o ambiente não existir: instala exatamente o uv.lock, sem atualizar
uv run python scripts/mcnemar_teste.py --so-verificar-teste   # ~30 s: hiperparâmetros + matrizes no teste
uv run python scripts/mcnemar_teste.py                         # completo: + VC aninhada de 4 modelos + testes
```

- Grava apenas em `docs/mcnemar_teste/`: `ambiente.json`, `hiperparametros_reproduzidos.csv`, `predicoes_teste.csv`, `vc_por_dobra.csv`, `vc_predicoes_fora_da_dobra.csv`, `nadeau_bengio.csv`, `mcnemar_teste.csv`, `mcnemar_vc_fora_da_dobra.csv` e `resultados.md`. Nenhuma tabela de `reports/` é tocada.
- **Termina com código 0** se tudo conferir e os testes forem calculados. **Termina com código 1** se algo divergir; nesse caso, `resultados.md` lista as diferenças e os testes não são calculados.
- Antes dos testes, o script verifica:
  - os hiperparâmetros contra o CSV (só os modelos usados bloqueiam a execução);
  - as 5 matrizes no teste contra o relatório e contra `02_erros_por_classe_teste.csv`;
  - acurácia balanceada, sensibilidade e ROC-AUC no teste contra `02_resultados_teste.csv`;
  - média e DP de cada métrica na VC contra `02_resultados_cv.csv`;
  - VN/FP/FN/VP fora da dobra contra `02_erros_por_classe_cv.csv`;
  - as diferenças por dobra da acurácia balanceada contra `02_comparacoes_pareadas.csv`.
  
  A tolerância é de 10⁻⁹. Comparar as diferenças por dobra garante que os valores dobra a dobra, e não só as médias, são os do notebook.
- Ao terminar, traga `docs/mcnemar_teste/` (por commit ou cópia) para preencher as seções pendentes.

## 8. Métodos e referências

- **McNemar exato:** binomial bilateral sobre os pares discordantes, com `scipy.stats.binomtest(min(b, c), b + c, 0,5)`. Equivale a `statsmodels.stats.contingency_tables.mcnemar(exact=True)`, que não está instalado. As dependências não foram alteradas. A implementação foi conferida contra a fórmula fechada em 6 casos.
- **Holm:** implementado no script (passo descendente com monotonicidade imposta) e conferido com um exemplo calculado à mão. A família tem 4 comparações por versão.
- **Nadeau-Bengio:** `src/tde2/evaluation.py`, função `paired_comparison()`.
- **Referências:**
  - DIETTERICH, T. G. Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, v. 10, n. 7, p. 1895–1923, 1998.
  - NADEAU, C.; BENGIO, Y. Inference for the generalization error. *Machine Learning*, v. 52, n. 3, p. 239–281, 2003.
  - HOLM, S. A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, v. 6, n. 2, p. 65–70, 1979.

## 9. Respostas diretas

**(i) A recomendação do Random Forest pode ser justificada só pela VC?** Sim, como o melhor equilíbrio entre os critérios, mas não como superior. Os números:
- O RF tem a maior ROC-AUC média (0,9292), mas com vantagem de apenas 0,0010 sobre a Votação suave.
- É o 2º em sensibilidade (0,8998), 0,59 ponto percentual e 2 FN (34 × 32) atrás do Bagging.
- É o único modelo entre os dois primeiros nos dois critérios.
- Na acurácia balanceada fica em 6º, sem diferença significativa para o KNN (p = 0,58) nem para o Bagging (p = 0,73).

Uma redação honesta para a Seção 6.1: "na VC, o Random Forest foi o único modelo entre os dois melhores em ROC-AUC (1º) e em sensibilidade (2º), e as diferenças para os demais ficaram dentro da variação entre dobras; o teste confirmou o desempenho (12 FN, sensibilidade de 92,9%)". O item 4 vai quantificar as comparações em ROC-AUC e sensibilidade (**pendente**). Com diferenças de 0,001 em ROC-AUC e 0,6 ponto percentual em sensibilidade, o esperado é p alto, mas isso não pode ser afirmado antes da execução.

**(ii) A vantagem do RF no teste (12 × 15 FN do KNN) é estatisticamente significativa?** **Não.** Entre os 169 doentes, b − c = 15 − 12 = 3. O menor p exato possível, se o RF detectasse todos os doentes que o KNN detecta, é 2 × 0,5³ = **0,25**. Qualquer outro pareamento dá p maior. Na versão (a), com os 306 pacientes, o RF tem até 1 erro a mais que o KNN (34 × 33). O p exato está **pendente** da execução no Mac, mas a conclusão não depende dele.

**(iii) Algum par apresentou diferença significativa depois da correção de Holm?** **Não, e nenhum poderia.** Pelas contagens do relatório, o menor p ajustado possível é 0,5 na versão (a) e 0,125 na versão (b). Mesmo sem correção, só RF × Bagging na versão (b) poderia ficar abaixo de 0,05, e apenas no caso extremo c = 0: nenhum doente detectado pelo Bagging e não pelo RF. Os valores exatos estão **pendentes** e vão confirmar ou descartar esse caso. Na análise descritiva fora da dobra (item 7), o menor p possível é 0,5 contra o Bagging e 1,0 contra o KNN.
