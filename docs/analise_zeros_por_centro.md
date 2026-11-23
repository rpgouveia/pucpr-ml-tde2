# Origem dos zeros de Cholesterol e RestingBP por centro

Verificação das afirmações do relatório do TDE 2 ([reports/relatorio_tde2_v2.pdf](../reports/relatorio_tde2_v2.pdf)) sobre os 172 registros com `Cholesterol = 0` e o registro com `RestingBP = 0` do Heart Failure Prediction Dataset (Kaggle, fedesoriano). Cada linha do Kaggle foi rastreada até o centro de origem nos dados da UCI (Heart Disease, id 45).

- **Script:** [scripts/analise_zeros_por_centro.py](../scripts/analise_zeros_por_centro.py). Só lê os dados e imprime as tabelas deste documento.
- **Reproduzir:** `uv run python scripts/analise_zeros_por_centro.py`
- **Ambiente:** Python 3.12, pandas 3.1.1, scikit-learn 1.9.1 (o `.venv` do projeto)

## 1. Resumo

| # | Afirmação do relatório | Onde | Veredito | Evidência principal |
|---|---|---|---|---|
| a | "a maior parte [dos zeros] provém da base de Switzerland" | Seção 1 | **Confirmada** | 123 dos 172 (71,5%) são de Switzerland. Os outros 49 (28,5%) são de Long Beach VA (§4.2). |
| b | Switzerland "não registra colesterol" | Seção 1 | **Confirmada nos dados** | Os 123 registros de Switzerland têm `chol = 0`. A documentação da UCI não diz por quê (§3). |
| c | "praticamente todas as duplicatas removidas correspondem aos registros da Statlog" | Seção 1 | **Confirmada, com ressalva** | 272 = 270 (a Statlog inteira) + 2 duplicatas internas da UCI. A Statlog não está disponível para comparar registro a registro (§6). |
| d | "A saída do pipeline tem 19 colunas" | Seção 2.2 | **Confirmada** | O único `RestingBP = 0` (row_id 449) está no teste. As 10 dobras e o ajuste final dão 19 colunas (§7). |
| e | "89,9% dos pacientes com colesterol zerado (98 de 109) têm a doença" | Seção 1 | **Confirmada** | 98/109 no treino (§5). |
| f | o indicador "reflete a forma de coleta de bases específicas, sobretudo a de Switzerland" | Seção 6 | **Confirmada** | No treino, 80 dos 109 zeros (73,4%) são de Switzerland e 29 de VA (§5). |
| g | a ausência é MNAR porque "o fato de o valor faltar está associado ao desfecho" | Seção 1 | **Os dados não sustentam a justificativa** | A associação com o desfecho vem do centro. Em VA, o único centro com os dois casos, a prevalência é igual com e sem colesterol: 75,5% × 74,0% (§5). |

Três achados que o relatório não menciona:

1. **O 172 não conta todos os colesterois ausentes.** A UCI tem 202 ausentes. O Kaggle só manteve zerados os 172 que vinham gravados como `0`. Os outros 30, gravados como `?`, aparecem no Kaggle com valores entre 110 e 295 mg/dl, preenchidos pelo autor por um método que não foi possível identificar (§4.4).
2. **42% das linhas do Kaggle têm algum valor preenchido.** São 387 das 918 linhas com pelo menos um `?` na origem, em atributos como `ST_Slope`, `MaxHR`, `Oldpeak` e `FastingBS`. "Sem NaN" não significa "sem ausentes" (§4.5).
3. **O indicador de colesterol ausente funciona como um marcador de centro.** Ele vale 1 em todos os pacientes de Switzerland e em cerca de 1/4 dos de VA, e vale 0 em todos os de Cleveland e Hungarian (§5).

## 2. Sugestões de redação para o relatório v2

| Seção | Texto atual | Sugestão |
|---|---|---|
| 1, Origem | "o que resultou em 918 pacientes sem valores nulos explícitos (NaN)" | "…918 pacientes sem valores nulos explícitos (NaN). Em 387 deles (42%), valores ausentes nas bases originais foram preenchidos pelo autor do conjunto, por método não documentado." |
| 1, Origem | "praticamente todas as duplicatas removidas correspondem aos registros da Statlog (Heart)" | "270 das 272 duplicatas removidas correspondem aos registros da Statlog (Heart). As outras duas são registros repetidos dentro das bases Hungarian e Long Beach VA." |
| 1, Ausência informativa | "a maior parte provém da base de Switzerland, que não registra colesterol" | "123 provêm da base de Switzerland, em que todos os registros têm colesterol igual a zero, e 49 da de Long Beach VA" |
| 1, Ausência informativa | "Trata-se de ausência não aleatória (MNAR; Rubin, 1976): o fato de o valor faltar está associado ao desfecho." | "A ausência é estrutural: depende do centro de origem, e não do paciente. A associação com o desfecho decorre da prevalência alta desses centros (93,5% em Switzerland). Em Long Beach VA, a prevalência é a mesma com e sem colesterol registrado (75,5% e 74,0%)." |
| 2.2, item 1 | "O indicador binário resultante (missingindicator_Cholesterol) é acrescentado como atributo." | Acrescentar: "Não há indicador para RestingBP porque o único zero dessa coluna está no conjunto de teste. No teste, ele é imputado pela mediana do treino (130 mmHg)." |
| 6, Centros | "…reflete a forma de coleta de bases específicas, sobretudo a de Switzerland." | Pode ser reforçado: "…na prática, o indicador identifica o centro de origem (todos os pacientes de Switzerland e cerca de 1/4 dos de Long Beach VA)." |

O rótulo MNAR não pode ser testado com estes dados, porque isso exigiria conhecer os colesterois que faltam. Mas a justificativa dada no relatório ("associado ao desfecho") não é a definição de MNAR, e os dados explicam essa associação pelo centro. Por isso a sugestão é descrever a ausência sem o rótulo.

Outros pontos do repositório com o mesmo erro (**não** alterados):

- [src/tde2/config.py:36](../src/tde2/config.py#L36): "Cholesterol = 0 em 172 linhas (vêm da base Switzerland)". O correto é 123 de Switzerland e 49 de Long Beach VA.
- [notebooks/01_caracterizacao_dados.ipynb](../notebooks/01_caracterizacao_dados.ipynb): diz que os zeros "vêm quase todos da base Switzerland". O correto é "a maior parte" (71,5%).

## 3. Dados usados e ausências por centro

Arquivos processados da UCI, soltos em `dataset/uci/` (sem `.zip`): `processed.cleveland.data`, `processed.hungarian.data`, `processed.switzerland.data` e `processed.va.data`. São CSV sem cabeçalho, com `?` = ausente e 14 colunas (`age … num`). Os arquivos brutos, com 76 atributos, não foram usados. No Kaggle, o arquivo é `dataset/heart.csv`, com 918 linhas e nenhum NaN.

| Centro | Registros | chol `?` | chol = 0 | **chol ausente** | trestbps `?` | trestbps = 0 | **trestbps ausente** | % dos chol ausentes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cleveland | 303 | 0 | 0 | **0** | 0 | 0 | **0** | 0,0 |
| Hungarian | 294 | 23 | 0 | **23** | 1 | 0 | **1** | 11,4 |
| Switzerland | 123 | 0 | 123 | **123** | 2 | 0 | **2** | 60,9 |
| Long Beach VA | 200 | 7 | 49 | **56** | 56 | 1 | **57** | 27,7 |
| **Total** | **920** | 30 | **172** | **202** | 59 | **1** | **60** | 100 |

O 172 do Kaggle é exatamente o total de `chol = 0`, e não o de ausentes (202). Com `trestbps` acontece o mesmo: dos 60 ausentes, só o único `0` aparece zerado no Kaggle. O `heart-disease.names` diz apenas que ausentes são `-9` nos brutos, que viram `?` nos processados. Ele não explica por que Switzerland tem `0` em vez de `?`.

`?` nas outras colunas:

| Centro | trestbps | chol | fbs | restecg | thalach | exang | oldpeak | slope | ca | thal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cleveland | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 2 |
| Hungarian | 1 | 23 | 8 | 1 | 1 | 1 | 0 | 190 | 291 | 266 |
| Switzerland | 2 | 0 | 75 | 1 | 1 | 1 | 6 | 17 | 118 | 52 |
| Long Beach VA | 56 | 7 | 7 | 0 | 53 | 53 | 56 | 102 | 198 | 166 |

## 4. Cruzamento Kaggle × UCI

### 4.1 Regra de casamento

Cada linha do Kaggle foi comparada com as 920 da UCI pelos 11 atributos e pelo alvo. As conversões foram: sex 1→M; cp 1..4→TA/ATA/NAP/ASY; restecg 0..2→Normal/ST/LVH; exang 1→Y; slope 1..3→Up/Flat/Down; `num > 0` ↔ `HeartDisease = 1`. Números foram comparados com `np.isclose`, e `oldpeak` foi arredondado para 1 casa, para que `1.0` e `1` sejam iguais.

- **Cholesterol:** `?` ou `0` na UCI ↔ `0` no Kaggle. Comparação estrita.
- **Demais colunas:** `?` na UCI é **curinga** e aceita qualquer valor, porque o Kaggle não tem ausentes. `ca` e `thal` não existem no Kaggle.
- **Status:** a linha é **única** quando os candidatos vêm de um só centro, **ambígua** quando vêm de mais de um centro e **não encontrada** quando não há candidato.

### 4.2 As 172 linhas com Cholesterol = 0

| Status | Todos os candidatos | Preferindo candidato exato (sem curinga) |
|---|---:|---:|
| Única | **170** | 171 |
| Ambígua | **2** | 1 |
| Não encontrada | **0** | 0 |

69 linhas têm candidato exato. As outras 103 dependem de curinga, principalmente em `fbs` (75 `?` em Switzerland).

| Centro | Todos os candidatos | Após exclusão (1-para-1) |
|---|---:|---:|
| Switzerland | 121 | **123** |
| Long Beach VA | 49 | **49** |
| Ambígua (VA ou Switzerland) | 2 | 0 |

**Resolução das 2 ambíguas por exclusão.** Nenhum registro da UCI é par de mais de uma linha "única". Nas duas ambíguas, o candidato de VA já é o par único de outra linha, e o de Switzerland está livre:

| row_id Kaggle | Candidato UCI | Já é par único de |
|---|---|---|
| 293 | Switzerland, linha 108 (o único exato) | — |
| 293 | Long Beach VA, linha 69 | row_id 484 |
| 401 | Switzerland, linha 34 | — |
| 401 | Long Beach VA, linha 27 | row_id 442 |

Supondo que cada registro da UCI gerou no máximo uma linha do Kaggle, as duas ambíguas são de Switzerland. O resultado é um casamento 1-para-1 perfeito entre os 172 zeros do Kaggle e os 172 `chol = 0` da UCI. Sem essa suposição, o veredito (a) continua valendo: são no mínimo 121 de 172 (70,3%).

### 4.3 A linha com RestingBP = 0

O row_id 449 tem correspondência **única e exata** (sem curinga) em **Long Beach VA**, linha 34 de `processed.va.data`: `55,1,3,0,0,0,0,155,0,1.5,2,?,?,3`. No Kaggle: `55, M, NAP, 0, 0, 0, Normal, 155, N, 1.5, Flat, 1`. É a única linha com os dois zeros e faz parte dos 49 de VA.

### 4.4 Os 30 colesterois `?`

Os 30 registros com `chol = ?` (23 Hungarian + 7 VA) não têm par entre os zeros do Kaggle. Casando esses registros **sem comparar o colesterol**:

- cada um dos 30 tem exatamente 1 par, e os pares formam **29 linhas distintas**, porque o par duplicado de Hungarian (§6) vira uma linha só;
- essas 29 são exatamente as linhas "não encontradas" do casamento geral (§4.6);
- o Kaggle traz para elas colesterol **diferente de zero**: mínimo 110, mediana 185, máximo 295, 26 valores distintos.

Os valores variam, então não houve imputação por uma constante. O método não pode ser identificado com os arquivos disponíveis. O `reprocessed.hungarian.data` também tem 23 ausentes (`-9`) no colesterol, então não é a fonte. **Consequência:** no Kaggle, `Cholesterol = 0` marca só os ausentes codificados como `0` na origem. Outros 29 pacientes sem colesterol medido têm valores que parecem medições reais.

### 4.5 Valores preenchidos fora do colesterol

| Centro | Registros | Com algum `?` nos 11 atributos | Com `?` fora do colesterol | % |
|---|---:|---:|---:|---:|
| Cleveland | 303 | 0 | 0 | 0,0 |
| Hungarian | 294 | 199 | 194 | 67,7 |
| Switzerland | 123 | 77 | 77 | 62,6 |
| Long Beach VA | 200 | 113 | 110 | 56,5 |
| **Total** | 920 | 389 | 381 | 42,3 |

Sem as 2 cópias duplicadas, **387 das 918 linhas do Kaggle (42,2%)** têm algum valor que era ausente na origem e foi preenchido pelo autor. A maior parte é `ST_Slope`, com 309 `?` somando os três centros.

### 4.6 Todas as 918 linhas

| Centro | Únicas | + ambíguas (§4.2) | + colesterol preenchido (§4.4) | **No Kaggle** | Na UCI | Diferença |
|---|---:|---:|---:|---:|---:|---|
| Cleveland | 303 | 0 | 0 | **303** | 303 | 0 |
| Hungarian | 271 | 0 | 22 | **293** | 294 | −1 (duplicata) |
| Switzerland | 121 | 2 | 0 | **123** | 123 | 0 |
| Long Beach VA | 192 | 0 | 7 | **199** | 200 | −1 (duplicata) |
| **Total** | 887 | 2 | 29 | **918** | 920 | −2 |

Todas as 918 linhas têm origem nos quatro centros. Nenhuma vem só da Statlog.

## 5. Centro, colesterol zerado e prevalência

Com o centro de cada linha (§4.6) e a partição de `dataset/splits/`:

| Centro | Linhas | Treino | Teste | Chol = 0 | Chol = 0 no treino | Chol = 0 no teste |
|---|---:|---:|---:|---:|---:|---:|
| Cleveland | 303 | 203 | 100 | 0 | 0 | 0 |
| Hungarian | 293 | 202 | 91 | 0 | 0 | 0 |
| Switzerland | 123 | 80 | 43 | 123 | 80 | 43 |
| Long Beach VA | 199 | 127 | 72 | 49 | 29 | 20 |
| **Total** | 918 | 612 | 306 | **172** | **109** | **63** |

Prevalência da doença (positivos/total):

| Centro | Base: Chol = 0 | Base: Chol > 0 | Treino: Chol = 0 | Treino: Chol > 0 |
|---|---:|---:|---:|---:|
| Cleveland | — | 139/303 (45,9%) | — | 98/203 (48,3%) |
| Hungarian | — | 106/293 (36,2%) | — | 69/202 (34,2%) |
| Switzerland | 115/123 (93,5%) | — | 77/80 (96,2%) | — |
| Long Beach VA | 37/49 (75,5%) | 111/150 (74,0%) | 21/29 (72,4%) | 74/98 (75,5%) |
| **Total** | **152/172 (88,4%)** | 356/746 (47,7%) | **98/109 (89,9%)** | 241/503 (47,9%) |

Leitura:

- **(e)** O "98 de 109 (89,9%)" do relatório está correto.
- **(f)** 73,4% dos zeros do treino são de Switzerland. A Seção 6 ("sobretudo a de Switzerland") está correta.
- **(g)** Long Beach VA é o único centro com pacientes com e sem colesterol. Ali, a prevalência praticamente não muda: 75,5% × 74,0% na base e 72,4% × 75,5% no treino. A diferença de 89,9% × 47,9% no treino vem de quem tem zero: em 3/4 dos casos são pacientes de Switzerland, centro com 93,5% de doentes, contra 36–46% em Cleveland e Hungarian. O indicador funciona, na prática, como uma pista do centro de origem. Ressalva: em VA há só 49 zeros, então a comparação tem pouca precisão.

## 6. Soma dos centros e duplicatas

| Verificação | Valor |
|---|---|
| Soma dos quatro centros | **920** (esperado: 920 ✔) |
| Duplicatas exatas entre os centros (14 colunas, `?` = `?`) | **2** cópias extras (4 linhas envolvidas) |
| Duplicatas pelos 11 atributos convertidos + alvo | 2 cópias extras (as mesmas) |
| Duplicatas no CSV do Kaggle | 0 |

| Centro | Linhas | Registro |
|---|---|---|
| Hungarian | 102 e 103 | `49,0,2,110,?,0,0,160,0,0,?,?,?,0` |
| Long Beach VA | 140 e 188 | `58,1,3,150,219,0,1,118,1,0,?,?,?,2` |

Contas:

- 920 + 270 (Statlog) = 1190.
- 1190 − 918 = **272** removidas, como informa o autor.
- 920 − 918 = **2**, exatamente as duplicatas internas.
- Logo, 272 − 2 = **270** removidas da Statlog, ou seja, a Statlog inteira.

Isso é coerente com §4.6: Cleveland aparece completa e nenhuma linha fica sem origem nos quatro centros.

## 7. RestingBP = 0 e colunas do pipeline

| Verificação | Resultado |
|---|---|
| Partição do row_id 449 | **`dataset/splits/test.csv`** |
| `RestingBP = 0` no treino / teste | 0 / 1 |
| Dobras externas em que a linha cai no treino da dobra | **0 de 10**, porque ela não está no treino |

Mesmo assim, `build_preprocessor(scale=True)` (com `SimpleImputer(missing_values=0, add_indicator=True)`) foi ajustado em cada dobra externa de `get_cv()` e no treino inteiro:

| Ajuste | n treino | RestingBP = 0 no ajuste | Colunas | `missingindicator_RestingBP` | `missingindicator_Cholesterol` |
|---|---:|---:|---:|---|---|
| Dobras 1–2 | 550 | 0 | **19** | não | sim |
| Dobras 3–10 | 551 | 0 | **19** | não | sim |
| Treino inteiro (final) | 612 | 0 | **19** | não | sim |

Colunas: RestingBP, Cholesterol, missingindicator_Cholesterol, Age, MaxHR, Oldpeak, FastingBS, Sex_M, ChestPainType_ASY, ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA, RestingECG_LVH, RestingECG_Normal, RestingECG_ST, ExerciseAngina_Y, ST_Slope_Down, ST_Slope_Flat, ST_Slope_Up.

- **Teste:** transformado com o ajuste final, `test.csv` vira uma matriz (306, 19). O zero da linha 449 é imputado pela mediana do treino (130 mmHg; a de Cholesterol é 239), sem indicador. O `SimpleImputer` imputa todas as colunas, mas só cria indicador para as que tinham ausentes no ajuste, e seu indicador interno usa `error_on_new=False`, por isso não há erro.
- **Contrafactual** (não é o procedimento do projeto): com a linha 449 no ajuste, a saída teria **20 colunas** e incluiria `missingindicator_RestingBP`. As 19 colunas dependem de ela ter caído no teste.
- **CV interna:** a CV interna usa subconjuntos do treino de cada dobra externa, então também gera 19 colunas. Isso é dedução, não foi rodado.

## 8. Limitações

- **Statlog:** o arquivo não está em `dataset/uci`. O veredito (c) vem da contagem, não do pareamento registro a registro.
- **Curingas:** tratar `?` como curinga pode, em tese, gerar pares falsos. Contra isso: nenhum registro da UCI é par de duas linhas "únicas", e as contagens por centro fecham com a UCI menos as duplicatas.
- **Exclusão 1-para-1:** a resolução das 2 ambíguas supõe que cada registro da UCI gerou no máximo uma linha do Kaggle. Sem essa suposição, elas continuam ambíguas e o veredito (a) não muda.
- **Preenchimento do autor:** não foi possível identificar o método usado nos 387 registros com valores preenchidos.
- **MNAR:** o mecanismo de ausência não pode ser testado sem os valores que faltam. A análise mostra apenas que o centro explica a associação com o desfecho.
