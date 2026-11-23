"""Escolha do modelo pela VC e teste de McNemar no conjunto de teste.

Feito para rodar na MESMA máquina em que notebooks/02_modelagem.ipynb foi
executado, para que as predições reproduzam exatamente o relatório.

Etapas:

1. Registra o ambiente (Python, numpy, scipy, pandas, scikit-learn, SO, BLAS).
2. Recongela os hiperparâmetros com `models.freeze()` (como o notebook) e
   compara com reports/tables/02_hiperparametros_congelados.csv.
3. Reajusta no treino inteiro Random Forest, KNN, Bagging, Regressão logística
   e Votação suave e prediz o teste.
4. Refaz a VC aninhada (mesmo get_cv(), mesmas sementes) para Random Forest,
   Votação suave, Bagging e KNN, guardando métricas por dobra e predições
   fora da dobra.
5. VERIFICAÇÃO: compara tudo com o relatório e com as tabelas exportadas
   (02_resultados_cv.csv, 02_erros_por_classe_cv.csv,
   02_comparacoes_pareadas.csv, 02_erros_por_classe_teste.csv,
   02_resultados_teste.csv). Se algo não bater, para e não calcula os testes.
6. Se tudo bater: Nadeau-Bengio na VC (item 4), McNemar exato no teste com
   Holm (item 6) e McNemar descritivo nas predições fora da dobra (item 7).

Não altera nenhum arquivo existente. Grava só em docs/mcnemar_teste/.
Dependências: apenas as do projeto (McNemar exato via scipy.stats.binomtest;
Holm implementado aqui).

    uv run python scripts/mcnemar_teste.py
    uv run python scripts/mcnemar_teste.py --so-verificar-teste   # pula a VC (rápido)
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tde2 import config as C  # noqa: E402
from tde2 import evaluation as E  # noqa: E402
from tde2 import models as M  # noqa: E402
from tde2.data import get_cv, load_test, load_train  # noqa: E402

TABLES = ROOT / "reports" / "tables"
OUT = ROOT / "docs" / "mcnemar_teste"

MCNEMAR_MODELS = [M.RF, M.KNN, M.BAGGING, M.LOGREG, M.SOFT_VOTING]
CV_MODELS = [M.RF, M.SOFT_VOTING, M.BAGGING, M.KNN]
RF_RIVALS_TEST = [M.KNN, M.BAGGING, M.LOGREG, M.SOFT_VOTING]
RF_RIVALS_OOF = [M.BAGGING, M.KNN]

# Matrizes de confusão no teste citadas no relatório: (VN, FP, FN, VP)
REPORT_TEST_CM = {
    M.RF: (115, 22, 12, 157),
    M.KNN: (119, 18, 15, 154),
    M.BAGGING: (117, 20, 18, 151),
    M.LOGREG: (116, 21, 17, 152),
    M.SOFT_VOTING: (118, 19, 17, 152),
}
# Hiperparâmetros que afetam os modelos usados aqui (a Votação suave usa os 5 membros).
BLOCKING_PARAMS = set(M.MEMBERS) | {M.RF, M.BAGGING}
TOL = 1e-9
METRICS = ["balanced_accuracy", "recall_doenca", "roc_auc"]
LABEL = {"balanced_accuracy": "acurácia balanceada", "recall_doenca": "sensibilidade", "roc_auc": "ROC-AUC"}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
class Report:
    """Acumula a saída em Markdown, imprime e grava em OUT/resultados.md."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    def h(self, title: str, level: int = 2) -> None:
        self("")
        self("#" * level + " " + title)
        self("")

    def table(self, df: pd.DataFrame, index: bool = False) -> None:
        self(md(df, index=index))

    def save(self, name: str = "resultados.md") -> None:
        (OUT / name).write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def md(df: pd.DataFrame, index: bool = False) -> str:
    d = df.reset_index() if index else df

    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return "sim" if v else "**NÃO**"
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.6g}"
        return str(v)

    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def fmt_param(v) -> str:
    """Mesmo formato usado pelo notebook ao exportar 02_hiperparametros_congelados.csv."""
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def cm(y, pred) -> tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def holm(pvalues) -> np.ndarray:
    """p-valores ajustados por Holm (1979), com monotonicidade imposta."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def mcnemar_exact(ok_a: np.ndarray, ok_b: np.ndarray) -> dict:
    """McNemar exato (binomial bilateral sobre os pares discordantes).

    Equivale a statsmodels.stats.contingency_tables.mcnemar(table, exact=True).
    b = A acerta e B erra; c = A erra e B acerta.
    """
    ok_a, ok_b = np.asarray(ok_a, bool), np.asarray(ok_b, bool)
    n11 = int((ok_a & ok_b).sum())
    b = int((ok_a & ~ok_b).sum())
    c = int((~ok_a & ok_b).sum())
    n00 = int((~ok_a & ~ok_b).sum())
    p = 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)
    return {"ambos_acertam": n11, "b (A acerta, B erra)": b, "c (A erra, B acerta)": c,
            "ambos_erram": n00, "n": n11 + b + c + n00, "p_exato": p}


def environment() -> dict:
    env = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "sistema": platform.platform(),
        "maquina": platform.machine(),
        "processador": platform.processor(),
    }
    try:
        from threadpoolctl import threadpool_info

        env["blas_openmp"] = [
            {k: i.get(k) for k in ("internal_api", "version", "num_threads", "architecture")}
            for i in threadpool_info()
        ]
    except Exception as exc:  # pragma: no cover
        env["blas_openmp"] = f"indisponível: {exc}"
    env["referencia_notebook"] = {
        "python": "3.12.15", "numpy": "2.5.5", "scipy": "1.18.1",
        "pandas": "3.1.1", "scikit-learn": "1.9.1",
    }
    return env


# ---------------------------------------------------------------------------
# Programa
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--so-verificar-teste", action="store_true",
                    help="só recongela, reajusta e confere as matrizes no teste (sem VC aninhada)")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    R = Report()
    R("# McNemar e escolha do modelo pela VC: saída do script")
    R(f"Gerado por `scripts/mcnemar_teste.py` em {time.strftime('%Y-%m-%d %H:%M:%S')}.")

    # --- 1. Ambiente ---------------------------------------------------------
    env = environment()
    (OUT / "ambiente.json").write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    R.h("1. Ambiente")
    ref = env["referencia_notebook"]
    R.table(pd.DataFrame([
        {"item": k, "esta execução": env[k], "notebook (02_modelagem)": ref.get(k, "—"),
         "igual": (env[k] == ref[k]) if k in ref else "—"}
        for k in ["python", "numpy", "scipy", "pandas", "scikit-learn", "sistema", "maquina", "processador"]
    ]))
    R("")
    R(f"BLAS/OpenMP: `{env['blas_openmp']}`")

    X, y = load_train()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # aviso de uso do teste: aqui só se reproduz a medição final
        X_test, y_test = load_test()
    problems: list[str] = []

    # --- 2. Hiperparâmetros congelados ----------------------------------------
    R.h("2. Hiperparâmetros: freeze() × 02_hiperparametros_congelados.csv")
    t0 = time.perf_counter()
    frozen, searches = M.freeze(X, y)
    R(f"freeze() em {time.perf_counter() - t0:.1f} s.")
    R("")
    csv = pd.read_csv(TABLES / "02_hiperparametros_congelados.csv", dtype=str, keep_default_na=False)
    rows = []
    for name, s in searches.items():
        got = {k.removeprefix("model__"): fmt_param(v) for k, v in s.best_params_.items()}
        exp = dict(zip(csv.loc[csv.alternativa == name, "hiperparâmetro"], csv.loc[csv.alternativa == name, "valor"]))
        for k in sorted(set(got) | set(exp)):
            ok = got.get(k) == exp.get(k)
            blocking = name in BLOCKING_PARAMS
            rows.append({"alternativa": name, "hiperparâmetro": k, "CSV": exp.get(k, "—"),
                         "reproduzido": got.get(k, "—"), "confere": ok,
                         "bloqueante": "sim" if blocking else "não (modelo não usado aqui)"})
            if not ok and blocking:
                problems.append(f"hiperparâmetro {name}.{k}: CSV {exp.get(k)} × reproduzido {got.get(k)}")
    params = pd.DataFrame(rows)
    params.to_csv(OUT / "hiperparametros_reproduzidos.csv", index=False)
    R.table(params)

    # --- 3. Reajuste e predição no teste --------------------------------------
    R.h("3. Teste: matrizes de confusão (VN, FP, FN, VP)")
    test_csv = pd.read_csv(TABLES / "02_erros_por_classe_teste.csv", index_col=0)
    test_scores_csv = pd.read_csv(TABLES / "02_resultados_teste.csv", index_col="modelo")
    test_pred, test_proba, rows = {}, {}, []
    for n in MCNEMAR_MODELS:
        model = clone(frozen[n]).fit(X, y)
        test_pred[n] = model.predict(X_test)
        test_proba[n] = model.predict_proba(X_test)[:, 1]
        got = cm(y_test, test_pred[n])
        exp_csv = tuple(int(v) for v in test_csv.loc[n, ["VN", "FP", "FN", "VP"]])
        sc = E.scores(y_test, test_pred[n], test_proba[n])
        sc_ok = all(abs(sc[m] - test_scores_csv.loc[n, m]) < TOL for m in METRICS)
        ok = got == REPORT_TEST_CM[n] == exp_csv and sc_ok
        rows.append({"modelo": n, "relatório": REPORT_TEST_CM[n], "02_erros_por_classe_teste.csv": exp_csv,
                     "reproduzido": got, "métricas = 02_resultados_teste.csv": sc_ok, "confere": ok})
        if not ok:
            problems.append(f"teste {n}: relatório {REPORT_TEST_CM[n]}, CSV {exp_csv}, reproduzido {got}, "
                            f"métricas conferem: {sc_ok}")
            near = np.argsort(np.abs(test_proba[n] - 0.5))[:5]
            R(f"- {n}: probabilidades mais próximas de 0,5: "
              + ", ".join(f"row_id {X_test.index[i]} = {test_proba[n][i]:.6f} (y={y_test.iloc[i]})" for i in near))
    R.table(pd.DataFrame(rows))
    preds = pd.DataFrame({"y": y_test})
    for n in MCNEMAR_MODELS:
        preds[f"pred | {n}"] = test_pred[n]
        preds[f"proba | {n}"] = test_proba[n]
    preds.to_csv(OUT / "predicoes_teste.csv")

    if args.so_verificar_teste:
        R.h("Execução parcial (--so-verificar-teste)")
        R("VC aninhada e testes não executados." if not problems else "Há divergências; ver acima.")
        R.save("resultados_parcial.md")
        return 1 if problems else 0

    # --- 4. VC aninhada -------------------------------------------------------
    R.h("4. VC aninhada reproduzida: comparação com as tabelas exportadas")
    cv_csv = pd.read_csv(TABLES / "02_resultados_cv.csv", index_col="modelo")
    cls_csv = pd.read_csv(TABLES / "02_erros_por_classe_cv.csv", index_col=0)
    pairs_csv = pd.read_csv(TABLES / "02_comparacoes_pareadas.csv")
    res = {}
    for n in CV_MODELS:
        t0 = time.perf_counter()
        res[n] = E.evaluate(n, M.nested(n), X, y, get_cv())
        R(f"- {n}: VC aninhada em {time.perf_counter() - t0:.1f} s")
    R("")

    rows = []
    for n in CV_MODELS:
        for m in C.SCORING:
            for stat, val in [("media", res[n].folds[m].mean()), ("dp", res[n].folds[m].std(ddof=1))]:
                exp = cv_csv.loc[n, f"{m}_{stat}"]
                ok = abs(val - exp) < TOL
                rows.append({"modelo": n, "métrica": m, "estatística": stat, "CSV": exp, "reproduzido": val, "confere": ok})
                if not ok:
                    problems.append(f"VC {n} {m} {stat}: CSV {exp} × reproduzido {val}")
    R("**02_resultados_cv.csv**")
    R("")
    R.table(pd.DataFrame(rows))

    rows = []
    for n in CV_MODELS:
        got = cm(y, res[n].pred)
        exp = tuple(int(v) for v in cls_csv.loc[n, ["VN", "FP", "FN", "VP"]])
        rows.append({"modelo": n, "CSV": exp, "reproduzido": got, "confere": got == exp})
        if got != exp:
            problems.append(f"VC fora da dobra {n}: CSV {exp} × reproduzido {got}")
    R("")
    R("**02_erros_por_classe_cv.csv** (predições fora da dobra, N = 612)")
    R("")
    R.table(pd.DataFrame(rows))

    # Diferenças por dobra (acurácia balanceada): valida os valores dobra a dobra, não só média/DP.
    rows = []
    for r in pairs_csv.itertuples(index=False):
        if r.A in res and r.B in res:
            got = E.paired_comparison(res[r.A], res[r.B])
            ok = (abs(got["dif_media"] - r.dif_media) < TOL and abs(got["dif_dp"] - r.dif_dp) < TOL
                  and got["dobras_A_melhor"] == r.dobras_A_melhor and got["dobras_empate"] == r.dobras_empate)
            rows.append({"A": r.A, "B": r.B, "dif_media CSV": r.dif_media, "dif_media reprod.": got["dif_media"],
                         "dif_dp CSV": r.dif_dp, "dif_dp reprod.": got["dif_dp"], "confere": ok})
            if not ok:
                problems.append(f"comparação pareada {r.A} × {r.B} difere de 02_comparacoes_pareadas.csv")
    R("")
    R("**02_comparacoes_pareadas.csv** (acurácia balanceada, dobra a dobra)")
    R("")
    R.table(pd.DataFrame(rows))

    folds = pd.concat(
        [res[n].folds[METRICS].assign(modelo=n, dobra=np.arange(1, len(res[n].folds) + 1)) for n in CV_MODELS]
    )[["modelo", "dobra", *METRICS]]
    folds.to_csv(OUT / "vc_por_dobra.csv", index=False)
    oof = pd.DataFrame({"y": y})
    for n in CV_MODELS:
        oof[f"pred | {n}"] = res[n].pred
        oof[f"proba | {n}"] = res[n].proba
    fold_of = np.empty(len(y), dtype=int)
    for k, idx in enumerate(res[CV_MODELS[0]].test_idx, start=1):
        fold_of[idx] = k
    oof.insert(1, "dobra", fold_of)
    oof.to_csv(OUT / "vc_predicoes_fora_da_dobra.csv")

    # --- Parada ---------------------------------------------------------------
    R.h("5. Resultado da verificação")
    if problems:
        R("**PARADO: há divergências. Os testes NÃO foram calculados.**")
        R("")
        for p in problems:
            R(f"- {p}")
        R.save()
        return 1
    R("Tudo confere com o relatório e com as tabelas exportadas. Seguindo para os testes.")

    # --- 6. Ranking da VC (itens 2 e 3) --------------------------------------
    R.h("6. VC: ranking das 12 alternativas (tabelas exportadas)")
    alts = M.MEMBERS + [M.STUMP] + M.COMMITTEES
    rk = pd.DataFrame({
        "modelo": alts,
        "acurácia balanceada": [f"{cv_csv.loc[n, 'balanced_accuracy_media']:.4f} ± {cv_csv.loc[n, 'balanced_accuracy_dp']:.4f}" for n in alts],
        "sensibilidade": [f"{cv_csv.loc[n, 'recall_doenca_media']:.4f} ± {cv_csv.loc[n, 'recall_doenca_dp']:.4f}" for n in alts],
        "ROC-AUC": [("—" if np.isnan(cv_csv.loc[n, "roc_auc_media"]) else
                     f"{cv_csv.loc[n, 'roc_auc_media']:.4f} ± {cv_csv.loc[n, 'roc_auc_dp']:.4f}") for n in alts],
        "FN fora da dobra": [int(cls_csv.loc[n, "FN"]) for n in alts],
        "_auc": [cv_csv.loc[n, "roc_auc_media"] for n in alts],
        "_rec": [cv_csv.loc[n, "recall_doenca_media"] for n in alts],
    })
    R("Ordenado por ROC-AUC:")
    R("")
    R.table(rk.sort_values("_auc", ascending=False, na_position="last").drop(columns=["_auc", "_rec"]))
    R("")
    R("Ordenado por sensibilidade:")
    R("")
    R.table(rk.sort_values("_rec", ascending=False).drop(columns=["_auc", "_rec"]))

    by_auc = cv_csv.loc[alts, "roc_auc_media"].dropna().sort_values(ascending=False)
    by_rec = cv_csv.loc[alts, "recall_doenca_media"].sort_values(ascending=False)
    second_auc = next(n for n in by_auc.index if n != M.RF)
    top_rec = by_rec.index[0] if by_rec.index[0] != M.RF else by_rec.index[1]
    R("")
    R(f"- (a) Maior ROC-AUC: **{by_auc.index[0]}** ({by_auc.iloc[0]:.4f}). "
      f"Segundo depois do RF: {second_auc} ({by_auc[second_auc]:.4f}); "
      f"diferença RF − {second_auc} = {by_auc[M.RF] - by_auc[second_auc]:+.4f}.")
    R(f"- (b) Maior sensibilidade: **{by_rec.index[0]}** ({by_rec.iloc[0]:.4f}). "
      f"Diferença {top_rec} − RF = {100 * (by_rec[top_rec] - by_rec[M.RF]):+.2f} p.p.; "
      f"FN fora da dobra: {top_rec} {int(cls_csv.loc[top_rec, 'FN'])} × RF {int(cls_csv.loc[M.RF, 'FN'])}.")

    # --- 7. Item 4: Nadeau-Bengio ---------------------------------------------
    R.h("7. Item 4: RF dobra a dobra (t pareado corrigido de Nadeau e Bengio, 9 g.l.)")
    rows = []
    for rival, metric in [(second_auc, "roc_auc"), (top_rec, "recall_doenca"),
                          (second_auc, "balanced_accuracy"), (top_rec, "balanced_accuracy")]:
        if rival not in res:
            problems.append(f"{rival} não está entre os modelos com VC reproduzida")
            continue
        r = E.paired_comparison(res[M.RF], res[rival], metric)
        rows.append({"A": M.RF, "B": rival, "métrica": LABEL[metric], "dif_media (A−B)": r["dif_media"],
                     "dif_dp": r["dif_dp"], "dobras A melhor": r["dobras_A_melhor"], "empates": r["dobras_empate"],
                     "t corrigido": r["t_corrigido"], "p (9 g.l.)": r["p_valor"]})
    nb = pd.DataFrame(rows)
    nb.to_csv(OUT / "nadeau_bengio.csv", index=False)
    R.table(nb)
    R("")
    R("Valores por dobra:")
    R("")
    sub = folds[folds.modelo.isin([M.RF, second_auc, top_rec])]
    wide = sub.pivot(index="dobra", columns="modelo", values=["roc_auc", "recall_doenca"]).round(4)
    wide.columns = [f"{LABEL[m]} | {n}" for m, n in wide.columns]
    R.table(wide, index=True)

    # --- 8. Item 6: McNemar no teste ------------------------------------------
    R.h("8. Item 6: McNemar exato no teste, RF × rivais (A = Random Forest)")
    sick = (y_test == 1).to_numpy()
    rows = []
    for version, mask, label in [("a", np.ones(len(y_test), bool), "acerto geral (N = 306)"),
                                 ("b", sick, f"sensibilidade (doentes, N = {sick.sum()})")]:
        block = []
        ok_rf = test_pred[M.RF][mask] == y_test.to_numpy()[mask]
        for rival in RF_RIVALS_TEST:
            ok_r = test_pred[rival][mask] == y_test.to_numpy()[mask]
            block.append({"versão": version, "descrição": label, "B": rival, **mcnemar_exact(ok_rf, ok_r)})
        adj = holm([b["p_exato"] for b in block])
        for b, a in zip(block, adj):
            b["p_holm (4 comparações)"] = a
        rows += block
    mc = pd.DataFrame(rows)
    mc.to_csv(OUT / "mcnemar_teste.csv", index=False)
    for version in ["a", "b"]:
        R(f"**Versão ({version})**: {mc.loc[mc['versão'] == version, 'descrição'].iloc[0]}")
        R("")
        R.table(mc[mc["versão"] == version].drop(columns=["versão", "descrição"]))
        R("")

    # --- 9. Item 7: McNemar descritivo nas predições fora da dobra -----------
    R.h("9. Item 7: McNemar nas predições fora da dobra (doentes do treino), DESCRITIVO")
    R("As predições vêm de um modelo diferente em cada dobra (hiperparâmetros reescolhidos na busca "
      "interna), então os pares não são independentes no sentido exigido pelo teste. O resultado é só descritivo.")
    R("")
    sick_tr = (y == 1).to_numpy()
    ok_rf = res[M.RF].pred[sick_tr] == 1
    rows = [{"B": rival, **mcnemar_exact(ok_rf, res[rival].pred[sick_tr] == 1)} for rival in RF_RIVALS_OOF]
    mo = pd.DataFrame(rows)
    mo.to_csv(OUT / "mcnemar_vc_fora_da_dobra.csv", index=False)
    R.table(mo)

    R.save()
    print(f"\nArquivos gravados em {OUT.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
