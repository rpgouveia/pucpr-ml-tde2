"""Origem dos zeros de Cholesterol/RestingBP do Heart Failure Prediction Dataset.

Cruza dataset/heart.csv (Kaggle, fedesoriano) com os quatro arquivos
processados da UCI Heart Disease (id 45) copiados em dataset/uci/ e confere:

1. totais e ausências (chol, trestbps) por centro;
2. de qual centro vem cada linha do Kaggle com Cholesterol = 0 ou RestingBP = 0;
3. duplicatas entre os quatro centros (relação com as 272 duplicatas e a Statlog);
4. em que partição está a linha com RestingBP = 0 e quantas colunas o
   pré-processamento produz em cada dobra externa.

Somente leitura: não altera nenhum arquivo do projeto. Imprime tabelas em
Markdown no stdout.

    uv run python scripts/analise_zeros_por_centro.py
"""

from __future__ import annotations

import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tde2 import config as C  # noqa: E402
from tde2.data import get_cv, load_train  # noqa: E402
from tde2.preprocessing import build_preprocessor  # noqa: E402

UCI_DIR = ROOT / "dataset" / "uci"
UCI_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
CENTERS = {
    "Cleveland": "processed.cleveland.data",
    "Hungarian": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "Long Beach VA": "processed.va.data",
}

# Conversões UCI -> Kaggle
SEX = {1: "M", 0: "F"}
CP = {1: "TA", 2: "ATA", 3: "NAP", 4: "ASY"}
RESTECG = {0: "Normal", 1: "ST", 2: "LVH"}
EXANG = {1: "Y", 0: "N"}
SLOPE = {1: "Up", 2: "Flat", 3: "Down"}

# Atributos comparados (11 + alvo). Cholesterol tem regra própria (ausente <-> 0);
# nas demais colunas, "?" na UCI é tratado como curinga, porque o CSV do Kaggle
# não tem nenhum ausente (o autor preencheu esses valores de algum modo).
KAGGLE_FEATURES = C.FEATURES  # Age, RestingBP, Cholesterol, MaxHR, Oldpeak, FastingBS, Sex, ...
WILDCARD_COLS = [c for c in KAGGLE_FEATURES if c != "Cholesterol"]


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------
def _read_bytes(fname: str) -> bytes:
    """Lê o arquivo solto em dataset/uci ou, se não existir, de dentro de um .zip."""
    path = UCI_DIR / fname
    if path.exists():
        return path.read_bytes()
    for z in sorted(UCI_DIR.rglob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if Path(name).name == fname:
                    return zf.read(name)
    raise FileNotFoundError(f"{fname} não encontrado em {UCI_DIR} (nem em .zip)")


def load_uci() -> pd.DataFrame:
    frames = []
    for center, fname in CENTERS.items():
        raw = pd.read_csv(BytesIO(_read_bytes(fname)), header=None, names=UCI_COLS,
                          na_values="?", skipinitialspace=True)
        raw.insert(0, "center", center)
        raw.insert(1, "uci_line", np.arange(1, len(raw) + 1))
        frames.append(raw)
    return pd.concat(frames, ignore_index=True)


def uci_to_kaggle(u: pd.DataFrame) -> pd.DataFrame:
    """Converte para o esquema do Kaggle. NaN = ausente na UCI (curinga)."""
    k = pd.DataFrame(index=u.index)
    k["Age"] = u["age"]
    k["Sex"] = u["sex"].map(SEX)
    k["ChestPainType"] = u["cp"].map(CP)
    k["RestingBP"] = u["trestbps"]
    k["Cholesterol"] = u["chol"].fillna(0)  # "?" ou 0 -> 0
    k["FastingBS"] = u["fbs"]
    k["RestingECG"] = u["restecg"].map(RESTECG)
    k["MaxHR"] = u["thalach"]
    k["ExerciseAngina"] = u["exang"].map(EXANG)
    k["Oldpeak"] = u["oldpeak"].round(1)
    k["ST_Slope"] = u["slope"].map(SLOPE)
    k["HeartDisease"] = (u["num"] > 0).astype(int)
    return k


# ---------------------------------------------------------------------------
# Casamento Kaggle -> UCI
# ---------------------------------------------------------------------------
def candidates(row: pd.Series, uk: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Índices UCI compatíveis com a linha do Kaggle e nº de curingas usados em cada um."""
    ok = (uk["HeartDisease"].to_numpy() == row["HeartDisease"]) & (
        np.isclose(uk["Cholesterol"].to_numpy(dtype=float), float(row["Cholesterol"]))
    )
    wild = np.zeros(len(uk), dtype=int)
    for col in WILDCARD_COLS:
        vals = uk[col]
        na = vals.isna().to_numpy()
        if pd.api.types.is_numeric_dtype(vals):
            eq = np.isclose(vals.to_numpy(dtype=float, na_value=np.nan), float(row[col]))
        else:
            eq = (vals == row[col]).fillna(False).to_numpy(dtype=bool)
        ok &= eq | na
        wild += na
    idx = np.flatnonzero(ok)
    return idx, wild[idx]


def classify(kag: pd.DataFrame, uk: pd.DataFrame, centers: pd.Series) -> pd.DataFrame:
    rows = []
    for rid, row in kag.iterrows():
        idx, wild = candidates(row, uk)
        cs_all = sorted(set(centers.iloc[idx]))
        exact = idx[wild == 0]
        cs_exact = sorted(set(centers.iloc[exact]))

        def status(cs):
            return "não encontrada" if not cs else ("única" if len(cs) == 1 else "ambígua")

        # Regra principal: considera todos os candidatos (com ou sem curinga).
        # Regra alternativa: se houver candidato exato (sem curinga), fica só com ele.
        cs_pref = cs_exact if cs_exact else cs_all
        rows.append({
            "row_id": rid,
            "n_cand": len(idx),
            "n_exatos": len(exact),
            "centros": ", ".join(cs_all),
            "status": status(cs_all),
            "centros_pref_exato": ", ".join(cs_pref),
            "status_pref_exato": status(cs_pref),
            "uci_idx": list(idx),
        })
    return pd.DataFrame(rows).set_index("row_id")


# ---------------------------------------------------------------------------
# Saída
# ---------------------------------------------------------------------------
def md(df: pd.DataFrame, index: bool = True) -> str:
    """Tabela Markdown sem depender de `tabulate` (não está nas dependências)."""
    d = df.reset_index() if index else df

    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else (f"{v:g}")
        return str(v)

    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def section(title: str) -> None:
    print(f"\n## {title}\n")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    uci = load_uci()
    uk = uci_to_kaggle(uci)
    kag = pd.read_csv(C.RAW_CSV)
    kag.index.name = "row_id"

    # --- Item 1/2: totais e ausências por centro ---------------------------
    section("Item 2 — Totais e ausências por centro (UCI processada)")
    g = uci.groupby("center", sort=False)
    t = pd.DataFrame({
        "registros": g.size(),
        "chol '?'": g["chol"].apply(lambda s: s.isna().sum()),
        "chol = 0": g["chol"].apply(lambda s: (s == 0).sum()),
        "trestbps '?'": g["trestbps"].apply(lambda s: s.isna().sum()),
        "trestbps = 0": g["trestbps"].apply(lambda s: (s == 0).sum()),
    })
    t.insert(3, "chol ausente", t["chol '?'"] + t["chol = 0"])
    t["trestbps ausente"] = t["trestbps '?'"] + t["trestbps = 0"]
    t.loc["Total"] = t.sum()
    t["% chol ausente do total"] = (100 * t["chol ausente"] / t.loc["Total", "chol ausente"]).round(1)
    print(md(t))

    section("Ausências ('?') em outras colunas, por centro")
    na = uci.groupby("center", sort=False)[UCI_COLS].apply(lambda d: d.isna().sum())
    na = na.loc[:, na.sum() > 0]
    print(md(na))

    # --- Item 3: casamento das linhas zeradas -----------------------------
    zero_chol = kag[kag["Cholesterol"] == 0]
    zero_bp = kag[kag["RestingBP"] == 0]
    section("Item 3 — Linhas do Kaggle com zeros")
    print(f"- Cholesterol = 0: {len(zero_chol)} linhas")
    print(f"- RestingBP = 0: {len(zero_bp)} linha(s): row_id {list(zero_bp.index)}")
    print(f"- Linhas com ambos os zeros: {int(((kag.Cholesterol == 0) & (kag.RestingBP == 0)).sum())}")
    print(f"- NaN no CSV do Kaggle: {int(kag.isna().sum().sum())}")

    res_all = classify(kag, uk, uci["center"])  # todas as 918, para contexto
    res_chol = res_all.loc[zero_chol.index].copy()
    res_chol["centro_exclusao"] = pd.Series(pd.NA, index=res_chol.index, dtype="object")
    res_bp = res_all.loc[zero_bp.index]

    for label, res in [("Cholesterol = 0", res_chol), ("RestingBP = 0", res_bp)]:
        section(f"Item 3 — Status do casamento: {label} (n = {len(res)})")
        s = pd.DataFrame({
            "regra principal (todos os candidatos)": res["status"].value_counts(),
            "regra alternativa (prefere exato)": res["status_pref_exato"].value_counts(),
        }).reindex(["única", "ambígua", "não encontrada"]).fillna(0).astype(int)
        s.loc["Total"] = s.sum()
        print(md(s))
        print()
        print(f"Linhas com pelo menos um candidato exato (sem curinga): {(res['n_exatos'] > 0).sum()}")

        section(f"Item 3 — Centro de origem: {label}")
        o = pd.DataFrame({
            "regra principal": res["centros"].replace("", "(não encontrada)").value_counts(),
            "regra alternativa": res["centros_pref_exato"].replace("", "(não encontrada)").value_counts(),
        }).fillna(0).astype(int)
        o.loc["Total"] = o.sum()
        print(md(o))

    # Detalhe da linha RestingBP = 0
    section("Item 3 — Detalhe da linha com RestingBP = 0")
    for rid, r in res_bp.iterrows():
        print(md(kag.loc[[rid]]))
        print()
        print(md(uci.iloc[r["uci_idx"]]))

    # Linhas ambíguas / não encontradas (para auditoria)
    odd = res_chol[res_chol["status"] != "única"]
    res_chol["centro_exclusao"] = res_chol["centro_exclusao"].fillna(res_chol["centros"])
    if len(odd):
        section("Item 3 — Linhas Cholesterol = 0 sem correspondência única (regra principal)")
        print(md(odd[["n_cand", "n_exatos", "centros", "centros_pref_exato"]]))

        # Resolução por exclusão (casamento 1-para-1): um candidato UCI que já é o
        # par de outra linha com correspondência única não pode ser reutilizado.
        claimed = {}
        for rid, r in res_chol[res_chol["status"] == "única"].iterrows():
            for i in r["uci_idx"]:
                claimed.setdefault(i, []).append(rid)
        section("Item 3 — Resolução das ambíguas por exclusão (casamento 1-para-1)")
        print(f"Registros UCI usados por mais de uma linha 'única': "
              f"{sum(len(v) > 1 for v in claimed.values())}")
        print()
        lines = []
        for rid, r in odd.iterrows():
            free = [i for i in r["uci_idx"] if i not in claimed]
            for i in r["uci_idx"]:
                lines.append({
                    "row_id Kaggle": rid,
                    "candidato UCI": f"{uci.at[i, 'center']} linha {uci.at[i, 'uci_line']}",
                    "já é par único de": ", ".join(map(str, claimed.get(i, []))) or "—",
                })
            res_chol.at[rid, "centro_exclusao"] = (
                ", ".join(sorted(set(uci.loc[free, "center"]))) if free else "(sem candidato livre)"
            )
        print(md(pd.DataFrame(lines), index=False))
        print()
        print(md(res_chol["centro_exclusao"].value_counts().rename("Cholesterol = 0 (após exclusão)").to_frame()))

    # Ausentes de colesterol da UCI que não aparecem no Kaggle com Cholesterol = 0
    used = {i for lst in res_chol["uci_idx"] for i in lst}
    chol_miss = uci.index[uci["chol"].isna() | (uci["chol"] == 0)]
    not_used = [i for i in chol_miss if i not in used]
    section("Item 3 — Registros UCI com colesterol ausente sem par entre os zeros do Kaggle")
    nu = uci.loc[not_used]
    print(md(nu.groupby("center", sort=False).agg(
        n=("chol", "size"), chol_interrogacao=("chol", lambda s: s.isna().sum())
    )))
    # Eles aparecem no Kaggle com colesterol preenchido? (casamento ignorando Cholesterol)
    nf_center: dict = {}  # row_id do Kaggle -> centro, para as linhas achadas só assim
    if len(nu):
        uk_nu = uk.loc[not_used]
        hit_rows, per_uci = set(), []
        for i, r in uk_nu.iterrows():
            m = pd.Series(True, index=kag.index)
            for col in WILDCARD_COLS + ["HeartDisease"]:
                if pd.isna(r[col]):
                    continue
                if isinstance(r[col], str):
                    m &= kag[col] == r[col]
                else:
                    m &= np.isclose(kag[col].astype(float), float(r[col]))
            ids = set(kag.index[m])
            per_uci.append(len(ids))
            hit_rows |= ids
            for rid in ids:
                nf_center[rid] = uci.at[i, "center"]
        print(f"\nCasando esses {len(nu)} registros com o Kaggle ignorando o colesterol: "
              f"{sum(n == 1 for n in per_uci)} têm exatamente 1 par, "
              f"{sum(n == 0 for n in per_uci)} nenhum, {sum(n > 1 for n in per_uci)} mais de um; "
              f"juntos correspondem a {len(hit_rows)} linhas distintas do Kaggle.")
        ch = kag.loc[sorted(hit_rows), "Cholesterol"]
        print(f"Cholesterol dessas linhas no Kaggle: min {ch.min()}, mediana {ch.median():g}, "
              f"max {ch.max()}, zeros {int((ch == 0).sum())}, valores distintos {ch.nunique()}.")
        not_found = set(res_all.index[res_all["status"] == "não encontrada"])
        print(f"Essas linhas são exatamente as 'não encontradas' do casamento geral: {hit_rows == not_found}")

    # Contexto: casamento de todas as 918 linhas
    section("Contexto — Casamento de todas as 918 linhas do Kaggle")
    s = res_all["status"].value_counts().reindex(["única", "ambígua", "não encontrada"]).fillna(0).astype(int)
    print(md(s.rename("linhas").to_frame()))
    print()
    print(md(res_all.loc[res_all.status == "única", "centros"].value_counts().rename("linhas (única)").to_frame()))
    matched_uci = {i for lst in res_all["uci_idx"] for i in lst}
    print(f"\nRegistros UCI que casam com pelo menos uma linha do Kaggle: {len(matched_uci)} de {len(uci)}")
    um = uci.loc[[i for i in uci.index if i not in matched_uci]]
    if len(um):
        print()
        print(md(um.groupby("center", sort=False).size().rename("UCI sem par no Kaggle").to_frame()))

    # Registros UCI com algum "?" nos 11 atributos: no Kaggle eles aparecem preenchidos.
    feat_uci = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
                "exang", "oldpeak", "slope"]
    anyq = uci[feat_uci].isna().any(axis=1)
    anyq_sem_chol = uci[[c for c in feat_uci if c != "chol"]].isna().any(axis=1)
    section("Contexto — Registros UCI com '?' em algum dos 11 atributos (preenchidos no Kaggle)")
    t = pd.DataFrame({
        "registros": uci.groupby("center", sort=False).size(),
        "com algum '?'": anyq.groupby(uci["center"], sort=False).sum(),
        "com '?' fora do colesterol": anyq_sem_chol.groupby(uci["center"], sort=False).sum(),
    })
    t.loc["Total"] = t.sum()
    t["% com algum '?'"] = (100 * t["com algum '?'"] / t["registros"]).round(1)
    print(md(t))
    dedup = uci.drop_duplicates(subset=UCI_COLS)
    n_q = int(dedup[feat_uci].isna().any(axis=1).sum())
    print(f"\nSem as cópias duplicadas ({len(dedup)} registros = linhas do Kaggle): {n_q} têm algum '?' "
          f"({100 * n_q / len(dedup):.1f}%), ou seja, chegam ao Kaggle com algum valor preenchido pelo autor.")

    # --- Complemento: centro de cada linha do Kaggle e prevalência ---------
    center_of = res_all["centros"].copy()
    center_of.loc[res_chol.index] = res_chol["centro_exclusao"]
    for rid, c in nf_center.items():
        center_of.at[rid] = c
    assert (center_of != "").all() and not center_of.str.contains(",").any(), "centro indefinido"
    train_ids = set(pd.read_csv(C.TRAIN_CSV, usecols=["row_id"])["row_id"])
    d = kag.assign(
        centro=center_of,
        particao=np.where(kag.index.isin(train_ids), "treino", "teste"),
        chol0=kag["Cholesterol"].eq(0),
    )
    order = list(CENTERS)

    section("Complemento — Linhas do Kaggle por centro e partição")
    t = pd.DataFrame({
        "linhas": d.groupby("centro").size(),
        "treino": d[d.particao == "treino"].groupby("centro").size(),
        "teste": d[d.particao == "teste"].groupby("centro").size(),
        "Chol=0": d[d.chol0].groupby("centro").size(),
        "Chol=0 treino": d[d.chol0 & (d.particao == "treino")].groupby("centro").size(),
        "Chol=0 teste": d[d.chol0 & (d.particao == "teste")].groupby("centro").size(),
    }).reindex(order).fillna(0).astype(int)
    t.loc["Total"] = t.sum()
    print(md(t))

    def prev(sub: pd.DataFrame) -> str:
        return f"{int(sub[C.TARGET].sum())}/{len(sub)} ({100 * sub[C.TARGET].mean():.1f}%)" if len(sub) else "—"

    for part in ["base completa", "treino"]:
        dd = d if part == "base completa" else d[d.particao == "treino"]
        section(f"Complemento — Prevalência da doença por centro e Cholesterol ({part})")
        rows = []
        for c in order + ["Total"]:
            s = dd if c == "Total" else dd[dd.centro == c]
            rows.append({
                "centro": c,
                "Cholesterol = 0": prev(s[s.chol0]),
                "Cholesterol > 0": prev(s[~s.chol0]),
                "todas": prev(s),
            })
        print(md(pd.DataFrame(rows), index=False))

    # --- Item 4: soma e duplicatas ----------------------------------------
    section("Item 4 — Soma dos centros e duplicatas")
    raw14 = uci[UCI_COLS]
    dup_raw = raw14.duplicated(keep=False)
    conv = uk[KAGGLE_FEATURES + [C.TARGET]]
    dup_conv = conv.duplicated(keep=False)
    print(f"- Soma dos quatro centros: {len(uci)} (esperado: 920)")
    print(f"- Linhas duplicadas (14 colunas brutas, '?' = '?'): {int(raw14.duplicated().sum())} "
          f"cópias extras; {int(dup_raw.sum())} linhas envolvidas")
    print(f"- Linhas duplicadas (11 atributos convertidos + alvo): {int(conv.duplicated().sum())} "
          f"cópias extras; {int(dup_conv.sum())} linhas envolvidas")
    if dup_raw.any():
        print()
        print(md(uci.loc[dup_raw, ["center", "uci_line"] + UCI_COLS]))
    print(f"- Duplicatas no próprio CSV do Kaggle: {int(kag.duplicated().sum())}")
    print(f"- 920 + 270 (Statlog) = 1190; 1190 − 918 = {1190 - len(kag)} removidas; "
          f"920 − 918 = {len(uci) - len(kag)}")

    # --- Item 5: partição e pipeline --------------------------------------
    section("Item 5 — Partição da linha RestingBP = 0 e colunas do pipeline")
    train = pd.read_csv(C.TRAIN_CSV, index_col="row_id")
    test = pd.read_csv(C.TEST_CSV, index_col="row_id")
    for rid in zero_bp.index:
        where = "test.csv" if rid in test.index else ("train.csv" if rid in train.index else "nenhuma")
        print(f"- row_id {rid}: {where}")
    print(f"- RestingBP = 0 no treino: {int((train.RestingBP == 0).sum())}; no teste: {int((test.RestingBP == 0).sum())}")
    print(f"- Cholesterol = 0 no treino: {int((train.Cholesterol == 0).sum())}; no teste: {int((test.Cholesterol == 0).sum())}")

    X, y = load_train()
    pos = {rid: X.index.get_loc(rid) for rid in zero_bp.index if rid in X.index}
    rows = []
    for k, (tr, va) in enumerate(get_cv().split(X, y), start=1):
        prep = build_preprocessor(scale=True).fit(X.iloc[tr], y.iloc[tr])
        names = list(prep.get_feature_names_out())
        rows.append({
            "dobra": k,
            "n_treino": len(tr),
            "RestingBP=0 no treino da dobra": int((X.iloc[tr].RestingBP == 0).sum()),
            "linha 449 no treino da dobra": any(p in tr for p in pos.values()) if pos else False,
            "colunas de saída": len(names),
            "missingindicator_RestingBP": "missingindicator_RestingBP" in names,
            "missingindicator_Cholesterol": "missingindicator_Cholesterol" in names,
        })
    full = build_preprocessor(scale=True).fit(X, y)
    rows.append({
        "dobra": "treino inteiro (ajuste final)",
        "n_treino": len(X),
        "RestingBP=0 no treino da dobra": int((X.RestingBP == 0).sum()),
        "linha 449 no treino da dobra": bool(pos),
        "colunas de saída": len(full.get_feature_names_out()),
        "missingindicator_RestingBP": "missingindicator_RestingBP" in full.get_feature_names_out(),
        "missingindicator_Cholesterol": "missingindicator_Cholesterol" in full.get_feature_names_out(),
    })
    print()
    print(md(pd.DataFrame(rows), index=False))
    print("\nColunas de saída (treino inteiro):")
    print(", ".join(full.get_feature_names_out()))

    # O que acontece com a linha RestingBP = 0 quando o teste é transformado.
    imp = full.named_transformers_["zero_missing"].named_steps["impute"]
    med = {c: float(v) for c, v in zip(C.ZERO_AS_MISSING, imp.statistics_)}
    Xt, _ = test[C.FEATURES], test[C.TARGET]
    out = full.transform(Xt)
    print(f"\nTransformando test.csv com o ajuste final: saída {out.shape}; "
          f"medianas do treino usadas na imputação: {med}")
    names = list(full.get_feature_names_out())
    for rid in zero_bp.index:
        if rid in Xt.index:
            r = out[Xt.index.get_loc(rid)]
            print(f"- row_id {rid}: RestingBP 0 é imputado pela mediana ({med['RestingBP']:g}) "
                  f"sem coluna indicadora; valor padronizado = {r[names.index('RestingBP')]:.4f}; "
                  f"missingindicator_Cholesterol (padronizado) = "
                  f"{r[names.index('missingindicator_Cholesterol')]:.4f}")

    # Contrafactual: o que aconteceria se a linha com RestingBP = 0 estivesse no ajuste.
    Xc = pd.concat([X, kag.loc[zero_bp.index, C.FEATURES]])
    yc = pd.concat([y, kag.loc[zero_bp.index, C.TARGET]])
    cf = build_preprocessor(scale=True).fit(Xc, yc).get_feature_names_out()
    print(f"\nContrafactual (NÃO é o procedimento do projeto): treino + linha(s) RestingBP = 0 "
          f"-> {len(cf)} colunas; missingindicator_RestingBP presente: {'missingindicator_RestingBP' in cf}")


if __name__ == "__main__":
    main()
