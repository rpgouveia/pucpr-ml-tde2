from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .preprocessing import build_preprocessor

LOGREG = "Regressão logística"
KNN = "KNN"
SVM = "SVM (RBF)"
TREE = "Árvore de decisão"
NB = "Naive Bayes"
STUMP = "Toco de decisão"
SOFT_VOTING = "Votação suave"
HARD_VOTING = "Votação dura"
RF = "Random Forest"
BAGGING = "Bagging de árvores"
GB = "Gradient Boosting"
ADABOOST = "AdaBoost"

MEMBERS = [LOGREG, KNN, SVM, TREE, NB]
BASE_MEMBERS = [TREE, STUMP]
HETEROGENEOUS = [SOFT_VOTING, HARD_VOTING]
VARIANCE = [RF, BAGGING]
BIAS = [GB, ADABOOST]
COMMITTEES = HETEROGENEOUS + VARIANCE + BIAS
TUNABLE = MEMBERS + [STUMP] + VARIANCE + BIAS

FAMILY = {
    **{m: "membro" for m in MEMBERS + [STUMP]},
    **{m: "heterogêneo" for m in HETEROGENEOUS},
    **{m: "homogêneo (variância)" for m in VARIANCE},
    **{m: "homogêneo (viés)" for m in BIAS},
}

REFERENCE_MEMBERS = {
    SOFT_VOTING: MEMBERS,
    HARD_VOTING: MEMBERS,
    RF: [TREE],
    BAGGING: [TREE],
    GB: [TREE],
    ADABOOST: [STUMP],
}

TREE_DEPTHS = [None, 2, 3, 4, 5, 6, 8, 10, 12]


def estimators() -> dict:
    return {
        LOGREG: LogisticRegression(max_iter=5000),
        KNN: KNeighborsClassifier(),
        SVM: CalibratedClassifierCV(SVC(kernel="rbf", random_state=C.SEED), ensemble=False),
        TREE: DecisionTreeClassifier(random_state=C.SEED),
        NB: GaussianNB(),
        STUMP: DecisionTreeClassifier(max_depth=1, random_state=C.SEED),
        RF: RandomForestClassifier(random_state=C.SEED),
        BAGGING: BaggingClassifier(DecisionTreeClassifier(random_state=C.SEED), random_state=C.SEED),
        GB: GradientBoostingClassifier(random_state=C.SEED),
        ADABOOST: AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=C.SEED), random_state=C.SEED),
    }


def search_spaces() -> dict:
    return {
        LOGREG: {
            "C": loguniform(1e-3, 1e2),
            "class_weight": [None, "balanced"],
        },
        KNN: {
            "n_neighbors": randint(1, 51),
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
        SVM: {
            "estimator__C": loguniform(1e-2, 1e2),
            "estimator__gamma": loguniform(1e-4, 1e0),
            "estimator__class_weight": [None, "balanced"],
        },
        TREE: {
            "max_depth": TREE_DEPTHS,
            "min_samples_leaf": randint(1, 31),
            "criterion": ["gini", "entropy"],
        },
        NB: {
            "var_smoothing": loguniform(1e-12, 1e-1),
        },
        STUMP: {
            "min_samples_leaf": randint(1, 51),
            "criterion": ["gini", "entropy"],
            "class_weight": [None, "balanced"],
        },
        RF: {
            "n_estimators": randint(50, 501),
            "max_depth": TREE_DEPTHS,
            "min_samples_leaf": randint(1, 21),
            "max_features": ["sqrt", "log2", 0.5, None],
        },
        BAGGING: {
            "n_estimators": randint(10, 301),
            "max_samples": uniform(0.3, 0.7),
            "max_features": uniform(0.3, 0.7),
            "estimator__max_depth": TREE_DEPTHS,
        },
        GB: {
            "n_estimators": randint(50, 501),
            "learning_rate": loguniform(1e-2, 5e-1),
            "max_depth": randint(1, 6),
            "subsample": uniform(0.5, 0.5),
        },
        ADABOOST: {
            "n_estimators": randint(50, 501),
            "learning_rate": loguniform(1e-2, 2e0),
        },
    }


def pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def inner_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=C.INNER_SPLITS, shuffle=True, random_state=C.SEED)


def tuned(name: str) -> RandomizedSearchCV:
    space = {f"model__{k}": v for k, v in search_spaces()[name].items()}
    return RandomizedSearchCV(
        pipeline(estimators()[name]),
        space,
        n_iter=C.N_ITER,
        cv=inner_cv(),
        scoring=C.PRIMARY_METRIC,
        random_state=C.SEED,
        n_jobs=-1,
    )


def voting(members: dict, kind: str) -> VotingClassifier:
    return VotingClassifier([(name, clone(est)) for name, est in members.items()], voting=kind)


def nested_voting(kind: str) -> VotingClassifier:
    return voting({name: tuned(name) for name in MEMBERS}, kind)


def nested(name: str):
    if name == SOFT_VOTING:
        return nested_voting("soft")
    if name == HARD_VOTING:
        return nested_voting("hard")
    return tuned(name)


def freeze(X, y) -> tuple[dict, dict]:
    searches = {name: tuned(name).fit(X, y) for name in TUNABLE}
    frozen = {name: clone(s.best_estimator_) for name, s in searches.items()}
    member_models = {name: frozen[name] for name in MEMBERS}
    frozen[SOFT_VOTING] = voting(member_models, "soft")
    frozen[HARD_VOTING] = voting(member_models, "hard")
    return frozen, searches
