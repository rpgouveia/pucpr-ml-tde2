"""Constantes do experimento. Altere aqui, nunca espalhado pelos notebooks."""

from pathlib import Path

# --- Caminhos -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
RAW_CSV = DATASET_DIR / "heart.csv"
SPLITS_DIR = DATASET_DIR / "splits"
TRAIN_CSV = SPLITS_DIR / "train.csv"
TEST_CSV = SPLITS_DIR / "test.csv"
SPLIT_REPORT = SPLITS_DIR / "split_report.json"
FIGURES_DIR = ROOT / "reports" / "figures"

# --- Reprodutibilidade --------------------------------------------------------
# Semente única para tudo: divisão treino/teste, dobras da validação cruzada e
# random_state dos modelos. Documentar no relatório (seção 4.2).
SEED = 42
TEST_SIZE = 1 / 3  # 1/3 separado no início e usado UMA única vez, ao final (612 treino / 306 teste)
N_SPLITS = 10     # validação cruzada estratificada sobre o treino
INNER_SPLITS = 5
N_ITER = 20
N_BOOTSTRAP = 1000
ALPHA = 0.05

# --- Esquema da base ----------------------------------------------------------
TARGET = "HeartDisease"
CLASS_NAMES = {0: "Normal", 1: "Doença cardíaca"}

NUMERIC = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
BINARY = ["FastingBS"]  # já vem codificado como 0/1
CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
FEATURES = NUMERIC + BINARY + CATEGORICAL

# Valores 0 fisiologicamente impossíveis -> tratados como ausentes no pipeline.
# Cholesterol = 0 em 172 linhas (vêm da base Switzerland); RestingBP = 0 em 1 linha.
ZERO_AS_MISSING = ["RestingBP", "Cholesterol"]
OTHER_NUMERIC = [c for c in NUMERIC if c not in ZERO_AS_MISSING]

# --- Métricas (sugestão; a decisão final é da equipe) -------------------------
# A base é levemente desbalanceada (55% positivos). balanced_accuracy funciona
# com predições "duras", então serve também para votação majoritária (hard).
PRIMARY_METRIC = "balanced_accuracy"
SCORING = {
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "recall_doenca": "recall",  # recall da classe positiva (falso negativo é o erro caro)
    "roc_auc": "roc_auc",
}
