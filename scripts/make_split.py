"""Gera (ou confere) a divisão treino/teste em dataset/splits/.

    uv run python scripts/make_split.py           # gera os arquivos
    uv run python scripts/make_split.py --check   # só verifica integridade
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tde2 import config as C  # noqa: E402
from tde2.data import make_split, verify_split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="apenas verifica os arquivos existentes")
    args = parser.parse_args()

    if not args.check:
        report = make_split()
        print(json.dumps({k: report[k] for k in ("seed", "n_train", "n_test", "class_distribution")}, indent=2))
    verify_split()
    print(f"OK: divisão íntegra em {C.SPLITS_DIR.relative_to(C.ROOT)}/")


if __name__ == "__main__":
    main()
