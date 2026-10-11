"""Сверка окружения с тем, в котором прогоняются ноутбуки семинара.

    python check_env.py

Печатает версию Python и версии ключевых пакетов рядом с версиями из requirements.lock
(это pip freeze образа colloq-kernel:kaggle-base, Python 3.11.16). Расхождение — не
ошибка само по себе, но если у вас что-то работает, а в прогоне упало, смотреть сюда.
"""

from __future__ import annotations

import importlib.metadata as metadata
import platform
import sys
from pathlib import Path

KEY = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "catboost", "optuna", "polars"]
EXPECTED_PYTHON = "3.11.16"


def locked() -> dict[str, str]:
    out: dict[str, str] = {}
    lock = Path(__file__).with_name("requirements.lock")
    if not lock.exists():
        return out
    for line in lock.read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.strip().split("==", 1)
            out[name.lower()] = version
    return out


def main() -> None:
    want = locked()
    here = platform.python_version()
    mark = "ok" if here == EXPECTED_PYTHON else f"в прогоне {EXPECTED_PYTHON}"
    print(f"python {here:12s} {mark}")
    for name in KEY:
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = "—"
        expected = want.get(name, "?")
        mark = "ok" if have == expected else f"в прогоне {expected}"
        print(f"{name:14s} {have:12s} {mark}")
    print("ядро прогона: без сети, 2 CPU, 4 ГБ, лимит 3 минуты; данные в data/ рядом с ноутбуком")
    sys.exit(0)


if __name__ == "__main__":
    main()
