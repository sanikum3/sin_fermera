"""Качество квантилей P — та часть Operational Score, которую можно посчитать на train.

Score = 100·(0.4·P + 0.6·B). B зависит от скрытых затрат и сценариев, P — только от спроса
и города: pinball loss трёх квантилей относительно эмпирических квантилей спроса по городу
(формула — в README задачи). В cv.json участник заявляет 100·P на своих OOF-предсказаниях.

В ноутбуке:  import sys; sys.path.insert(0, "data"); from metric_cv import quantile_quality
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)


def pinball(y_true, prediction, quantile: float) -> float:
    error = np.asarray(y_true, dtype=float) - np.asarray(prediction, dtype=float)
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def quantile_quality(y_true, p10, p50, p90, city) -> float:
    """P в [0, 1]: 1 − L / L0, где L0 — та же ошибка у эмпирических квантилей спроса по городу."""
    y = np.asarray(y_true, dtype=float)
    city = pd.Series(np.asarray(city)).reset_index(drop=True)
    losses, references = [], []
    for quantile, prediction in zip(QUANTILES, (p10, p50, p90)):
        losses.append(pinball(y, prediction, quantile))
        by_city = pd.Series(y).groupby(city).quantile(quantile)
        references.append(pinball(y, city.map(by_city).to_numpy(dtype=float), quantile))
    return float(np.clip(1.0 - np.mean(losses) / (np.mean(references) + 1e-12), 0.0, 1.0))


def cv_json(name: str, fold_qualities, path: str = "cv.json") -> dict:
    """Записать и вернуть cv.json: folds и cv — в баллах 100·P."""
    folds = [round(100.0 * float(q), 2) for q in fold_qualities]
    cv = {"name": name, "metric": "quantile_quality_x100", "folds": folds,
          "cv": round(float(np.mean(folds)), 2), "std": round(float(np.std(folds)), 2)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cv, f, ensure_ascii=False)
    return cv
