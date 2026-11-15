"""Métricas de evaluación compartidas por los cuadernos de modelado."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)


def metricas_regresion(y, pred) -> dict[str, float]:
    return {"R2": float(r2_score(y, pred)), "RMSE": float(np.sqrt(mean_squared_error(y, pred))), "MAE": float(mean_absolute_error(y, pred))}


def metricas_clasificacion(y, prob, umbral: float = 0.5) -> dict[str, float]:
    pred = (np.asarray(prob) >= umbral).astype(int)
    return {"Accuracy": float(accuracy_score(y, pred)), "Precision": float(precision_score(y, pred, zero_division=0)),
            "Recall": float(recall_score(y, pred)), "F1": float(f1_score(y, pred)), "AUC_ROC": float(roc_auc_score(y, prob)),
            "AUC_PR": float(average_precision_score(y, prob))}
