"""Ingeniería de variables y preprocesador para los modelos supervisados.

Dos etapas con responsabilidades separadas:
- `derivar`: transformaciones fila a fila que no aprenden nada de los datos (se pueden
  aplicar antes o después del split sin riesgo de fuga).
- `construir_preprocesador`: imputación, codificación y escalado, que SÍ aprenden
  parámetros y por eso se ajustan únicamente con el conjunto de entrenamiento.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from . import config as c

SIN_DATO = "Sin dato"

ORDINALES: dict[str, list[str]] = {
    "fami_estratovivienda": [SIN_DATO] + c.ORDEN_ESTRATO,
    "fami_educacionpadre": [SIN_DATO] + c.ORDEN_EDUCACION,
    "fami_educacionmadre": [SIN_DATO] + c.ORDEN_EDUCACION,
    "fami_numlibros": [SIN_DATO, "0 A 10 LIBROS", "11 A 25 LIBROS", "26 A 100 LIBROS", "MÁS DE 100 LIBROS"],
    "fami_personashogar": [SIN_DATO, "1 a 2", "3 a 4", "5 a 6", "7 a 8", "9 o más"],
    "estu_dedicacionlecturadiaria": [SIN_DATO, "No leo por entretenimiento", "30 minutos o menos", "Entre 30 y 60 minutos", "Entre 1 y 2 horas", "Más de 2 horas"],
    "estu_horassemanatrabaja": [SIN_DATO, "0", "Menos de 10 horas", "Entre 11 y 20 horas", "Entre 21 y 30 horas", "Más de 30 horas"],
}
NOMINALES: list[str] = ["estu_genero", "fami_tieneinternet", "fami_tienecomputador", "cole_naturaleza", "cole_area_ubicacion",
                        "cole_jornada", "cole_caracter", "cole_calendario", "cole_bilingue"]
ALTA_CARDINALIDAD: list[str] = ["cole_depto_ubicacion"]
NUMERICAS: list[str] = ["estu_inse_individual", "edad", "indice_tic", "educ_max_padres"]
BINARIAS: list[str] = ["extraedad", "inse_faltante"]

PREDICTORES: list[str] = list(ORDINALES) + NOMINALES + ALTA_CARDINALIDAD + NUMERICAS + BINARIAS

# Excluidas a propósito: los cinco puntajes por área (componen el target), `desemp_ingles`
# (se calcula a partir de punt_ingles) y los `estu_nse_*` (discretización del INSE).


def derivar(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las variables derivadas y marca los faltantes de las categóricas como categoría propia."""
    out = df.copy()

    # 1) Índice TIC: número de recursos digitales del hogar (0, 1 o 2). Resume en una sola
    #    variable la capacidad de estudiar a distancia, que fue la condición de 2020.
    internet = out["fami_tieneinternet"].astype(object).map({"Si": 1.0, "No": 0.0})
    computador = out["fami_tienecomputador"].astype(object).map({"Si": 1.0, "No": 0.0})
    out["indice_tic"] = internet + computador  # queda faltante si falta alguno; se imputa en el pipeline

    # 2) Máximo nivel educativo entre padre y madre (0–9): capital educativo del hogar.
    niveles = {nivel: i for i, nivel in enumerate(c.ORDEN_EDUCACION)}
    padre = out["fami_educacionpadre"].astype(object).map(niveles)
    madre = out["fami_educacionmadre"].astype(object).map(niveles)
    out["educ_max_padres"] = np.fmax(padre.astype(float), madre.astype(float))

    # 3) Extraedad: más de 18 años al presentar el examen (rezago escolar o validación de adultos).
    out["extraedad"] = (out["edad"] > 18).astype("int8")

    # Indicador de INSE faltante: el patrón de ausencia es informativo (ver diagnóstico MAR).
    out["inse_faltante"] = out["estu_inse_individual"].isna().astype("int8")

    for col in list(ORDINALES) + NOMINALES + ALTA_CARDINALIDAD:
        out[col] = out[col].astype(object).where(out[col].notna(), SIN_DATO).astype(str)
    return out


class Winsorizador(BaseEstimator, TransformerMixin):
    """Recorta cada columna a los percentiles aprendidos en el ajuste (solo con train)."""

    def __init__(self, inferior: float = 0.01, superior: float = 0.99) -> None:
        self.inferior = inferior
        self.superior = superior

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lim_inf_ = np.nanquantile(X, self.inferior, axis=0)
        self.lim_sup_ = np.nanquantile(X, self.superior, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lim_inf_, self.lim_sup_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def construir_preprocesador(tarea: str = "regresion") -> ColumnTransformer:
    """ColumnTransformer completo. `tarea` fija el tipo de target del TargetEncoder."""
    numerico = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("winsorizar", Winsorizador(0.01, 0.99)),
        ("escalar", StandardScaler()),
    ])
    ordinal = Pipeline([
        ("codificar", OrdinalEncoder(categories=list(ORDINALES.values()), handle_unknown="use_encoded_value", unknown_value=-1)),
        ("escalar", StandardScaler()),
    ])
    nominal = OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)
    # TargetEncoder de scikit-learn hace validación cruzada interna en fit_transform,
    # así que la codificación de cada fila de train no usa su propio target.
    alta = Pipeline([
        ("codificar", TargetEncoder(target_type="continuous" if tarea == "regresion" else "binary", cv=5, random_state=c.SEED)),
        ("escalar", StandardScaler()),
    ])
    return ColumnTransformer(
        [("num", numerico, NUMERICAS), ("ord", ordinal, list(ORDINALES)), ("nom", nominal, NOMINALES),
         ("depto", alta, ALTA_CARDINALIDAD), ("bin", "passthrough", BINARIAS)],
        verbose_feature_names_out=False,
    )
