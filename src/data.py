"""Carga, limpieza mínima y muestreo del Saber 11° 2020-2.

La limpieza de aquí es solo estructural (tipos, etiquetas, edad). La imputación,
el escalado y la codificación viven en el pipeline de modelado, después del split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as c


def cargar_crudo() -> pd.DataFrame:
    """Lee el CSV tal como se descargó, todo como texto para no perder ceros ni códigos."""
    return pd.read_csv(c.RAW_CSV, dtype=str, usecols=lambda col: col in c.COLUMNAS_UTILES)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Tipa las variables, normaliza etiquetas y deriva la edad y el target binario."""
    df = df.copy()

    for col in c.PUNTAJES_AREA + [c.TARGET_REG, "estu_inse_individual"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Edad al 1 de septiembre de 2020 (el examen de calendario A fue en el segundo semestre).
    nac = pd.to_datetime(df["estu_fechanacimiento"], errors="coerce")
    df["edad"] = ((pd.Timestamp("2020-09-01") - nac).dt.days / 365.25).round(1)
    # Edades imposibles para un bachiller son errores de digitación: quedan como faltantes.
    df.loc[(df["edad"] < 12) | (df["edad"] > 70), "edad"] = np.nan
    df = df.drop(columns=["estu_fechanacimiento"])

    for col in c.CATEGORICAS:
        df[col] = df[col].str.strip().replace({"": np.nan, "-": np.nan})

    df["fami_estratovivienda"] = pd.Categorical(
        df["fami_estratovivienda"], categories=c.ORDEN_ESTRATO, ordered=True
    )
    for col in ("fami_educacionpadre", "fami_educacionmadre"):
        # "No sabe" y "No Aplica" no son un nivel educativo: se tratan como faltantes.
        df[col] = pd.Categorical(df[col], categories=c.ORDEN_EDUCACION, ordered=True)
    for col in c.CATEGORICAS:
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")

    return df


def agregar_target_clf(df: pd.DataFrame, umbral: float | None = None) -> tuple[pd.DataFrame, float]:
    """Añade `alto_desempeno`. En modelado, el umbral se calcula solo con train y se pasa aquí."""
    if umbral is None:
        umbral = float(df[c.TARGET_REG].quantile(0.75))
    df = df.copy()
    df[c.TARGET_CLF] = (df[c.TARGET_REG] >= umbral).astype("int8")
    return df, umbral


def muestra_estratificada(df: pd.DataFrame, n: int = c.N_MUESTRA) -> pd.DataFrame:
    """Muestra proporcional por departamento y naturaleza del colegio, reproducible."""
    estratos = ["cole_depto_ubicacion", "cole_naturaleza"]
    frac = n / len(df)
    return (
        df.groupby(estratos, observed=True, dropna=False, group_keys=False)
        .sample(frac=frac, random_state=c.SEED)
        .reset_index(drop=True)
    )


def construir_procesados() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera y guarda el parquet limpio y la muestra estratificada."""
    df = limpiar(cargar_crudo())
    c.CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(c.CLEAN_PARQUET, index=False)
    muestra = muestra_estratificada(df)
    muestra.to_parquet(c.SAMPLE_PARQUET, index=False)
    return df, muestra


def cargar_limpio() -> pd.DataFrame:
    return pd.read_parquet(c.CLEAN_PARQUET)


def cargar_muestra() -> pd.DataFrame:
    return pd.read_parquet(c.SAMPLE_PARQUET)


if __name__ == "__main__":
    completo, muestra = construir_procesados()
    print(f"limpio: {completo.shape} | muestra: {muestra.shape}")
