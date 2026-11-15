"""Descriptivos, calidad de datos y tamaños de efecto reutilizados por notebooks e informes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st


def descriptivos(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Tendencia central, dispersión y forma de cada variable cuantitativa."""
    filas = {}
    for col in cols:
        x = df[col].dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        filas[col] = {
            "n": int(x.size),
            "media": x.mean(),
            "mediana": x.median(),
            "moda": x.mode().iloc[0],
            "varianza": x.var(ddof=1),
            "desv_est": x.std(ddof=1),
            "min": x.min(),
            "max": x.max(),
            "rango": x.max() - x.min(),
            "IQR": q3 - q1,
            "CV_%": 100 * x.std(ddof=1) / x.mean(),
            "asimetria": st.skew(x, bias=False),
            "curtosis": st.kurtosis(x, bias=False),  # exceso: 0 en la normal
        }
    return pd.DataFrame(filas).T


def frecuencias(s: pd.Series) -> pd.DataFrame:
    """Tabla de frecuencias absoluta, relativa y acumulada (incluye faltantes)."""
    f = s.value_counts(dropna=False, sort=False)
    out = pd.DataFrame({"n": f, "%": 100 * f / f.sum()})
    out["% acum."] = out["%"].cumsum()
    return out


def faltantes(df: pd.DataFrame) -> pd.DataFrame:
    n = df.isna().sum()
    return (
        pd.DataFrame({"faltantes": n, "%": 100 * n / len(df)})
        .sort_values("%", ascending=False)
    )


def outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cuenta atípicos por la regla de Tukey (1,5·IQR) y por |z| > 3."""
    filas = {}
    for col in cols:
        x = df[col].dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_iqr = int(((x < lim_inf) | (x > lim_sup)).sum())
        n_z = int((np.abs((x - x.mean()) / x.std(ddof=1)) > 3).sum())
        filas[col] = {
            "lim_inf": lim_inf,
            "lim_sup": lim_sup,
            "n_IQR": n_iqr,
            "%_IQR": 100 * n_iqr / x.size,
            "n_Z3": n_z,
            "%_Z3": 100 * n_z / x.size,
        }
    return pd.DataFrame(filas).T


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    """d de Cohen con desviación combinada."""
    a, b = a.dropna(), b.dropna()
    s = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    return float((a.mean() - b.mean()) / s)


def eta_cuadrado(grupos: list[pd.Series]) -> float:
    """η² = SC entre grupos / SC total."""
    todos = pd.concat(grupos)
    sc_entre = sum(g.size * (g.mean() - todos.mean()) ** 2 for g in grupos)
    sc_total = ((todos - todos.mean()) ** 2).sum()
    return float(sc_entre / sc_total)


def epsilon_cuadrado(h: float, n: int) -> float:
    """Tamaño de efecto de Kruskal-Wallis."""
    return float(h / (n - 1))


def v_cramer(tabla: pd.DataFrame) -> tuple[float, float, float, int]:
    """χ² de independencia y V de Cramér. Devuelve (chi2, p, V, gl)."""
    chi2, p, gl, _ = st.chi2_contingency(tabla)
    n = tabla.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(tabla.shape) - 1)))
    return float(chi2), float(p), float(v), int(gl)


def interpretar_d(d: float) -> str:
    d = abs(d)
    return "despreciable" if d < 0.2 else "pequeño" if d < 0.5 else "mediano" if d < 0.8 else "grande"


def interpretar_eta(e: float) -> str:
    return "despreciable" if e < 0.01 else "pequeño" if e < 0.06 else "mediano" if e < 0.14 else "grande"


def interpretar_v(v: float, gl_min: int = 1) -> str:
    # Umbrales de Cohen (1988) ajustados por los grados de libertad menores de la tabla.
    p, m, g = (0.1, 0.3, 0.5)
    k = np.sqrt(gl_min)
    return "despreciable" if v < p / k else "pequeño" if v < m / k else "mediano" if v < g / k else "grande"
