"""Estilo común y utilidades de guardado para las figuras del proyecto."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from . import config as c

ETIQUETAS: dict[str, str] = {
    "punt_global": "Puntaje global (0–500)",
    "punt_lectura_critica": "Lectura crítica (0–100)",
    "punt_matematicas": "Matemáticas (0–100)",
    "punt_c_naturales": "Ciencias naturales (0–100)",
    "punt_sociales_ciudadanas": "Sociales y ciudadanas (0–100)",
    "punt_ingles": "Inglés (0–100)",
    "estu_inse_individual": "Índice socioeconómico INSE (0–100)",
    "edad": "Edad (años)",
    "fami_estratovivienda": "Estrato de la vivienda",
    "cole_naturaleza": "Naturaleza del colegio",
    "cole_area_ubicacion": "Zona del colegio",
    "estu_genero": "Género",
    "fami_tieneinternet": "Internet en el hogar",
    "fami_tienecomputador": "Computador en el hogar",
    "fami_educacionmadre": "Educación de la madre",
    "fami_educacionpadre": "Educación del padre",
    "cole_jornada": "Jornada del colegio",
    "cole_depto_ubicacion": "Departamento del colegio",
    "cole_bilingue": "Colegio bilingüe",
    "cole_caracter": "Carácter del colegio",
    "cole_calendario": "Calendario escolar",
    "estu_horassemanatrabaja": "Horas de trabajo semanales",
    "estu_dedicacionlecturadiaria": "Lectura diaria",
    "fami_personashogar": "Personas en el hogar",
    "fami_numlibros": "Libros en el hogar",
    "estu_nse_individual": "Nivel socioeconómico (NSE)",
    "estu_nse_establecimiento": "NSE del colegio",
    "desemp_ingles": "Nivel de inglés",
}


def etiqueta(col: str) -> str:
    return ETIQUETAS.get(col, col)


def aplicar_estilo() -> None:
    """Estilo sobrio y legible en impresión; mismo criterio para todas las figuras."""
    sns.set_theme(style="whitegrid", palette=c.PALETA)
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
        }
    )


def guardar(fig: Figure, nombre: str, fuente: bool = True) -> str:
    """Guarda en reports/figures a 300 dpi con la nota de fuente al pie."""
    c.FIG_DIR.mkdir(parents=True, exist_ok=True)
    if fuente:
        fig.supxlabel(c.FUENTE, fontsize=7, color="#555555", x=0.01, ha="left")
    ruta = c.FIG_DIR / f"{nombre}.png"
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(ruta)
