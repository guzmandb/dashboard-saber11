"""Configuración central del proyecto: rutas, semilla y grupos de variables."""
from __future__ import annotations

from pathlib import Path

SEED: int = 42

ROOT: Path = Path(__file__).resolve().parents[1]
RAW_CSV: Path = ROOT / "data" / "raw" / "saber11_2020-2_rnvb-vnyh.csv"
CLEAN_PARQUET: Path = ROOT / "data" / "processed" / "saber11_2020-2_clean.parquet"
SAMPLE_PARQUET: Path = ROOT / "data" / "processed" / "saber11_2020-2_muestra.parquet"
FIG_DIR: Path = ROOT / "reports" / "figures"
TAB_DIR: Path = ROOT / "reports" / "tables"

DATASET_ID: str = "rnvb-vnyh"
DATASET_URL: str = "https://www.datos.gov.co/d/rnvb-vnyh"
FUENTE: str = "Fuente: ICFES, Saber 11° 2020-2 (datos.gov.co). Elaboración propia."

# Tamaño de la muestra estratificada para lo que no escala (dendrogramas, dispersión, silueta).
N_MUESTRA: int = 60_000

# Puntajes por área: son los componentes del puntaje global, así que nunca entran
# como predictores de punt_global (sería fuga de datos). Sí se usan para clustering.
PUNTAJES_AREA: list[str] = [
    "punt_lectura_critica",
    "punt_matematicas",
    "punt_c_naturales",
    "punt_sociales_ciudadanas",
    "punt_ingles",
]
TARGET_REG: str = "punt_global"
TARGET_CLF: str = "alto_desempeno"  # 1 si punt_global está en el cuartil superior

NUMERICAS: list[str] = PUNTAJES_AREA + [TARGET_REG, "estu_inse_individual", "edad"]

ORDEN_ESTRATO: list[str] = [
    "Sin Estrato", "Estrato 1", "Estrato 2", "Estrato 3", "Estrato 4", "Estrato 5", "Estrato 6",
]
ORDEN_EDUCACION: list[str] = [
    "Ninguno",
    "Primaria incompleta",
    "Primaria completa",
    "Secundaria (Bachillerato) incompleta",
    "Secundaria (Bachillerato) completa",
    "Técnica o tecnológica incompleta",
    "Técnica o tecnológica completa",
    "Educación profesional incompleta",
    "Educación profesional completa",
    "Postgrado",
]

CATEGORICAS: list[str] = [
    "estu_genero",
    "fami_estratovivienda",
    "fami_educacionpadre",
    "fami_educacionmadre",
    "fami_tieneinternet",
    "fami_tienecomputador",
    "fami_numlibros",
    "fami_personashogar",
    "estu_dedicacionlecturadiaria",
    "estu_horassemanatrabaja",
    "cole_naturaleza",
    "cole_area_ubicacion",
    "cole_jornada",
    "cole_caracter",
    "cole_calendario",
    "cole_bilingue",
    "cole_depto_ubicacion",
    "desemp_ingles",
]

# Columnas que se conservan en el parquet limpio (identificadores y textos libres se descartan).
COLUMNAS_UTILES: list[str] = (
    ["periodo", "estu_fechanacimiento", "estu_nse_individual", "estu_nse_establecimiento"]
    + CATEGORICAS
    + PUNTAJES_AREA
    + [TARGET_REG, "estu_inse_individual"]
)

# Paleta apta para daltonismo (Okabe-Ito), compartida por Python y R.
PALETA: list[str] = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000",
]
