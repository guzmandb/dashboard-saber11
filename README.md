# Factores asociados al puntaje del Saber 11° 2020-2

Proyecto Integrador de Ciencia de Datos · Maestría en Inteligencia Artificial, Universidad de La Salle
Autor: Jherson Guzman Ramarez · Docente: Fabián Camilo Castro Riveros

¿Cuánto del puntaje global del Saber 11° se asocia con las condiciones del hogar y del colegio, y qué tan bien se puede predecir a partir de ellas? El proyecto recorre el ciclo completo sobre los 504 872 estudiantes que presentaron el examen en el segundo semestre de 2020: exploración, pruebas de hipótesis, PCA y clústeres, regresión, modelos de aprendizaje automático, una red neuronal y un tablero interactivo.

## Datos

"Saber 11° 2020-2", ICFES, publicado en Datos Abiertos Colombia (<https://www.datos.gov.co/d/rnvb-vnyh>) con licencia CC BY-SA 4.0. El archivo original (461 MB) no se versiona; se descarga con el segundo comando de abajo. En el repositorio va solo la muestra estratificada de 60 000 estudiantes que usa el tablero.

## Cómo reproducir

Requiere Python 3.11 o superior y, para la réplica en R, R 4.4 con `ggplot2`, `dplyr`, `arrow`, `broom`, `car`, `factoextra`, `effectsize`, `patchwork` y `hexbin`.

```bash
pip install -r requirements.txt
curl -o data/raw/saber11_2020-2_rnvb-vnyh.csv "https://www.datos.gov.co/resource/rnvb-vnyh.csv?\$limit=600000&\$order=:id"
python -m src.py2nb notebooks/_fuente/01_eda.py
```

El tercer comando genera y ejecuta el primer cuaderno, que además crea los datos limpios en `data/processed/`. Los demás se ejecutan igual, en este orden: `02_preprocesamiento`, `03_regresion`, `04_hipotesis`, `05_multivariante`, `06_modelado_ml`, `07_mlp_pytorch`, `08_visualizacion_gran_volumen`, `09_comparativa_herramientas`. Las figuras y tablas se escriben en `reports/`. Los scripts de R se corren desde la raíz con `Rscript --vanilla r/01_descriptiva.R` (y así con los otros tres).

## Tablero

```bash
streamlit run dashboard/app.py
```

Para ejecutar solo el tablero basta `pip install -r dashboard/requirements.txt`.

## Estructura

| Carpeta | Contenido |
|---|---|
| `data/` | `raw/` (descarga original, no versionada) y `processed/` (muestra del tablero; el resto se genera) |
| `notebooks/` | Cuadernos ejecutados (`.ipynb`) y su fuente en `_fuente/*.py` |
| `src/` | Código reutilizable: configuración, carga y limpieza, variables derivadas y preprocesador, estadísticos, figuras, métricas |
| `r/` | Réplica en R: descriptiva, regresión con `lm()`, pruebas de hipótesis, PCA y clústeres |
| `dashboard/` | Tablero en Streamlit y modelos serializados |

## Decisiones de método

- La partición en entrenamiento, validación y prueba (70/15/15) se hace antes de cualquier ajuste. Imputación, codificación y escalado van dentro de un `Pipeline` que se ajusta solo con entrenamiento.
- Los cinco puntajes por área nunca se usan como predictores del puntaje global, porque lo componen.
- Con medio millón de registros casi todo contraste es significativo: las conclusiones se apoyan en tamaños de efecto.
- Semilla fija (`SEED = 42`) en todo el proyecto.
