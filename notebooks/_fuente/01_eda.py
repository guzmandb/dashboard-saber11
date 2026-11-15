# %% [markdown]
# # 01 · Análisis exploratorio — Saber 11° 2020-2
#
# **Proyecto Integrador de Ciencia de Datos · Maestría en IA, Universidad de La Salle**
# Autor: Jherson Guzmán Ramírez · Docente: Fabián Camilo Castro Riveros
#
# ## 1. Problema y contexto
# El examen Saber 11° es la prueba estandarizada que el ICFES aplica a quienes terminan
# la educación media en Colombia. Su puntaje global condiciona el acceso a la educación
# superior y a programas de financiación. La cohorte 2020-2 presentó la prueba tras meses
# de educación remota por la pandemia, cuando tener internet y computador en casa dejó de
# ser una comodidad y pasó a ser la condición para asistir a clase. Entender cuánto del
# puntaje se asocia con el origen socioeconómico y con el tipo de colegio —y cuánto no—
# permite orientar recursos hacia los grupos donde la brecha es mayor.
#
# **Pregunta principal.** ¿En qué medida las condiciones socioeconómicas del hogar y las
# características del colegio explican y permiten predecir el puntaje global del Saber 11° 2020-2?
#
# **Preguntas secundarias.**
# 1. ¿Qué diferencia de puntaje hay entre colegios oficiales y no oficiales, y entre zona urbana y rural, y qué tan grande es en términos prácticos?
# 2. ¿Se asocian el acceso a internet y computador con el estrato y con el desempeño?
# 3. ¿Existen perfiles de desempeño por áreas que distingan grupos de estudiantes más allá del puntaje global?
#
# **Objetivo general (SMART).** Construir, antes del 28 de noviembre de 2026, un flujo analítico
# reproducible sobre los 504 872 registros del Saber 11° 2020-2 que cuantifique la asociación entre
# factores socioeconómicos y desempeño, y entregue modelos de regresión y clasificación evaluados
# en un conjunto de prueba independiente.
#
# **Objetivos específicos.**
# 1. Describir la distribución de los seis puntajes y diagnosticar la calidad de los datos (faltantes, atípicos, duplicados) antes del 15 de noviembre.
# 2. Contrastar con pruebas de hipótesis y tamaños de efecto las diferencias de puntaje por naturaleza del colegio, zona y estrato antes del 15 de noviembre.
# 3. Identificar con PCA y clustering perfiles de desempeño por áreas, validados con el coeficiente de silueta, antes del 15 de noviembre.

# %%
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy import stats as st

from src import config as c
from src import data, stats, viz

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(c.SEED)
viz.aplicar_estilo()
c.TAB_DIR.mkdir(parents=True, exist_ok=True)

df, muestra = data.construir_procesados()
print(f"Registros: {len(df):,} | columnas: {df.shape[1]} | muestra estratificada: {len(muestra):,}")

# %% [markdown]
# ## 2. Diccionario de datos
# El conjunto original trae 81 columnas. Se conservan 27 con valor analítico; se descartan
# identificadores, códigos DANE y nombres de sede, que no aportan a las preguntas.

# %%
DESCRIPCION = {
    "punt_global": ("Cuantitativa discreta", "Puntaje global del examen", "0–500"),
    "punt_lectura_critica": ("Cuantitativa discreta", "Puntaje en lectura crítica", "0–100"),
    "punt_matematicas": ("Cuantitativa discreta", "Puntaje en matemáticas", "0–100"),
    "punt_c_naturales": ("Cuantitativa discreta", "Puntaje en ciencias naturales", "0–100"),
    "punt_sociales_ciudadanas": ("Cuantitativa discreta", "Puntaje en sociales y ciudadanas", "0–100"),
    "punt_ingles": ("Cuantitativa discreta", "Puntaje en inglés", "0–100"),
    "estu_inse_individual": ("Cuantitativa continua", "Índice de nivel socioeconómico (ICFES)", "0–100"),
    "edad": ("Cuantitativa continua", "Edad al 1-sep-2020 (derivada de la fecha de nacimiento)", "12–70 años"),
    "estu_genero": ("Cualitativa nominal", "Género del estudiante", "F, M"),
    "fami_estratovivienda": ("Cualitativa ordinal", "Estrato de la vivienda", "Sin estrato, 1–6"),
    "fami_educacionpadre": ("Cualitativa ordinal", "Máximo nivel educativo del padre", "Ninguno … Postgrado"),
    "fami_educacionmadre": ("Cualitativa ordinal", "Máximo nivel educativo de la madre", "Ninguno … Postgrado"),
    "fami_tieneinternet": ("Cualitativa nominal", "El hogar tiene internet", "Si, No"),
    "fami_tienecomputador": ("Cualitativa nominal", "El hogar tiene computador", "Si, No"),
    "fami_numlibros": ("Cualitativa ordinal", "Libros en el hogar", "0–10 … más de 100"),
    "fami_personashogar": ("Cualitativa ordinal", "Personas en el hogar", "1–2 … 9 o más"),
    "estu_dedicacionlecturadiaria": ("Cualitativa ordinal", "Tiempo diario de lectura por gusto", "No lee … más de 2 h"),
    "estu_horassemanatrabaja": ("Cualitativa ordinal", "Horas de trabajo a la semana", "0 … más de 30"),
    "cole_naturaleza": ("Cualitativa nominal", "Naturaleza del colegio", "Oficial, No oficial"),
    "cole_area_ubicacion": ("Cualitativa nominal", "Zona del colegio", "Urbano, Rural"),
    "cole_jornada": ("Cualitativa nominal", "Jornada", "Mañana, Tarde, Única, Completa, Noche, Sabatina"),
    "cole_caracter": ("Cualitativa nominal", "Carácter del colegio", "Académico, Técnico, Técnico/Académico"),
    "cole_calendario": ("Cualitativa nominal", "Calendario escolar", "A, B, Otro"),
    "cole_bilingue": ("Cualitativa nominal", "Colegio bilingüe", "S, N"),
    "cole_depto_ubicacion": ("Cualitativa nominal", "Departamento del colegio", "33 valores"),
    "desemp_ingles": ("Cualitativa ordinal", "Nivel de desempeño en inglés (MCER)", "A-, A1, A2, B1, B+"),
}
diccionario = pd.DataFrame(DESCRIPCION, index=["Tipo", "Descripción", "Rango / valores"]).T
diccionario.index.name = "Variable"
diccionario.to_csv(c.TAB_DIR / "diccionario.csv", encoding="utf-8")
diccionario

# %% [markdown]
# ## 3. Calidad de los datos

# %%
falt = stats.faltantes(df)
falt.to_csv(c.TAB_DIR / "faltantes.csv", encoding="utf-8")

crudo = pd.read_csv(c.RAW_CSV, dtype=str, usecols=["estu_consecutivo"])
n_dup = int(crudo["estu_consecutivo"].duplicated().sum())
n_ceros = int((df["punt_global"] == 0).sum())
calidad = pd.Series(
    {
        "registros": len(df),
        "duplicados_por_consecutivo": n_dup,
        "puntaje_global_cero": n_ceros,
        "edad_fuera_de_rango_o_sin_fecha": int(df["edad"].isna().sum()),
        "variables_con_faltantes": int((falt["faltantes"] > 0).sum()),
    }
)
calidad.to_csv(c.TAB_DIR / "calidad.csv", header=["valor"], encoding="utf-8")
print(calidad.to_string())
falt.head(12).round(2)

# %%
top = falt[falt["%"] > 0].head(14).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh([viz.etiqueta(i) for i in top.index], top["%"], color=c.PALETA[0])
for y, v in enumerate(top["%"]):
    ax.text(v + 0.2, y, f"{v:.1f} %", va="center", fontsize=8)
ax.set(xlabel="Registros sin dato (%)", title="Porcentaje de datos faltantes por variable")
viz.guardar(fig, "fig_01_faltantes")

# %%
out = stats.outliers(df, c.NUMERICAS)
out.to_csv(c.TAB_DIR / "outliers.csv", encoding="utf-8")
out.round(2)

# %% [markdown]
# ## 4. Análisis univariado

# %%
desc = stats.descriptivos(df, c.NUMERICAS)
desc.to_csv(c.TAB_DIR / "descriptivos.csv", encoding="utf-8")
desc.round(2)

# %%
# Distribución de frecuencias del puntaje global por intervalos de 50 puntos.
intervalos = pd.cut(df["punt_global"], bins=range(0, 501, 50), include_lowest=True)
freq_global = stats.frecuencias(intervalos).loc[lambda t: t.index.notna()]
freq_global.to_csv(c.TAB_DIR / "freq_puntaje_global.csv", encoding="utf-8")

freq_estrato = stats.frecuencias(df["fami_estratovivienda"]).sort_index()
freq_estrato.to_csv(c.TAB_DIR / "freq_estrato.csv", encoding="utf-8")
freq_cat = {col: stats.frecuencias(df[col]) for col in ["cole_naturaleza", "cole_area_ubicacion", "estu_genero", "fami_tieneinternet", "fami_tienecomputador"]}
pd.concat(freq_cat).to_csv(c.TAB_DIR / "freq_categoricas.csv", encoding="utf-8")
freq_global.round(2)

# %%
fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
for ax, col, color in zip(axes.ravel(), [c.TARGET_REG] + c.PUNTAJES_AREA, c.PALETA):
    x = df[col].dropna()
    # Los puntajes son enteros: el ancho de clase debe ser entero para no crear picos artificiales.
    ax.hist(x, bins=np.arange(0, 510, 10) if col == c.TARGET_REG else np.arange(0, 102, 2), color=color, alpha=0.85)
    ax.axvline(x.mean(), color="black", lw=1, label=f"Media {x.mean():.1f}")
    ax.axvline(x.median(), color="black", lw=1, ls="--", label=f"Mediana {x.median():.0f}")
    ax.set(title=viz.etiqueta(col), xlabel="Puntaje", ylabel="Estudiantes")
    ax.legend(fontsize=7)
fig.suptitle("Distribución de los puntajes del Saber 11° 2020-2 (n = 504 872)", fontweight="bold")
viz.guardar(fig, "fig_02_histogramas_puntajes")

# %% [markdown]
# ### Efectividad de distintos gráficos para el mismo dato
# Una variable cuantitativa (puntaje global) vista como histograma, diagrama de caja y violín;
# una variable cualitativa (estrato) vista como barras, pastel y diagrama de puntos.

# %%
x = df["punt_global"].dropna()
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
axes[0].hist(x, bins=np.arange(0, 510, 10), color=c.PALETA[0])
axes[0].set(title="Histograma", xlabel=viz.etiqueta("punt_global"), ylabel="Estudiantes")
axes[1].boxplot(x, vert=False, widths=0.5, flierprops={"markersize": 2, "alpha": 0.3})
axes[1].set(title="Diagrama de caja", xlabel=viz.etiqueta("punt_global"), yticks=[])
axes[2].violinplot(muestra["punt_global"].dropna(), vert=False, showmedians=True)
axes[2].set(title="Violín (muestra de 60 000)", xlabel=viz.etiqueta("punt_global"), yticks=[])
fig.suptitle("Tres representaciones de la misma variable cuantitativa", fontweight="bold")
viz.guardar(fig, "fig_03_hist_caja_violin")

# %%
f = df["fami_estratovivienda"].value_counts(sort=False)
pct = 100 * f / f.sum()
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
axes[0].bar(pct.index, pct.values, color=c.PALETA[0])
axes[0].set(title="Barras", ylabel="Estudiantes (%)")
axes[0].tick_params(axis="x", rotation=45)
axes[1].pie(pct.values, labels=pct.index, colors=c.PALETA, autopct="%1.0f%%", textprops={"fontsize": 7})
axes[1].set(title="Pastel")
axes[2].hlines(pct.index, 0, pct.values, color="#BBBBBB", lw=1)
axes[2].plot(pct.values, pct.index, "o", color=c.PALETA[3])
axes[2].set(title="Diagrama de puntos", xlabel="Estudiantes (%)")
fig.suptitle("Tres representaciones de la misma variable cualitativa: estrato", fontweight="bold")
viz.guardar(fig, "fig_04_barras_pastel_puntos")

# %% [markdown]
# ## 5. Análisis bivariado y multivariado

# %%
num = c.PUNTAJES_AREA + [c.TARGET_REG, "estu_inse_individual", "edad"]
corr_s = df[num].corr(method="spearman")
corr_p = df[num].corr(method="pearson")
corr_s.to_csv(c.TAB_DIR / "corr_spearman.csv", encoding="utf-8")
corr_p.to_csv(c.TAB_DIR / "corr_pearson.csv", encoding="utf-8")

nombres = ["Lectura", "Matemáticas", "Naturales", "Sociales", "Inglés", "Global", "INSE", "Edad"]
fig, ax = plt.subplots(figsize=(6.2, 5))
sns.heatmap(corr_s, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True,
            xticklabels=nombres, yticklabels=nombres, cbar_kws={"label": "ρ de Spearman"}, ax=ax)
ax.set_title("Matriz de correlación de Spearman entre variables cuantitativas")
viz.guardar(fig, "fig_05_correlacion_spearman")

# %%
m = muestra.dropna(subset=["estu_inse_individual", "punt_global"])
rho, _ = st.spearmanr(df["estu_inse_individual"], df["punt_global"], nan_policy="omit")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(m["estu_inse_individual"], m["punt_global"], s=2, alpha=0.15, color=c.PALETA[0])
axes[0].set(title="Dispersión (muestra de 60 000)", xlabel=viz.etiqueta("estu_inse_individual"), ylabel=viz.etiqueta("punt_global"))
d = df.dropna(subset=["estu_inse_individual", "punt_global"])
hb = axes[1].hexbin(d["estu_inse_individual"], d["punt_global"], gridsize=45, cmap="viridis", mincnt=1, bins="log")
fig.colorbar(hb, ax=axes[1], label="Estudiantes (escala log)")
axes[1].set(title="Hexágonos (todos los registros)", xlabel=viz.etiqueta("estu_inse_individual"), ylabel=viz.etiqueta("punt_global"))
fig.suptitle(f"Nivel socioeconómico y puntaje global (ρ de Spearman = {rho:.2f})", fontweight="bold")
viz.guardar(fig, "fig_06_inse_vs_global")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), gridspec_kw={"width_ratios": [1.6, 1]})
sns.boxplot(data=df, x="fami_estratovivienda", y="punt_global", color=c.PALETA[5], fliersize=0.5, ax=axes[0])
axes[0].set(title="Puntaje global por estrato", xlabel=viz.etiqueta("fami_estratovivienda"), ylabel=viz.etiqueta("punt_global"))
axes[0].tick_params(axis="x", rotation=30)
sns.boxplot(data=df, x="cole_naturaleza", y="punt_global", hue="cole_area_ubicacion", fliersize=0.5, ax=axes[1])
axes[1].set(title="Por naturaleza y zona del colegio", xlabel=viz.etiqueta("cole_naturaleza"), ylabel="")
axes[1].legend(title="Zona", fontsize=8, loc="lower center", ncol=2)
axes[1].set_ylim(-70, 510)
viz.guardar(fig, "fig_07_global_por_estrato_naturaleza")

# %%
# Pruebas de independencia χ² entre variables cualitativas, con V de Cramér.
pares = [
    ("fami_estratovivienda", "fami_tieneinternet"),
    ("fami_estratovivienda", "fami_tienecomputador"),
    ("cole_naturaleza", "cole_area_ubicacion"),
    ("fami_tieneinternet", "cole_area_ubicacion"),
    ("estu_genero", "desemp_ingles"),
]
filas = []
for a, b in pares:
    tabla = pd.crosstab(df[a], df[b])
    chi2, p, v, gl = stats.v_cramer(tabla)
    filas.append({"variable_1": a, "variable_2": b, "chi2": chi2, "gl": gl, "p": p, "V_Cramer": v,
                  "magnitud": stats.interpretar_v(v, min(tabla.shape) - 1)})
chi = pd.DataFrame(filas)
chi.to_csv(c.TAB_DIR / "chi2_eda.csv", index=False, encoding="utf-8")
chi.round(4)

# %%
dep = df.groupby("cole_depto_ubicacion", observed=True)["punt_global"].agg(["mean", "count"]).sort_values("mean")
dep.to_csv(c.TAB_DIR / "global_por_departamento.csv", encoding="utf-8")
fig, ax = plt.subplots(figsize=(6.5, 7))
ax.hlines(dep.index, dep["mean"].min() - 5, dep["mean"], color="#CCCCCC", lw=1)
ax.plot(dep["mean"], dep.index, "o", color=c.PALETA[0])
ax.axvline(df["punt_global"].mean(), color=c.PALETA[3], ls="--", lw=1, label=f"Media nacional {df['punt_global'].mean():.0f}")
ax.set(title="Puntaje global promedio por departamento", xlabel=viz.etiqueta("punt_global"))
ax.tick_params(axis="y", labelsize=7)
ax.legend()
viz.guardar(fig, "fig_08_global_por_departamento")

# %% [markdown]
# ## 6. Un mismo hallazgo en tres bibliotecas de Python
# El hallazgo: el puntaje global medio sube del estrato 1 al 4 y vuelve a bajar en los estratos 5 y 6,
# que en este periodo son grupos pequeños y con mucha más dispersión.
# Se grafica igual en Matplotlib, Seaborn y Plotly (la versión en ggplot2 está en `r/01_descriptiva.R`)
# y se cronometra cuánto tarda cada una en dibujar una dispersión con todos los registros.

# %%
res = (df.dropna(subset=["fami_estratovivienda"])
       .groupby("fami_estratovivienda", observed=True)["punt_global"]
       .agg(media="mean", sd="std", n="count").reset_index())
res["ic95"] = 1.96 * res["sd"] / np.sqrt(res["n"])
res.to_csv(c.TAB_DIR / "global_por_estrato.csv", index=False, encoding="utf-8")
TITULO = "Puntaje global medio por estrato (IC 95 %)"

# Matplotlib: control total, todo se declara a mano.
t0 = time.perf_counter()
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.bar(res["fami_estratovivienda"].astype(str), res["media"], yerr=res["ic95"], color=c.PALETA[0], capsize=3)
ax.set(title=f"Matplotlib · {TITULO}", xlabel=viz.etiqueta("fami_estratovivienda"), ylabel=viz.etiqueta("punt_global"), ylim=(0, 320))
ax.tick_params(axis="x", rotation=30)
viz.guardar(fig, "fig_09a_estrato_matplotlib")
t_mpl = time.perf_counter() - t0

# Seaborn: calcula la media y el intervalo por su cuenta a partir de los datos crudos.
t0 = time.perf_counter()
fig, ax = plt.subplots(figsize=(6, 3.8))
sns.barplot(data=df, x="fami_estratovivienda", y="punt_global", errorbar=("ci", 95), n_boot=200, seed=c.SEED, color=c.PALETA[2], ax=ax)
ax.set(title=f"Seaborn · {TITULO}", xlabel=viz.etiqueta("fami_estratovivienda"), ylabel=viz.etiqueta("punt_global"), ylim=(0, 320))
ax.tick_params(axis="x", rotation=30)
viz.guardar(fig, "fig_09b_estrato_seaborn")
t_sns = time.perf_counter() - t0

# Plotly: interactivo; se exporta HTML para el informe y PNG estático para el PDF.
t0 = time.perf_counter()
figp = px.bar(res, x="fami_estratovivienda", y="media", error_y="ic95", color_discrete_sequence=[c.PALETA[3]],
              labels={"fami_estratovivienda": viz.etiqueta("fami_estratovivienda"), "media": viz.etiqueta("punt_global")},
              title=f"Plotly · {TITULO}", hover_data={"n": ":,", "sd": ":.1f"}, template="plotly_white")
figp.update_yaxes(range=[0, 320])
figp.update_layout(font={"size": 20}, title_font_size=22, margin={"t": 70})
figp.write_html(c.FIG_DIR / "fig_09c_estrato_plotly.html", include_plotlyjs="cdn")
figp.write_image(c.FIG_DIR / "fig_09c_estrato_plotly.png", width=900, height=570, scale=2)
t_ply = time.perf_counter() - t0

# %%
# Rendimiento con muchos datos: dispersión INSE vs. global con todos los registros válidos.
d = df.dropna(subset=["estu_inse_individual", "punt_global"])
tiempos = {}

t0 = time.perf_counter()
fig, ax = plt.subplots(figsize=(5, 4)); ax.scatter(d["estu_inse_individual"], d["punt_global"], s=1, alpha=0.05)
fig.savefig(c.FIG_DIR / "_tmp_mpl.png", dpi=150); plt.close(fig)
tiempos["Matplotlib"] = time.perf_counter() - t0

t0 = time.perf_counter()
fig, ax = plt.subplots(figsize=(5, 4)); sns.scatterplot(data=d, x="estu_inse_individual", y="punt_global", s=2, alpha=0.05, linewidth=0, ax=ax)
fig.savefig(c.FIG_DIR / "_tmp_sns.png", dpi=150); plt.close(fig)
tiempos["Seaborn"] = time.perf_counter() - t0

t0 = time.perf_counter()
px.scatter(d, x="estu_inse_individual", y="punt_global", render_mode="webgl", opacity=0.05).write_html(c.FIG_DIR / "_tmp_ply.html", include_plotlyjs="cdn")
tiempos["Plotly (WebGL, HTML)"] = time.perf_counter() - t0
mb_html = (c.FIG_DIR / "_tmp_ply.html").stat().st_size / 1e6

for tmp in ("_tmp_mpl.png", "_tmp_sns.png", "_tmp_ply.html"):
    (c.FIG_DIR / tmp).unlink()

rend = pd.DataFrame({"segundos": tiempos}).round(2)
rend.loc["Plotly (WebGL, HTML)", "tamano_archivo_MB"] = round(mb_html, 1)
rend.attrs["n"] = len(d)
rend.to_csv(c.TAB_DIR / "rendimiento_python.csv", encoding="utf-8")
print(f"n = {len(d):,} puntos"); print(rend)
pd.Series({"Matplotlib": t_mpl, "Seaborn": t_sns, "Plotly (HTML + PNG)": t_ply}).round(2).to_csv(
    c.TAB_DIR / "rendimiento_barras_python.csv", header=["segundos"], encoding="utf-8")
print(f"Gráfico de barras: Matplotlib {t_mpl:.2f} s | Seaborn {t_sns:.2f} s | Plotly {t_ply:.2f} s")

# %% [markdown]
# ## 7. Hallazgos principales
# Los valores concretos están en `reports/tables/`; la interpretación va en los informes.

# %%
resumen = {
    "media_global": df["punt_global"].mean(),
    "mediana_global": df["punt_global"].median(),
    "desv_global": df["punt_global"].std(),
    "rho_inse_global": rho,
    "brecha_estrato4_vs_1": float(res.set_index("fami_estratovivienda").loc["Estrato 4", "media"] - res.set_index("fami_estratovivienda").loc["Estrato 1", "media"]),
    "pct_sin_internet": 100 * (df["fami_tieneinternet"] == "No").mean(),
    "pct_oficial": 100 * (df["cole_naturaleza"] == "OFICIAL").mean(),
    "pct_rural": 100 * (df["cole_area_ubicacion"] == "RURAL").mean(),
}
pd.Series(resumen).round(3).to_csv(c.TAB_DIR / "resumen_eda.csv", header=["valor"], encoding="utf-8")
pd.Series(resumen).round(2)
