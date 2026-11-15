# %% [markdown]
# # 09 · Un mismo gráfico en varias bibliotecas
#
# Dos comparaciones adicionales entre Matplotlib, Seaborn y Plotly (la versión en ggplot2 está en
# los scripts de R):
# 1. un gráfico básico: el histograma del puntaje global;
# 2. un gráfico multivariante: los clústeres en el plano de las dos primeras componentes.
#
# En cada caso se anota cuánto tarda cada biblioteca y cuánto pesa el archivo que produce.

# %%
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from src import config as c
from src import data, viz

warnings.filterwarnings("ignore")
viz.aplicar_estilo()
df = data.cargar_limpio()
x = df["punt_global"]
filas = []

def anotar(grafico, herramienta, t0, ruta):
    filas.append({"grafico": grafico, "herramienta": herramienta, "segundos": time.perf_counter() - t0, "archivo_KB": ruta.stat().st_size / 1024})

# %% [markdown]
# ## 1. Histograma del puntaje global (504 872 estudiantes)

# %%
XL, TIT = viz.etiqueta("punt_global"), "Distribución del puntaje global"

# Matplotlib: hay que dar los intervalos; por defecto usaría solo 10.
t0 = time.perf_counter()
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.hist(x, bins=np.arange(0, 510, 10), color=c.PALETA[0])
ax.axvline(x.mean(), color="black", ls="--", lw=1)
ax.set(title=f"Matplotlib · {TIT}", xlabel=XL, ylabel="Estudiantes")
viz.guardar(fig, "fig_40a_histograma_matplotlib")
anotar("histograma", "Matplotlib", t0, c.FIG_DIR / "fig_40a_histograma_matplotlib.png")

# Seaborn: el mismo histograma y, con un argumento más, la curva de densidad.
t0 = time.perf_counter()
fig, ax = plt.subplots(figsize=(6, 3.8))
sns.histplot(x=x, binwidth=10, kde=True, color=c.PALETA[2], ax=ax)
ax.set(title=f"Seaborn · {TIT}", xlabel=XL, ylabel="Estudiantes")
viz.guardar(fig, "fig_40b_histograma_seaborn")
anotar("histograma", "Seaborn", t0, c.FIG_DIR / "fig_40b_histograma_seaborn.png")

# Plotly: interactivo. Con los datos crudos el HTML lleva los 504 872 valores; con los datos
# ya contados por intervalo pesa una fracción.
t0 = time.perf_counter()
ruta_crudo = c.FIG_DIR / "_tmp_hist_crudo.html"
px.histogram(df, x="punt_global", nbins=50).write_html(ruta_crudo, include_plotlyjs="cdn")
anotar("histograma", "Plotly (datos crudos, HTML)", t0, ruta_crudo)
ruta_crudo.unlink()

t0 = time.perf_counter()
conteo = pd.cut(x, np.arange(0, 510, 10), include_lowest=True).value_counts(sort=False)
tabla = pd.DataFrame({"puntaje": [i.mid for i in conteo.index], "estudiantes": conteo.to_numpy()})
figp = px.bar(tabla, x="puntaje", y="estudiantes", color_discrete_sequence=[c.PALETA[3]], template="plotly_white",
              labels={"puntaje": XL, "estudiantes": "Estudiantes"}, title=f"Plotly · {TIT}")
figp.update_layout(bargap=0, font={"size": 20}, title_font_size=22, margin={"t": 70})
ruta_html = c.FIG_DIR / "fig_40c_histograma_plotly.html"
figp.write_html(ruta_html, include_plotlyjs="cdn")
anotar("histograma", "Plotly (datos agregados, HTML)", t0, ruta_html)
figp.write_image(c.FIG_DIR / "fig_40c_histograma_plotly.png", width=900, height=570, scale=2)

# %% [markdown]
# ## 2. Clústeres en el plano de componentes principales (muestra de 8 000)

# %%
m = pd.read_parquet(c.ROOT / "data" / "processed" / "muestra_pca_clusters.parquet").sample(8000, random_state=c.SEED).sort_values("cluster")
grupos = list(m["perfil"].unique())
TIT2 = "Clústeres en el plano de componentes principales"

# Matplotlib: un llamado por grupo para que cada uno tenga color y entrada en la leyenda.
t0 = time.perf_counter()
fig, ax = plt.subplots(figsize=(6, 4.2))
for g, color in zip(grupos, c.PALETA):
    s = m[m["perfil"] == g]
    ax.scatter(s["CP1"], s["CP2"], s=4, alpha=0.4, color=color, label=g, linewidths=0)
ax.set(title=f"Matplotlib · {TIT2}", xlabel="CP1 · desempeño general", ylabel="CP2 · inglés frente al resto")
ax.legend(fontsize=7, markerscale=3)
viz.guardar(fig, "fig_41a_clusters_matplotlib")
anotar("clusteres", "Matplotlib", t0, c.FIG_DIR / "fig_41a_clusters_matplotlib.png")

# Seaborn: el color por grupo y la leyenda salen del argumento `hue`.
t0 = time.perf_counter()
fig, ax = plt.subplots(figsize=(6, 4.2))
sns.scatterplot(data=m, x="CP1", y="CP2", hue="perfil", palette=c.PALETA[:len(grupos)], s=6, alpha=0.4, linewidth=0, ax=ax)
ax.set(title=f"Seaborn · {TIT2}", xlabel="CP1 · desempeño general", ylabel="CP2 · inglés frente al resto")
ax.legend(fontsize=7, title=None, markerscale=2)
viz.guardar(fig, "fig_41b_clusters_seaborn")
anotar("clusteres", "Seaborn", t0, c.FIG_DIR / "fig_41b_clusters_seaborn.png")

# Plotly: la versión interactiva ya se genera en el cuaderno 05; aquí solo se mide.
t0 = time.perf_counter()
ruta_tmp = c.FIG_DIR / "_tmp_clusters.html"
px.scatter(m, x="CP1", y="CP2", color="perfil", opacity=0.45, render_mode="webgl", hover_data=["punt_global", "cole_naturaleza"]).write_html(ruta_tmp, include_plotlyjs="cdn")
anotar("clusteres", "Plotly (WebGL, HTML)", t0, ruta_tmp)
ruta_tmp.unlink()

medidas = pd.DataFrame(filas)
medidas.to_csv(c.TAB_DIR / "comparativa_adicional.csv", index=False, encoding="utf-8")
medidas.round(2)
