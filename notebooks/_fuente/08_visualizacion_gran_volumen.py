# %% [markdown]
# # 08 · Visualización de grandes volúmenes: comparación de técnicas
#
# Con medio millón de estudiantes, un diagrama de dispersión corriente deja de funcionar: los
# puntos se superponen y la figura muestra una mancha, no una distribución. Aquí se dibuja la
# misma relación —nivel socioeconómico (INSE) frente a puntaje global— con seis técnicas y se
# compara qué deja ver cada una, cuánto tarda y cuánto pesa el archivo resultante.

# %%
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as st

from src import config as c
from src import data, viz

warnings.filterwarnings("ignore")
viz.aplicar_estilo()
rng = np.random.default_rng(c.SEED)
df = data.cargar_limpio().dropna(subset=["estu_inse_individual", "punt_global"])
x, y = df["estu_inse_individual"].to_numpy(), df["punt_global"].to_numpy()
N = len(df)
XL, YL = viz.etiqueta("estu_inse_individual"), viz.etiqueta("punt_global")
print(f"{N:,} puntos")

# %%
def cronometrar(nombre, dibujar):
    """Dibuja la técnica sola, la guarda en PNG y mide tiempo y tamaño del archivo."""
    t0 = time.perf_counter()
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    dibujar(ax)
    ruta = c.FIG_DIR / "_tmp_volumen.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    seg, kb = time.perf_counter() - t0, ruta.stat().st_size / 1024
    ruta.unlink()
    return {"tecnica": nombre, "segundos": seg, "PNG_KB": kb}

idx = rng.choice(N, 5000, replace=False)
idx_kde = rng.choice(N, 20000, replace=False)
gx, gy = np.meshgrid(np.linspace(15, 85, 120), np.linspace(80, 430, 120))

def kde(ax):
    # La estimación de densidad por núcleos es cuadrática en n: se ajusta con una muestra de 20 000.
    k = st.gaussian_kde(np.vstack([x[idx_kde], y[idx_kde]]))
    z = k(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contourf(gx, gy, z, levels=12, cmap="viridis")
    ax.contour(gx, gy, z, levels=6, colors="white", linewidths=0.4)

TECNICAS = {
    "Dispersión con todos los puntos": lambda ax: ax.scatter(x, y, s=3, color=c.PALETA[0]),
    "Dispersión con transparencia (alfa = 0,02)": lambda ax: ax.scatter(x, y, s=3, alpha=0.02, color=c.PALETA[0], linewidths=0),
    "Muestra aleatoria de 5 000": lambda ax: ax.scatter(x[idx], y[idx], s=3, alpha=0.4, color=c.PALETA[0], linewidths=0),
    "Hexágonos (escala logarítmica)": lambda ax: ax.hexbin(x, y, gridsize=45, cmap="viridis", bins="log", mincnt=1),
    "Histograma 2D": lambda ax: ax.hist2d(x, y, bins=[70, 70], cmap="viridis", cmin=1),
    "Densidad por núcleos (KDE, muestra de 20 000)": kde,
}
tiempos = pd.DataFrame([cronometrar(n, f) for n, f in TECNICAS.items()])

fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex=True, sharey=True)
for ax, (nombre, dibujar) in zip(axes.ravel(), TECNICAS.items()):
    dibujar(ax)
    ax.set_title(nombre, fontsize=9)
    ax.set(xlim=(15, 85), ylim=(80, 430))
for ax in axes[1]: ax.set_xlabel(XL)
for ax in axes[:, 0]: ax.set_ylabel(YL)
fig.suptitle(f"La misma relación con seis técnicas (n = {N:,})".replace(",", " "), fontweight="bold")
viz.guardar(fig, "fig_38_tecnicas_gran_volumen")

# %% [markdown]
# ## Qué conserva cada técnica
# Además del tiempo, importa cuánta información sobrevive. Se mide con dos preguntas concretas:
# ¿se ve dónde está la mayor concentración de estudiantes?, ¿se ven los casos raros (las colas)?

# %%
t0 = time.perf_counter()
H, xe, ye = np.histogram2d(x, y, bins=[70, 70], range=[[15, 85], [80, 430]])
figp = go.Figure(go.Heatmap(z=np.where(H.T > 0, H.T, np.nan), x=(xe[:-1] + xe[1:]) / 2, y=(ye[:-1] + ye[1:]) / 2, colorscale="Viridis",
                            colorbar={"title": "Estudiantes"}, hovertemplate="INSE: %{x:.0f}<br>Puntaje: %{y:.0f}<br>Estudiantes: %{z}<extra></extra>"))
figp.update_layout(template="plotly_white", width=900, height=620, font={"size": 18}, title_font_size=19,
                   title="Nivel socioeconómico y puntaje global: densidad interactiva", xaxis_title=XL, yaxis_title=YL)
ruta_html = c.FIG_DIR / "fig_39_densidad_interactiva.html"
figp.write_html(ruta_html, include_plotlyjs="cdn")
seg_plotly = time.perf_counter() - t0
figp.write_image(c.FIG_DIR / "fig_39_densidad_interactiva.png", scale=2)

# Dispersión interactiva con todos los puntos: lo que NO conviene hacer (solo se mide, no se guarda).
t0 = time.perf_counter()
ruta_tmp = c.FIG_DIR / "_tmp_scatter.html"
go.Figure(go.Scattergl(x=x, y=y, mode="markers", marker={"size": 2, "opacity": 0.05})).write_html(ruta_tmp, include_plotlyjs="cdn")
seg_sc, mb_sc = time.perf_counter() - t0, ruta_tmp.stat().st_size / 1e6
ruta_tmp.unlink()

tiempos["ve_la_concentracion"] = ["No", "Sí", "Parcialmente", "Sí", "Sí", "Sí"]
tiempos["ve_las_colas"] = ["Sí", "No", "No", "Sí", "Sí", "No"]
tiempos["puntos_usados"] = [N, N, 5000, N, N, 20000]
tiempos.to_csv(c.TAB_DIR / "vol_tecnicas.csv", index=False, encoding="utf-8")
pd.Series({"n": N, "plotly_densidad_seg": seg_plotly, "plotly_densidad_html_MB": ruta_html.stat().st_size / 1e6,
           "plotly_dispersion_webgl_seg": seg_sc, "plotly_dispersion_webgl_html_MB": mb_sc,
           "celdas_ocupadas_hist2d": int((H > 0).sum()), "max_estudiantes_por_celda": int(H.max()),
           "pct_puntos_en_el_10pct_de_celdas_mas_densas": 100 * np.sort(H.ravel())[::-1][: int(0.1 * (H > 0).sum())].sum() / N}).to_csv(
    c.TAB_DIR / "vol_resumen.csv", header=["valor"], encoding="utf-8")
tiempos.round(2)
