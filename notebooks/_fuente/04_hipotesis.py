# %% [markdown]
# # 04 · Pruebas de hipótesis — Saber 11° 2020-2
#
# Se contrastan cuatro hipótesis sobre el puntaje global con t de Student, ANOVA y χ².
# Antes de cada prueba se verifican sus supuestos; cuando fallan se reporta también la
# alternativa robusta o no paramétrica. Con medio millón de registros casi cualquier
# diferencia resulta "significativa", así que la conclusión de cada contraste descansa
# en el **tamaño del efecto** y no en el valor p.
#
# | # | Hipótesis nula | Prueba |
# |---|---|---|
# | H1 | El puntaje global medio es igual en colegios oficiales y no oficiales | t de Welch |
# | H2 | El puntaje global medio es igual en zona urbana y rural | t de Welch |
# | H3 | El puntaje global medio es igual en todos los estratos | ANOVA + Tukey |
# | H4 | Tener internet en el hogar es independiente de la zona del colegio | χ² |
# | H5 | Estar en el cuartil superior es independiente del estrato | χ² |
#
# Nivel de significancia: α = 0,05.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats as st
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from src import config as c
from src import data, stats, viz

warnings.filterwarnings("ignore")
rng = np.random.default_rng(c.SEED)
viz.aplicar_estilo()

df = data.cargar_limpio()
df, umbral = data.agregar_target_clf(df)
Y = "punt_global"
print(f"n = {len(df):,} | umbral del cuartil superior = {umbral:.0f} puntos")

# %% [markdown]
# ## 1. Supuestos
# **Normalidad.** Shapiro-Wilk pierde validez por encima de 5 000 observaciones, así que se
# aplica a una submuestra aleatoria de 5 000 por grupo y se complementa con Anderson-Darling,
# la asimetría, la curtosis y los gráficos Q-Q. **Homocedasticidad.** Prueba de Levene
# (centrada en la mediana) y razón entre la mayor y la menor desviación estándar.

# %%
def supuestos(col: str) -> pd.DataFrame:
    filas = []
    for nivel, g in df.dropna(subset=[col]).groupby(col, observed=True)[Y]:
        sub = g.sample(min(5000, len(g)), random_state=c.SEED)
        w, p_sw = st.shapiro(sub)
        ad = st.anderson(sub, dist="norm")
        filas.append({"variable": col, "grupo": str(nivel), "n": len(g), "media": g.mean(), "desv_est": g.std(),
                      "asimetria": st.skew(g), "curtosis": st.kurtosis(g), "W_Shapiro(n=5000)": w, "p_Shapiro": p_sw,
                      "A2_Anderson": ad.statistic, "A2_critico_5%": ad.critical_values[2]})
    t = pd.DataFrame(filas)
    lev, p_lev = st.levene(*[g.to_numpy() for _, g in df.dropna(subset=[col]).groupby(col, observed=True)[Y]], center="median")
    t["W_Levene"], t["p_Levene"] = lev, p_lev
    t["razon_desv_max_min"] = t["desv_est"].max() / t["desv_est"].min()
    return t

sup = pd.concat([supuestos(v) for v in ["cole_naturaleza", "cole_area_ubicacion", "fami_estratovivienda"]], ignore_index=True)
sup.to_csv(c.TAB_DIR / "hip_supuestos.csv", index=False, encoding="utf-8")
sup.round(4)

# %%
fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
for ax, (col, nivel) in zip(axes, [("cole_naturaleza", "OFICIAL"), ("cole_naturaleza", "NO OFICIAL"),
                                   ("cole_area_ubicacion", "URBANO"), ("cole_area_ubicacion", "RURAL")]):
    sub = df.loc[df[col] == nivel, Y].sample(5000, random_state=c.SEED)
    (osm, osr), (pend, inter, _) = st.probplot(sub, dist="norm")
    ax.scatter(osm, osr, s=3, alpha=0.4, color=c.PALETA[0])
    ax.plot(osm, pend * osm + inter, color=c.PALETA[3], lw=1)
    ax.set(title=nivel.capitalize(), xlabel="Cuantiles teóricos N(0,1)", ylabel="Puntaje global observado")
fig.suptitle("Gráficos Q-Q del puntaje global por grupo (submuestras de 5 000)", fontweight="bold")
viz.guardar(fig, "fig_10_qq_por_grupo")

# %% [markdown]
# ## 2. H1 y H2 — comparación de dos medias
# Las varianzas difieren entre grupos (Levene), por lo que se usa la t de **Welch**, que no
# exige igualdad de varianzas. Como la normalidad se rechaza formalmente, se añade
# Mann-Whitney. Con estos tamaños de muestra el teorema del límite central respalda la
# prueba t sobre las medias aunque los datos no sean normales.

# %%
def dos_grupos(col: str, a: str, b: str) -> dict:
    x, y = df.loc[df[col] == a, Y], df.loc[df[col] == b, Y]
    t, p = st.ttest_ind(x, y, equal_var=False)
    gl = (x.var() / len(x) + y.var() / len(y)) ** 2 / ((x.var() / len(x)) ** 2 / (len(x) - 1) + (y.var() / len(y)) ** 2 / (len(y) - 1))
    dif = x.mean() - y.mean()
    ee = np.sqrt(x.var() / len(x) + y.var() / len(y))
    d = stats.cohen_d(x, y)
    ee_d = np.sqrt((len(x) + len(y)) / (len(x) * len(y)) + d**2 / (2 * (len(x) + len(y))))
    u, p_u = st.mannwhitneyu(x, y, alternative="two-sided")
    return {"variable": col, "grupo_A": a, "grupo_B": b, "n_A": len(x), "n_B": len(y), "media_A": x.mean(), "media_B": y.mean(),
            "diferencia": dif, "IC95_inf": dif - 1.96 * ee, "IC95_sup": dif + 1.96 * ee, "t_Welch": t, "gl": gl, "p_t": p,
            "d_Cohen": d, "d_IC95_inf": d - 1.96 * ee_d, "d_IC95_sup": d + 1.96 * ee_d, "magnitud": stats.interpretar_d(d),
            "U_Mann_Whitney": u, "p_U": p_u, "r_biserial": 2 * u / (len(x) * len(y)) - 1}

contrastes = [("cole_naturaleza", "NO OFICIAL", "OFICIAL"), ("cole_area_ubicacion", "URBANO", "RURAL"),
              ("fami_tieneinternet", "Si", "No"), ("fami_tienecomputador", "Si", "No"), ("estu_genero", "M", "F")]
tt = pd.DataFrame([dos_grupos(*x) for x in contrastes])
tt.to_csv(c.TAB_DIR / "hip_t.csv", index=False, encoding="utf-8")
tt[["variable", "grupo_A", "grupo_B", "media_A", "media_B", "diferencia", "t_Welch", "p_t", "d_Cohen", "magnitud", "p_U", "r_biserial"]].round(3)

# %%
ROT = {"cole_naturaleza": "No oficial − oficial", "cole_area_ubicacion": "Urbano − rural", "fami_tieneinternet": "Con − sin internet",
       "fami_tienecomputador": "Con − sin computador", "estu_genero": "Hombres − mujeres"}
o = tt.sort_values("d_Cohen")
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
axes[0].errorbar(o["diferencia"], [ROT[v] for v in o["variable"]], xerr=1.96 * (o["IC95_sup"] - o["diferencia"]) / 1.96, fmt="o", color=c.PALETA[0], capsize=3)
axes[0].axvline(0, color="black", lw=0.8)
axes[0].set(title="Diferencia de medias (IC 95 %)", xlabel="Puntos del puntaje global")
axes[1].errorbar(o["d_Cohen"], [ROT[v] for v in o["variable"]], xerr=o["d_IC95_sup"] - o["d_Cohen"], fmt="o", color=c.PALETA[3], capsize=3)
for lim, txt in [(0.2, "pequeño"), (0.5, "mediano"), (0.8, "grande")]:
    axes[1].axvline(lim, color="#999999", ls=":", lw=1)
    axes[1].text(lim, len(o) - 0.45, txt, fontsize=9, ha="center", color="#555555")
axes[1].axvline(0, color="black", lw=0.8)
axes[1].set(title="Tamaño del efecto: d de Cohen (IC 95 %)", xlabel="d de Cohen")
viz.guardar(fig, "fig_11_diferencias_y_efectos")

# %% [markdown]
# ## 3. H3 — ANOVA de un factor por estrato
# Con varianzas desiguales se reporta el ANOVA clásico, el de **Welch** y Kruskal-Wallis.
# Las comparaciones por pares usan Tukey HSD (el que pide el curso) y Games-Howell, que es
# el indicado cuando las varianzas difieren.

# %%
d3 = df.dropna(subset=["fami_estratovivienda"]).copy()
d3["estrato"] = d3["fami_estratovivienda"].astype(str)
grupos = [g for _, g in d3.groupby("fami_estratovivienda", observed=True)[Y]]
F, p_F = st.f_oneway(*grupos)
welch = pg.welch_anova(data=d3, dv=Y, between="estrato")
H, p_H = st.kruskal(*grupos)
eta2 = stats.eta_cuadrado(grupos)
k, n = len(grupos), len(d3)
anova = pd.DataFrame([{
    "F_clasico": F, "gl_entre": k - 1, "gl_dentro": n - k, "p_F": p_F, "eta2": eta2, "omega2": (F - 1) * (k - 1) / ((F - 1) * (k - 1) + n),
    "magnitud": stats.interpretar_eta(eta2), "F_Welch": welch["F"].iloc[0], "gl2_Welch": welch["ddof2"].iloc[0], "p_Welch": welch["p_unc"].iloc[0] if "p_unc" in welch else welch.filter(like="p").iloc[0, 0],
    "H_Kruskal": H, "p_H": p_H, "epsilon2": stats.epsilon_cuadrado(H, n)}])
anova.to_csv(c.TAB_DIR / "hip_anova.csv", index=False, encoding="utf-8")
anova.round(4)

# %%
tk = pairwise_tukeyhsd(d3[Y].to_numpy(), d3["estrato"].to_numpy(), alpha=0.05)
tukey = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])
desv = d3.groupby("estrato")[Y].agg(["std", "count"])
sp = lambda a, b: np.sqrt(((desv.loc[a, "count"] - 1) * desv.loc[a, "std"] ** 2 + (desv.loc[b, "count"] - 1) * desv.loc[b, "std"] ** 2) / (desv.loc[a, "count"] + desv.loc[b, "count"] - 2))
tukey["d_Cohen"] = [m / sp(a, b) for a, b, m in zip(tukey["group1"], tukey["group2"], tukey["meandiff"])]
gh = pg.pairwise_gameshowell(data=d3, dv=Y, between="estrato")
col_p = [x for x in gh.columns if x.startswith("p")][0]
tukey = tukey.merge(gh[["A", "B", col_p]].rename(columns={"A": "group1", "B": "group2", col_p: "p_Games_Howell"}), how="left")
tukey.to_csv(c.TAB_DIR / "hip_tukey.csv", index=False, encoding="utf-8")
tukey.round(3)

# %%
tukey["par"] = tukey["group2"].str.replace("Estrato ", "E") + " − " + tukey["group1"].str.replace("Estrato ", "E")
o = tukey.sort_values("meandiff")
fig, ax = plt.subplots(figsize=(6.5, 5.6))
colores = [c.PALETA[0] if r else "#999999" for r in o["reject"]]
ax.hlines(o["par"], o["lower"], o["upper"], colors=colores, lw=2)
ax.scatter(o["meandiff"], o["par"], c=colores, s=18, zorder=3)
ax.axvline(0, color="black", lw=0.8)
ax.set(title="Comparaciones por pares de Tukey HSD (IC 95 % simultáneo)", xlabel="Diferencia de medias del puntaje global")
ax.tick_params(axis="y", labelsize=9)
viz.guardar(fig, "fig_12_tukey_estrato")

# %% [markdown]
# ## 4. H4 y H5 — χ² de independencia
# Supuesto: frecuencias esperadas ≥ 5 en todas las celdas (se verifica el mínimo).
# Los residuos estandarizados de Pearson muestran qué celdas sostienen la asociación.

# %%
def chi(a: str, b: str) -> tuple[dict, pd.DataFrame]:
    tabla = pd.crosstab(df[a], df[b])
    chi2, p, gl, esp = st.chi2_contingency(tabla)
    v = np.sqrt(chi2 / (tabla.to_numpy().sum() * (min(tabla.shape) - 1)))
    resid = (tabla - esp) / np.sqrt(esp)
    return ({"variable_1": a, "variable_2": b, "n": int(tabla.to_numpy().sum()), "chi2": chi2, "gl": gl, "p": p,
             "esperada_minima": esp.min(), "V_Cramer": v, "magnitud": stats.interpretar_v(v, min(tabla.shape) - 1)}, resid)

r4, res4 = chi("fami_tieneinternet", "cole_area_ubicacion")
r5, res5 = chi("fami_estratovivienda", c.TARGET_CLF)
r6, _ = chi("cole_naturaleza", c.TARGET_CLF)
chi_t = pd.DataFrame([r4, r5, r6])
chi_t.to_csv(c.TAB_DIR / "hip_chi2.csv", index=False, encoding="utf-8")
pct5 = pd.crosstab(df["fami_estratovivienda"], df[c.TARGET_CLF], normalize="index") * 100
pct5.to_csv(c.TAB_DIR / "hip_pct_alto_por_estrato.csv", encoding="utf-8")
pct4 = pd.crosstab(df["cole_area_ubicacion"], df["fami_tieneinternet"], normalize="index") * 100
pct4.to_csv(c.TAB_DIR / "hip_pct_internet_por_zona.csv", encoding="utf-8")
print(pct4.round(1)); chi_t.round(4)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1, 1.3]})
res5.columns = ["Resto (75 %)", "Cuartil superior"]
sns.heatmap(res5, annot=True, fmt=".0f", cmap="RdBu_r", center=0, cbar_kws={"label": "Residuo estandarizado"}, ax=axes[0])
axes[0].set(title="Residuos de Pearson: estrato × alto desempeño", xlabel="", ylabel="")
axes[1].bar(pct5.index.astype(str), pct5[1], color=c.PALETA[2])
axes[1].axhline(100 * df[c.TARGET_CLF].mean(), color="black", ls="--", lw=1, label=f"Total: {100 * df[c.TARGET_CLF].mean():.1f} %".replace(".", ","))
for i, v in enumerate(pct5[1]):
    axes[1].text(i, v + 0.8, f"{v:.0f} %", ha="center", fontsize=8)
axes[1].set(title="Estudiantes en el cuartil superior por estrato", ylabel="Estudiantes (%)")
axes[1].tick_params(axis="x", rotation=30)
axes[1].legend()
viz.guardar(fig, "fig_13_chi2_estrato_alto")

# %% [markdown]
# ## 5. Por qué no basta el valor p con muestras grandes
# La diferencia entre hombres y mujeres es pequeña. Se repite la prueba t con submuestras
# de tamaño creciente: el tamaño del efecto se mantiene, mientras el valor p cae hasta
# cruzar cualquier umbral.

# %%
h, m = df.loc[df["estu_genero"] == "M", Y].to_numpy(), df.loc[df["estu_genero"] == "F", Y].to_numpy()
tam = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 200000]
filas = []
for n_g in tam:
    ps, ds = [], []
    for _ in range(200):
        a, b = rng.choice(h, n_g, replace=False), rng.choice(m, n_g, replace=False)
        ps.append(st.ttest_ind(a, b, equal_var=False).pvalue)
        ds.append((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    filas.append({"n_por_grupo": n_g, "p_mediana": np.median(ps), "pct_significativo": 100 * np.mean(np.array(ps) < 0.05),
                  "d_mediana": np.median(ds), "d_p05": np.quantile(ds, 0.05), "d_p95": np.quantile(ds, 0.95)})
pn = pd.DataFrame(filas)
pn.to_csv(c.TAB_DIR / "hip_p_vs_n.csv", index=False, encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(pn["n_por_grupo"], pn["p_mediana"].clip(lower=1e-300), "o-", color=c.PALETA[0])
axes[0].axhline(0.05, color=c.PALETA[3], ls="--", lw=1, label="α = 0,05")
axes[0].set(xscale="log", yscale="log", title="Valor p (mediana de 200 réplicas)", xlabel="Estudiantes por grupo", ylabel="Valor p")
axes[0].legend()
axes[1].fill_between(pn["n_por_grupo"], pn["d_p05"], pn["d_p95"], color=c.PALETA[5], alpha=0.4, label="Percentiles 5–95")
axes[1].plot(pn["n_por_grupo"], pn["d_mediana"], "o-", color=c.PALETA[0], label="Mediana")
axes[1].axhline(0.2, color="#999999", ls=":", lw=1); axes[1].text(tam[-1], 0.21, "efecto pequeño", fontsize=7, ha="right", color="#555555")
axes[1].set(xscale="log", title="d de Cohen", xlabel="Estudiantes por grupo", ylabel="d (hombres − mujeres)")
axes[1].legend(fontsize=8)
fig.suptitle("Diferencia por género: el valor p depende de n, el tamaño del efecto no", fontweight="bold")
viz.guardar(fig, "fig_14_p_vs_n")
pn.round(4)

# %% [markdown]
# ## 6. Visualización interactiva de los resultados (Plotly)
# Medias por estrato con su intervalo de confianza; al pasar el cursor se ven n, desviación
# y el porcentaje en el cuartil superior. Se guarda en HTML (interactivo) y PNG (para el PDF).

# %%
r = d3.groupby("fami_estratovivienda", observed=True)[Y].agg(media="mean", sd="std", n="count").reset_index()
r["ic"] = 1.96 * r["sd"] / np.sqrt(r["n"])
r["alto"] = pct5[1].reindex(r["fami_estratovivienda"]).to_numpy()
figp = go.Figure()
figp.add_trace(go.Scatter(x=r["fami_estratovivienda"].astype(str), y=r["media"], mode="markers+lines",
                          error_y={"type": "data", "array": r["ic"], "visible": True}, marker={"size": 11, "color": c.PALETA[0]},
                          customdata=r[["n", "sd", "alto"]],
                          hovertemplate="<b>%{x}</b><br>Media: %{y:.1f}<br>n: %{customdata[0]:,}<br>Desv. est.: %{customdata[1]:.1f}<br>Cuartil superior: %{customdata[2]:.1f} %<extra></extra>"))
figp.add_hline(y=df[Y].mean(), line_dash="dash", annotation_text=f"Media nacional {df[Y].mean():.0f}")
titulo = f"Puntaje global medio por estrato (IC 95 %) · ANOVA F = {F:,.0f}, η² = {eta2:.3f}".replace(",", " ").replace(".", ",")
figp.update_layout(template="plotly_white", title=titulo, font={"size": 20}, title_font_size=21,
                   xaxis_title=viz.etiqueta("fami_estratovivienda"), yaxis_title=viz.etiqueta(Y), width=900, height=520)
figp.write_html(c.FIG_DIR / "fig_15_anova_interactivo.html", include_plotlyjs="cdn")
figp.write_image(c.FIG_DIR / "fig_15_anova_interactivo.png", scale=2)

# %%
resumen = pd.concat([
    tt[["variable", "diferencia", "p_t", "d_Cohen", "magnitud"]].assign(prueba="t de Welch").rename(columns={"p_t": "p", "d_Cohen": "efecto"}),
    pd.DataFrame([{"variable": "fami_estratovivienda", "diferencia": np.nan, "p": p_F, "efecto": eta2, "magnitud": stats.interpretar_eta(eta2), "prueba": "ANOVA (η²)"}]),
    chi_t.assign(variable=chi_t["variable_1"] + " × " + chi_t["variable_2"], prueba="χ² (V de Cramér)", diferencia=np.nan).rename(columns={"V_Cramer": "efecto"})[["variable", "diferencia", "p", "efecto", "magnitud", "prueba"]],
], ignore_index=True)
resumen.to_csv(c.TAB_DIR / "hip_resumen.csv", index=False, encoding="utf-8")
resumen.round(4)
