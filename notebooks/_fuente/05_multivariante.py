# %% [markdown]
# # 05 · Análisis multivariante — PCA y clustering
#
# Los cinco puntajes por área se resumen con componentes principales y se segmentan con
# K-Means y clustering jerárquico. El objetivo es responder la tercera pregunta secundaria:
# ¿hay perfiles de desempeño que vayan más allá de "puntaje alto o bajo"?
#
# Las variables socioeconómicas **no** entran al clustering: se usan después, para describir
# quiénes componen cada grupo.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy import stats as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config as c
from src import data, viz

warnings.filterwarnings("ignore")
rng = np.random.default_rng(c.SEED)
viz.aplicar_estilo()

AREAS = c.PUNTAJES_AREA
CORTO = ["Lectura", "Matemáticas", "Naturales", "Sociales", "Inglés"]
df = data.cargar_limpio().dropna(subset=AREAS).reset_index(drop=True)
escala = StandardScaler().fit(df[AREAS])
X = escala.transform(df[AREAS])
print(f"n = {len(df):,} estudiantes con los cinco puntajes")

# %% [markdown]
# ## 1. ¿Tiene sentido reducir dimensiones?
# La prueba de esfericidad de Bartlett contrasta si la matriz de correlaciones es la identidad;
# el índice KMO mide cuánta varianza es común. Un KMO superior a 0,8 se considera meritorio.

# %%
R = np.corrcoef(X, rowvar=False)
n, p = X.shape
chi2_b = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R))
gl_b = p * (p - 1) // 2
Rinv = np.linalg.inv(R)
parcial = -Rinv / np.sqrt(np.outer(np.diag(Rinv), np.diag(Rinv)))
fuera = ~np.eye(p, dtype=bool)
kmo = (R[fuera] ** 2).sum() / ((R[fuera] ** 2).sum() + (parcial[fuera] ** 2).sum())
adecuacion = pd.Series({"chi2_Bartlett": chi2_b, "gl": gl_b, "p_Bartlett": st.chi2.sf(chi2_b, gl_b), "KMO": kmo,
                        "correlacion_minima": R[fuera].min(), "correlacion_maxima": R[fuera].max()})
adecuacion.to_csv(c.TAB_DIR / "mv_adecuacion.csv", header=["valor"], encoding="utf-8")
adecuacion.round(4)

# %% [markdown]
# ## 2. Análisis de componentes principales
# Los puntajes se estandarizan (media 0, desviación 1) para que ninguna área pese más por su escala.

# %%
pca = PCA(random_state=c.SEED).fit(X)
Z = pca.transform(X)
var = pd.DataFrame({"autovalor": pca.explained_variance_, "varianza_%": 100 * pca.explained_variance_ratio_,
                    "acumulada_%": 100 * pca.explained_variance_ratio_.cumsum()}, index=[f"CP{i+1}" for i in range(p)])
cargas = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_), index=CORTO, columns=var.index)
var.to_csv(c.TAB_DIR / "mv_pca_varianza.csv", encoding="utf-8")
cargas.to_csv(c.TAB_DIR / "mv_pca_cargas.csv", encoding="utf-8")
print(var.round(2)); cargas.round(3)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
axes[0].bar(var.index, var["varianza_%"], color=c.PALETA[0], label="Individual")
axes[0].plot(var.index, var["acumulada_%"], "o-", color=c.PALETA[3], label="Acumulada")
axes[0].axhline(100 / p, color="#999999", ls=":", lw=1)
for i, v in enumerate(var["varianza_%"]):
    axes[0].text(i, v + 1.5, f"{v:.1f} %", ha="center", fontsize=8)
axes[0].set(title="Gráfico de sedimentación", ylabel="Varianza explicada (%)", ylim=(0, 108))
axes[0].legend(loc="center right")

m = rng.choice(len(Z), 15000, replace=False)
axes[1].scatter(Z[m, 0], Z[m, 1], s=2, alpha=0.12, color="#888888")
esc = 3.2
for j, nombre in enumerate(CORTO):
    axes[1].arrow(0, 0, esc * cargas.iloc[j, 0], esc * cargas.iloc[j, 1], color=c.PALETA[3], width=0.02, head_width=0.15, length_includes_head=True)
    axes[1].text(esc * cargas.iloc[j, 0] * 1.08, esc * cargas.iloc[j, 1] * 1.12, nombre, fontsize=8, color=c.PALETA[3], fontweight="bold")
axes[1].set(title="Biplot (muestra de 15 000)", xlabel=f"CP1 ({var['varianza_%'].iloc[0]:.1f} %)", ylabel=f"CP2 ({var['varianza_%'].iloc[1]:.1f} %)")
viz.guardar(fig, "fig_16_pca_sedimentacion_biplot")

# %% [markdown]
# ## 3. K-Means: elección de k
# El codo se calcula con todos los registros. La silueta exige las distancias entre todos los
# pares, así que se evalúa sobre una submuestra aleatoria de 20 000 estudiantes.

# %%
idx = rng.choice(len(X), 20000, replace=False)
filas, modelos = [], {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=c.SEED).fit(X)
    modelos[k] = km
    filas.append({"k": k, "inercia": km.inertia_, "silueta": silhouette_score(X[idx], km.labels_[idx]),
                  "calinski_harabasz": calinski_harabasz_score(X[idx], km.labels_[idx]),
                  "davies_bouldin": davies_bouldin_score(X[idx], km.labels_[idx]), "cluster_menor_%": 100 * np.bincount(km.labels_).min() / len(X)})
km1 = KMeans(n_clusters=1, n_init=1, random_state=c.SEED).fit(X)
eleccion = pd.DataFrame(filas)
eleccion["caida_inercia_%"] = 100 * (1 - eleccion["inercia"] / np.r_[km1.inertia_, eleccion["inercia"].to_numpy()[:-1]])
eleccion.to_csv(c.TAB_DIR / "mv_kmeans_eleccion_k.csv", index=False, encoding="utf-8")
eleccion.round(3)

# %% [markdown]
# **Decisión: k = 3.** La silueta es máxima en k = 2 (0,40) y cae a 0,31 en k = 3. Con k = 2 la
# partición solo separa "por debajo" y "por encima" de la media, algo que el puntaje global ya dice.
# El codo de la inercia está en k = 3: pasar de 2 a 3 grupos reduce la inercia un 25 %, y de 3 a 4
# solo un 14 %. Se elige k = 3 porque añade un grupo interpretable con un costo moderado en
# cohesión. Una silueta de 0,31 indica estructura débil: los grupos son tramos de un continuo,
# no poblaciones separadas, y así se reportan.

# %%
K = 3
km = modelos[K]
# Se renumeran los clústeres de menor a mayor puntaje medio para que las etiquetas sean estables.
orden = np.argsort(km.cluster_centers_.mean(axis=1))
mapa = {viejo: nuevo for nuevo, viejo in enumerate(orden)}
df["cluster"] = pd.Series(km.labels_).map(mapa).to_numpy()
NOMBRES = {0: "C1 · Desempeño bajo", 1: "C2 · Desempeño medio", 2: "C3 · Desempeño alto"}
df["perfil"] = df["cluster"].map(NOMBRES)
df["CP1"], df["CP2"] = Z[:, 0], Z[:, 1]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
axes[0].plot(eleccion["k"], eleccion["inercia"] / 1e6, "o-", color=c.PALETA[0])
axes[0].axvline(K, color=c.PALETA[3], ls="--", lw=1)
axes[0].set(title="Método del codo", xlabel="Número de clústeres (k)", ylabel="Inercia (millones)")
axes[1].plot(eleccion["k"], eleccion["silueta"], "o-", color=c.PALETA[2])
axes[1].axvline(K, color=c.PALETA[3], ls="--", lw=1)
axes[1].set(title="Coeficiente de silueta medio", xlabel="Número de clústeres (k)", ylabel="Silueta")
sil = silhouette_samples(X[idx], df["cluster"].to_numpy()[idx])
y0 = 0
for g in range(K):
    s = np.sort(sil[df["cluster"].to_numpy()[idx] == g])
    axes[2].fill_betweenx(np.arange(y0, y0 + len(s)), 0, s, color=c.PALETA[g], label=NOMBRES[g][:2])
    y0 += len(s) + 200
axes[2].axvline(sil.mean(), color="black", ls="--", lw=1)
axes[2].set(title=f"Siluetas individuales (k = {K}, media = {sil.mean():.2f})", xlabel="Coeficiente de silueta", yticks=[])
axes[2].legend(fontsize=8)
viz.guardar(fig, "fig_17_codo_silueta")

# %% [markdown]
# ### Estabilidad de la partición
# Con una silueta baja cabe preguntarse si los grupos dependen de la inicialización. Se repite
# K-Means con cinco semillas distintas (una sola inicialización cada vez) y se mide el acuerdo
# con la partición de referencia mediante el índice de Rand ajustado sobre todos los registros.

# %%
from itertools import permutations

estab = []
for semilla in [1, 7, 21, 99, 2026]:
    alt = KMeans(n_clusters=K, n_init=1, random_state=semilla).fit(X)
    cruce = pd.crosstab(km.labels_, alt.labels_).to_numpy()
    coinciden = max(cruce[list(range(K)), list(perm)].sum() for perm in permutations(range(K)))
    estab.append({"semilla": semilla, "ARI_vs_referencia": adjusted_rand_score(km.labels_, alt.labels_),
                  "inercia": alt.inertia_, "cambian_de_cluster_%": 100 * (1 - coinciden / len(X))})
estab = pd.DataFrame(estab)
estab.to_csv(c.TAB_DIR / "mv_estabilidad.csv", index=False, encoding="utf-8")
estab.round(4)

# %% [markdown]
# ## 4. Perfiles de los clústeres

# %%
perfil = df.groupby("perfil")[AREAS + ["punt_global", "estu_inse_individual"]].mean()
perfil.insert(0, "n", df["perfil"].value_counts())
perfil.insert(1, "%", 100 * perfil["n"] / len(df))
comp = pd.DataFrame({
    "oficial_%": 100 * df.groupby("perfil")["cole_naturaleza"].apply(lambda s: (s == "OFICIAL").mean()),
    "rural_%": 100 * df.groupby("perfil")["cole_area_ubicacion"].apply(lambda s: (s == "RURAL").mean()),
    "sin_internet_%": 100 * df.groupby("perfil")["fami_tieneinternet"].apply(lambda s: (s == "No").sum() / s.notna().sum()),
    "estrato_1_o_sin_%": 100 * df.groupby("perfil")["fami_estratovivienda"].apply(lambda s: s.isin(["Sin Estrato", "Estrato 1"]).sum() / s.notna().sum()),
    "madre_profesional_%": 100 * df.groupby("perfil")["fami_educacionmadre"].apply(lambda s: s.isin(["Educación profesional completa", "Postgrado"]).sum() / s.notna().sum()),
})
perfil = perfil.join(comp)
perfil.to_csv(c.TAB_DIR / "mv_perfiles_cluster.csv", encoding="utf-8")
# Centroides en unidades z: muestran en qué área se aleja más cada grupo de la media nacional.
centros_z = pd.DataFrame(km.cluster_centers_[orden], index=[NOMBRES[i] for i in range(K)], columns=CORTO)
centros_z.to_csv(c.TAB_DIR / "mv_centroides_z.csv", encoding="utf-8")
perfil.round(1).T

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
sns.heatmap(centros_z, annot=True, fmt=".2f", cmap="RdBu_r", center=0, cbar_kws={"label": "Desviaciones estándar"}, ax=axes[0])
axes[0].set(title="Centroides estandarizados por área")
axes[0].tick_params(axis="y", rotation=0)
axes[0].tick_params(axis="x", rotation=30, labelsize=8)
comp_g =comp[["oficial_%", "rural_%", "sin_internet_%", "estrato_1_o_sin_%", "madre_profesional_%"]]
comp_g.columns = ["Colegio oficial", "Zona rural", "Sin internet", "Estrato 1 o sin estrato", "Madre profesional"]
comp_g.T.plot.barh(ax=axes[1], color=c.PALETA[:K], width=0.8)
axes[1].set(title="Composición socioeconómica de cada clúster", xlabel="Estudiantes del clúster (%)")
axes[1].legend(fontsize=7, title=None)
viz.guardar(fig, "fig_18_perfiles_cluster")

# %% [markdown]
# ## 5. Clustering jerárquico aglomerativo (Ward)
# El método de Ward necesita la matriz de distancias completa, inviable con medio millón de
# filas. Se aplica a una submuestra aleatoria de 3 000 estudiantes y se compara con K-Means
# mediante el índice de Rand ajustado (ARI): 1 es acuerdo total, 0 es acuerdo por azar.

# %%
sub = rng.choice(len(X), 3000, replace=False)
enlace = linkage(X[sub], method="ward")
jer = fcluster(enlace, t=K, criterion="maxclust")
ari = adjusted_rand_score(df["cluster"].to_numpy()[sub], jer)
sil_jer = silhouette_score(X[sub], jer)
sil_km_sub = silhouette_score(X[sub], df["cluster"].to_numpy()[sub])
comparacion = pd.DataFrame({"metodo": ["K-Means", "Jerárquico (Ward)"], "silueta_submuestra_3000": [sil_km_sub, sil_jer],
                            "ARI_frente_a_KMeans": [1.0, ari], "tamanos": [np.bincount(df["cluster"].to_numpy()[sub]).tolist(), np.bincount(jer)[1:].tolist()]})
comparacion.to_csv(c.TAB_DIR / "mv_kmeans_vs_jerarquico.csv", index=False, encoding="utf-8")

fig, ax = plt.subplots(figsize=(9, 3.8))
corte = enlace[-(K - 1), 2]
dendrogram(enlace, truncate_mode="lastp", p=40, color_threshold=corte, above_threshold_color="#777777", ax=ax, leaf_font_size=6)
ax.axhline(corte, color=c.PALETA[3], ls="--", lw=1, label=f"Corte en {K} clústeres")
ax.set(title="Dendrograma de Ward (submuestra de 3 000; últimas 40 fusiones)", xlabel="Grupos fusionados (tamaño entre paréntesis)", ylabel="Distancia de Ward")
ax.legend()
viz.guardar(fig, "fig_19_dendrograma")
comparacion.round(3)

# %% [markdown]
# ## 6. Clústeres en el espacio de componentes principales (Plotly)

# %%
vis = df.iloc[rng.choice(len(df), 8000, replace=False)].sort_values("cluster")
figp = px.scatter(vis, x="CP1", y="CP2", color="perfil", opacity=0.45, color_discrete_sequence=c.PALETA[:K], render_mode="webgl",
                  hover_data={"punt_global": True, "punt_ingles": True, "punt_matematicas": True, "cole_naturaleza": True, "CP1": ":.2f", "CP2": ":.2f"},
                  labels={"CP1": f"CP1 · desempeño general ({var['varianza_%'].iloc[0]:.1f} %)", "CP2": f"CP2 · inglés frente al resto ({var['varianza_%'].iloc[1]:.1f} %)", "perfil": "Clúster"},
                  title="Clústeres K-Means en el plano de las dos primeras componentes (muestra de 8 000)", template="plotly_white", width=900, height=560)
figp.update_traces(marker={"size": 4})
figp.update_layout(font={"size": 17}, title_font_size=19, legend={"orientation": "h", "y": -0.18})
figp.write_html(c.FIG_DIR / "fig_20_clusters_pca.html", include_plotlyjs="cdn")
figp.write_image(c.FIG_DIR / "fig_20_clusters_pca.png", scale=2)

# %% [markdown]
# ## 7. Enlace con la regresión: cuánto del desempeño general explica el contexto
# La primera componente resume el desempeño. Aquí se mide con mínimos cuadrados ordinarios qué
# parte del puntaje global se asocia con el contexto socioeconómico. Es un avance descriptivo:
# el diagnóstico completo de supuestos y la partición train/val/test están en `03_regresion.ipynb`.

# %%
import statsmodels.formula.api as smf

dr = df.dropna(subset=["estu_inse_individual", "fami_tieneinternet", "fami_tienecomputador"])
simple = smf.ols("punt_global ~ estu_inse_individual", data=dr).fit()
multiple = smf.ols("punt_global ~ estu_inse_individual + C(cole_naturaleza, Treatment('OFICIAL')) + C(cole_area_ubicacion, Treatment('RURAL'))"
                   " + C(fami_tieneinternet, Treatment('No')) + C(fami_tienecomputador, Treatment('No'))", data=dr).fit()
ETQ = ["Intercepto", "Colegio no oficial", "Zona urbana", "Tiene internet", "Tiene computador", "INSE (por punto)"]
ic = multiple.conf_int()
reg = pd.DataFrame({"termino": ETQ, "coef": multiple.params.to_numpy(), "ee": multiple.bse.to_numpy(), "IC95_inf": ic[0].to_numpy(), "IC95_sup": ic[1].to_numpy(), "t": multiple.tvalues.to_numpy()})
reg.to_csv(c.TAB_DIR / "mv_regresion_coef.csv", index=False, encoding="utf-8")
ajuste = pd.DataFrame({"modelo": ["Simple: INSE", "Múltiple: INSE + colegio + zona + internet + computador"], "n": [int(simple.nobs), int(multiple.nobs)],
                       "R2": [simple.rsquared, multiple.rsquared], "R2_ajustado": [simple.rsquared_adj, multiple.rsquared_adj],
                       "RMSE": [np.sqrt(simple.mse_resid), np.sqrt(multiple.mse_resid)], "pendiente_INSE": [simple.params["estu_inse_individual"], multiple.params["estu_inse_individual"]]})
ajuste.to_csv(c.TAB_DIR / "mv_regresion_ajuste.csv", index=False, encoding="utf-8")
r_cp1 = float(np.corrcoef(df["CP1"], df["punt_global"])[0, 1])
print(f"Correlación entre CP1 y puntaje global: {r_cp1:.4f}"); print(ajuste.round(4).to_string()); reg.round(3)

# %% [markdown]
# ## 8. ¿Difieren los clústeres en variables que no se usaron para formarlos?
# ANOVA del índice socioeconómico entre clústeres y χ² entre clúster y tenencia de internet.

# %%
from src import stats as est

gi = [g.dropna() for _, g in df.groupby("cluster")["estu_inse_individual"]]
F_inse, p_inse = st.f_oneway(*gi)
chi2_i, p_i, v_i, gl_i = est.v_cramer(pd.crosstab(df["perfil"], df["fami_tieneinternet"]))
chi2_n, p_n, v_n, gl_n = est.v_cramer(pd.crosstab(df["perfil"], df["cole_naturaleza"]))
validacion = pd.DataFrame([
    {"contraste": "INSE entre clústeres (ANOVA)", "estadistico": F_inse, "p": p_inse, "efecto": est.eta_cuadrado(gi), "medida": "eta2"},
    {"contraste": "Clúster × internet (χ²)", "estadistico": chi2_i, "p": p_i, "efecto": v_i, "medida": "V de Cramér"},
    {"contraste": "Clúster × naturaleza del colegio (χ²)", "estadistico": chi2_n, "p": p_n, "efecto": v_n, "medida": "V de Cramér"}])
validacion.to_csv(c.TAB_DIR / "mv_validacion_externa.csv", index=False, encoding="utf-8")
validacion.round(4)

# %%
# Insumo del dashboard: muestra estratificada con componentes y clúster asignados.
muestra = data.cargar_muestra().dropna(subset=AREAS).reset_index(drop=True)
Xm = escala.transform(muestra[AREAS])
Zm = pca.transform(Xm)
muestra["CP1"], muestra["CP2"] = Zm[:, 0], Zm[:, 1]
muestra["cluster"] = pd.Series(km.predict(Xm)).map(mapa).to_numpy()
muestra["perfil"] = muestra["cluster"].map(NOMBRES)
muestra.to_parquet(c.ROOT / "data" / "processed" / "muestra_pca_clusters.parquet", index=False)
resumen = pd.Series({"CP1_%": var["varianza_%"].iloc[0], "CP1_CP2_%": var["acumulada_%"].iloc[1], "KMO": kmo, "k": K,
                     "silueta_k": float(eleccion.set_index("k").loc[K, "silueta"]), "silueta_k2": float(eleccion.set_index("k").loc[2, "silueta"]),
                     "ARI_jerarquico": ari, "silueta_jerarquico": sil_jer, "r_CP1_global": r_cp1})
resumen.to_csv(c.TAB_DIR / "mv_resumen.csv", header=["valor"], encoding="utf-8")
resumen.round(3)
