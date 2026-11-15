# %% [markdown]
# # 03 · Regresión — Saber 11° 2020-2
#
# Pregunta: ¿cuánto del puntaje global se asocia con el contexto socioeconómico y del colegio,
# y qué aporta cada factor cuando los demás se mantienen constantes?
#
# 1. Regresión lineal **simple** (puntaje ~ índice socioeconómico) con `statsmodels`.
# 2. Regresión lineal **múltiple** con `statsmodels`: coeficientes, IC, VIF y diagnóstico de supuestos.
# 3. El mismo modelo en **scikit-learn**, para comprobar que ambas bibliotecas coinciden y evaluar en validación.
# 4. Regresión **polinómica** y **logística**.
#
# Todo se ajusta con el conjunto de **entrenamiento** de `02_preprocesamiento.ipynb` y se evalúa en
# **validación**. Los puntajes por área no entran como predictores (componen el target).

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from src import config as c
from src import viz

warnings.filterwarnings("ignore")
viz.aplicar_estilo()
proc = c.ROOT / "data" / "processed"
train, val = pd.read_parquet(proc / "train.parquet"), pd.read_parquet(proc / "val.parquet")

# Variables interpretables para el modelo explicativo. Se trabaja con casos completos de train:
# aquí interesa leer coeficientes, no maximizar la predicción (eso es la Fase 4).
# El código del colegio no es un predictor: solo sirve para agrupar los errores estándar.
colegios = pd.read_csv(c.RAW_CSV, usecols=["cole_cod_dane_establecimiento"], dtype=str)["cole_cod_dane_establecimiento"]
for d in (train, val):
    d["colegio"] = colegios.iloc[d["id_fila"].to_numpy()].to_numpy()

VARS = ["colegio", "punt_global", c.TARGET_CLF, "estu_inse_individual", "educ_max_padres", "indice_tic", "extraedad",
        "estu_genero", "cole_naturaleza", "cole_area_ubicacion", "cole_jornada"]
CATS = ["estu_genero", "cole_naturaleza", "cole_area_ubicacion", "cole_jornada"]
# En las particiones los faltantes de las categóricas ya están marcados como "Sin dato": aquí se excluyen.
tr = train.loc[(train[CATS] != "Sin dato").all(axis=1), VARS].dropna().copy()
va = val.loc[(val[CATS] != "Sin dato").all(axis=1), VARS].dropna().copy()
for d in (tr, va):
    d["jornada"] = d["cole_jornada"].where(d["cole_jornada"].isin(["MAÑANA", "TARDE", "UNICA", "COMPLETA"]), "NOCHE O SABATINA")
print(f"train: {len(tr):,} de {len(train):,} ({100 * len(tr) / len(train):.1f} % con datos completos) | val: {len(va):,}")

def evaluar(y, pred):
    return {"R2": r2_score(y, pred), "RMSE": float(np.sqrt(mean_squared_error(y, pred))), "MAE": mean_absolute_error(y, pred)}

# %% [markdown]
# ## 1. Regresión lineal simple

# %%
simple = smf.ols("punt_global ~ estu_inse_individual", data=tr).fit()
ic = simple.conf_int()
t_simple = pd.DataFrame({"coef": simple.params, "ee": simple.bse, "t": simple.tvalues, "IC95_inf": ic[0], "IC95_sup": ic[1]})
t_simple.to_csv(c.TAB_DIR / "reg_simple_coef.csv", encoding="utf-8")
r_pearson = float(np.corrcoef(tr["estu_inse_individual"], tr["punt_global"])[0, 1])
print(f"R² = {simple.rsquared:.4f} | r de Pearson = {r_pearson:.4f} | F = {simple.fvalue:,.0f} | error estándar residual = {np.sqrt(simple.mse_resid):.2f}")
t_simple.round(3)

# %% [markdown]
# ## 2. Regresión polinómica
# ¿Es recta la relación con el nivel socioeconómico? Se comparan grados 1, 2 y 3 en validación.

# %%
Xtr, Xva = tr[["estu_inse_individual"]].to_numpy(), va[["estu_inse_individual"]].to_numpy()
poli, curvas = [], {}
rejilla = np.linspace(tr["estu_inse_individual"].quantile(0.001), tr["estu_inse_individual"].quantile(0.999), 200).reshape(-1, 1)
for grado in (1, 2, 3):
    m = make_pipeline(PolynomialFeatures(grado, include_bias=False), LinearRegression()).fit(Xtr, tr["punt_global"])
    poli.append({"grado": grado, "R2_train": r2_score(tr["punt_global"], m.predict(Xtr)), **{f"{k}_val": v for k, v in evaluar(va["punt_global"], m.predict(Xva)).items()}})
    curvas[grado] = m.predict(rejilla)
poli = pd.DataFrame(poli)
poli.to_csv(c.TAB_DIR / "reg_polinomica.csv", index=False, encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
hb = axes[0].hexbin(tr["estu_inse_individual"], tr["punt_global"], gridsize=45, cmap="Greys", bins="log", mincnt=1)
for grado, color in zip((1, 2, 3), (c.PALETA[0], c.PALETA[1], c.PALETA[3])):
    axes[0].plot(rejilla, curvas[grado], color=color, lw=2, label=f"Grado {grado} (R² val = {poli.loc[grado - 1, 'R2_val']:.3f})".replace(".", ","))
axes[0].set(title="Ajuste lineal y polinómico", xlabel=viz.etiqueta("estu_inse_individual"), ylabel=viz.etiqueta("punt_global"), ylim=(100, 420))
axes[0].legend(fontsize=8)
# Media observada por tramos de INSE: muestra la forma real de la relación sin suponer nada.
tramos = tr.groupby(pd.cut(tr["estu_inse_individual"], np.arange(20, 85, 2.5)), observed=True)["punt_global"].agg(["mean", "count"])
centros = [i.mid for i in tramos.index]
axes[1].scatter(centros, tramos["mean"], s=np.sqrt(tramos["count"]) / 2, color=c.PALETA[2], label="Media observada por tramo (área ∝ n)")
axes[1].plot(rejilla, curvas[1], color=c.PALETA[0], lw=1.5, label="Recta")
axes[1].plot(rejilla, curvas[3], color=c.PALETA[3], lw=1.5, label="Cúbica")
axes[1].set(title="Media del puntaje por tramo de INSE", xlabel=viz.etiqueta("estu_inse_individual"), ylabel="Puntaje global medio")
axes[1].legend(fontsize=8)
viz.guardar(fig, "fig_25_regresion_simple_polinomica")
poli.round(4)

# %% [markdown]
# ## 3. Regresión lineal múltiple (statsmodels)
# Categorías de referencia: mujer, colegio oficial, zona rural, jornada de la mañana.

# %%
FORMULA = ("punt_global ~ estu_inse_individual + educ_max_padres + indice_tic + extraedad"
           " + C(estu_genero, Treatment('F')) + C(cole_naturaleza, Treatment('OFICIAL'))"
           " + C(cole_area_ubicacion, Treatment('RURAL')) + C(jornada, Treatment('MAÑANA'))")
ols = smf.ols(FORMULA, data=tr).fit()
ETQ = {"Intercept": "Intercepto", "estu_inse_individual": "INSE (por punto)", "educ_max_padres": "Educación de los padres (por nivel)",
       "indice_tic": "Índice TIC (por recurso)", "extraedad": "Extraedad (más de 18 años)", "C(estu_genero, Treatment('F'))[T.M]": "Hombre",
       "C(cole_naturaleza, Treatment('OFICIAL'))[T.NO OFICIAL]": "Colegio no oficial", "C(cole_area_ubicacion, Treatment('RURAL'))[T.URBANO]": "Zona urbana",
       "C(jornada, Treatment('MAÑANA'))[T.COMPLETA]": "Jornada completa", "C(jornada, Treatment('MAÑANA'))[T.NOCHE O SABATINA]": "Jornada nocturna o sabatina",
       "C(jornada, Treatment('MAÑANA'))[T.TARDE]": "Jornada de la tarde", "C(jornada, Treatment('MAÑANA'))[T.UNICA]": "Jornada única"}
rob = ols.get_robustcov_results(cov_type="HC3")
# Los estudiantes de un mismo colegio no son independientes: errores agrupados por colegio.
grupos = pd.factorize(tr["colegio"])[0]
clu = ols.get_robustcov_results(cov_type="cluster", groups=grupos)
ic = ols.conf_int()
coef = pd.DataFrame({"termino": [ETQ[k] for k in ols.params.index], "coef": ols.params.to_numpy(), "ee": ols.bse.to_numpy(), "t": ols.tvalues.to_numpy(),
                     "IC95_inf": ic[0].to_numpy(), "IC95_sup": ic[1].to_numpy(), "ee_robusto_HC3": rob.bse,
                     "IC95_inf_HC3": rob.conf_int()[:, 0], "IC95_sup_HC3": rob.conf_int()[:, 1],
                     "ee_cluster": clu.bse, "IC95_inf_cluster": clu.conf_int()[:, 0], "IC95_sup_cluster": clu.conf_int()[:, 1],
                     "t_cluster": clu.tvalues, "p_cluster": clu.pvalues})
# Coeficientes estandarizados: permiten comparar el peso de predictores con unidades distintas.
X = pd.DataFrame(ols.model.exog, columns=ols.model.exog_names)
coef["beta_estandarizado"] = (ols.params.to_numpy() * X.std().to_numpy() / tr["punt_global"].std())
coef.loc[0, "beta_estandarizado"] = np.nan
coef["VIF"] = [np.nan] + [variance_inflation_factor(X.to_numpy(), i) for i in range(1, X.shape[1])]
coef.to_csv(c.TAB_DIR / "reg_multiple_coef.csv", index=False, encoding="utf-8")
ajuste = pd.Series({"n": int(ols.nobs), "R2": ols.rsquared, "R2_ajustado": ols.rsquared_adj, "F": ols.fvalue, "gl_modelo": ols.df_model, "gl_residual": ols.df_resid,
                    "error_estandar_residual": np.sqrt(ols.mse_resid), "AIC": ols.aic, "n_colegios": int(tr["colegio"].nunique()),
                    "razon_ee_cluster_sobre_clasico_max": float((clu.bse / ols.bse.to_numpy()).max()), "razon_ee_cluster_sobre_clasico_min": float((clu.bse / ols.bse.to_numpy()).min()), "numero_de_condicion": ols.condition_number})
ajuste.to_csv(c.TAB_DIR / "reg_multiple_ajuste.csv", header=["valor"], encoding="utf-8")
print(ajuste.round(4).to_string()); coef.round(3)

# %% [markdown]
# ## 4. Diagnóstico de supuestos

# %%
res, aj = ols.resid, ols.fittedvalues
est = ols.get_influence().resid_studentized_internal
bp = het_breuschpagan(res, ols.model.exog)
jb = jarque_bera(res)
diag = pd.Series({"Breusch_Pagan_LM": bp[0], "Breusch_Pagan_p": bp[1], "Jarque_Bera": jb[0], "Jarque_Bera_p": jb[1], "asimetria_residuos": jb[2],
                  "curtosis_residuos": jb[3], "Durbin_Watson": durbin_watson(res), "VIF_maximo": coef["VIF"].max(),
                  "residuos_|z|>3_%": 100 * np.mean(np.abs(est) > 3), "razon_ee_HC3_sobre_clasico_max": (coef["ee_robusto_HC3"] / coef["ee"]).max(),
                  "desv_residuos_decil_bajo": res[aj <= aj.quantile(0.1)].std(), "desv_residuos_decil_alto": res[aj >= aj.quantile(0.9)].std()})
diag.to_csv(c.TAB_DIR / "reg_diagnostico.csv", header=["valor"], encoding="utf-8")

fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
axes[0].hexbin(aj, res, gridsize=40, cmap="viridis", bins="log", mincnt=1)
axes[0].axhline(0, color="white", lw=1)
axes[0].set(title="Residuos vs. ajustados", xlabel="Valor ajustado", ylabel="Residuo")
m = np.random.default_rng(c.SEED).choice(len(est), 5000, replace=False)
(osm, osr), (pend, inter, _) = st.probplot(est[m], dist="norm")
axes[1].scatter(osm, osr, s=3, alpha=0.4, color=c.PALETA[0]); axes[1].plot(osm, pend * osm + inter, color=c.PALETA[3], lw=1)
axes[1].set(title="Q-Q de residuos (muestra de 5 000)", xlabel="Cuantiles teóricos", ylabel="Residuo estandarizado")
dec = pd.qcut(aj, 10, labels=False)
axes[2].plot(aj.groupby(dec).mean(), res.groupby(dec).std(), "o-", color=c.PALETA[0])
axes[2].set(title="Dispersión por decil del ajustado", xlabel="Valor ajustado medio", ylabel="Desv. est. de los residuos", ylim=(0, None))
axes[3].hist(est, bins=80, density=True, color=c.PALETA[5])
z = np.linspace(-4, 4, 200); axes[3].plot(z, st.norm.pdf(z), color=c.PALETA[3], lw=1.2, label="N(0, 1)")
axes[3].set(title="Residuos estandarizados", xlabel="Residuo estandarizado", ylabel="Densidad", xlim=(-4.5, 4.5)); axes[3].legend(fontsize=8)
viz.guardar(fig, "fig_26_diagnostico_residuos")
diag.round(4)

# %%
g = coef.iloc[1:].sort_values("coef")
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
axes[0].errorbar(g["coef"], g["termino"], xerr=[g["coef"] - g["IC95_inf_cluster"], g["IC95_sup_cluster"] - g["coef"]], fmt="o", color=c.PALETA[0], capsize=3)
axes[0].axvline(0, color="black", lw=0.8)
axes[0].set(title="Coeficientes (IC 95 % agrupado por colegio)", xlabel="Puntos del puntaje global")
axes[1].barh(g["termino"], g["beta_estandarizado"], color=[c.PALETA[3] if v < 0 else c.PALETA[2] for v in g["beta_estandarizado"]])
axes[1].axvline(0, color="black", lw=0.8)
axes[1].set(title="Coeficientes estandarizados (β)", xlabel="Desviaciones estándar del puntaje")
axes[0].tick_params(axis="y", labelsize=8)
viz.guardar(fig, "fig_27_coeficientes")

# %% [markdown]
# ## 5. El mismo modelo en scikit-learn y evaluación en validación
# `LinearRegression` resuelve los mismos mínimos cuadrados, así que debe dar los mismos coeficientes.
# La diferencia es de propósito: `statsmodels` entrega inferencia (errores estándar, IC, pruebas);
# scikit-learn entrega predicción y se integra en pipelines.

# %%
def disenar(d: pd.DataFrame) -> pd.DataFrame:
    """Matriz de diseño construida a mano, con las mismas categorías de referencia que la fórmula."""
    return pd.DataFrame({
        "INSE (por punto)": d["estu_inse_individual"], "Educación de los padres (por nivel)": d["educ_max_padres"],
        "Índice TIC (por recurso)": d["indice_tic"], "Extraedad (más de 18 años)": d["extraedad"],
        "Hombre": (d["estu_genero"] == "M").astype(float), "Colegio no oficial": (d["cole_naturaleza"] == "NO OFICIAL").astype(float),
        "Zona urbana": (d["cole_area_ubicacion"] == "URBANO").astype(float), "Jornada completa": (d["jornada"] == "COMPLETA").astype(float),
        "Jornada nocturna o sabatina": (d["jornada"] == "NOCHE O SABATINA").astype(float),
        "Jornada de la tarde": (d["jornada"] == "TARDE").astype(float), "Jornada única": (d["jornada"] == "UNICA").astype(float)})

Xtr_m, Xva_m = disenar(tr), disenar(va)
sk = LinearRegression().fit(Xtr_m, tr["punt_global"])
coef_sm = coef.set_index("termino")["coef"]
dif_max = float(max(abs(sk.intercept_ - coef_sm["Intercepto"]), np.max(np.abs(sk.coef_ - coef_sm[Xtr_m.columns].to_numpy()))))
pred_va = ols.predict(va)
pred_va_sk = sk.predict(Xva_m)
base = pd.read_csv(c.TAB_DIR / "pre_baseline_regresion.csv", index_col=0)
comp = pd.DataFrame({
    "Media de train (Dummy)": {"R2_train": 0.0, **{f"{k}_val": v for k, v in evaluar(va["punt_global"], np.full(len(va), tr["punt_global"].mean())).items()}},
    "Lineal simple (INSE)": {"R2_train": simple.rsquared, **{f"{k}_val": v for k, v in evaluar(va["punt_global"], simple.predict(va)).items()}},
    "Polinómica grado 3 (INSE)": {"R2_train": poli.loc[2, "R2_train"], "R2_val": poli.loc[2, "R2_val"], "RMSE_val": poli.loc[2, "RMSE_val"], "MAE_val": poli.loc[2, "MAE_val"]},
    "Lineal múltiple, 11 términos (statsmodels)": {"R2_train": ols.rsquared, **{f"{k}_val": v for k, v in evaluar(va["punt_global"], pred_va).items()}},
    "Lineal múltiple, 11 términos (scikit-learn)": {"R2_train": sk.score(Xtr_m, tr["punt_global"]), **{f"{k}_val": v for k, v in evaluar(va["punt_global"], pred_va_sk).items()}},
    "Lineal con todos los predictores (pipeline, 02)": {"R2_train": base.loc["Lineal múltiple (todos los predictores)", "R2_train"],
                                                         **{f"{k}_val": base.loc["Lineal múltiple (todos los predictores)", k] for k in ("R2", "RMSE", "MAE")}},
}).T
comp["n_val"] = [len(va)] * 5 + [len(val)]
comp.to_csv(c.TAB_DIR / "reg_comparacion_modelos.csv", encoding="utf-8")
print(f"Diferencia máxima entre coeficientes de statsmodels y scikit-learn: {dif_max:.2e}")
pd.Series({"dif_max_coef_statsmodels_sklearn": dif_max}).to_csv(c.TAB_DIR / "reg_statsmodels_vs_sklearn.csv", header=["valor"], encoding="utf-8")
comp.round(4)

# %% [markdown]
# ## 6. Regresión logística: probabilidad de estar en el cuartil superior

# %%
logit = smf.logit(FORMULA.replace("punt_global", c.TARGET_CLF), data=tr).fit(disp=0)
ic_l = logit.conf_int()
orr = pd.DataFrame({"termino": [ETQ[k] for k in logit.params.index], "coef_logit": logit.params.to_numpy(), "OR": np.exp(logit.params.to_numpy()),
                    "OR_IC95_inf": np.exp(ic_l[0].to_numpy()), "OR_IC95_sup": np.exp(ic_l[1].to_numpy()), "z": logit.tvalues.to_numpy()})
orr.to_csv(c.TAB_DIR / "reg_logistica_or.csv", index=False, encoding="utf-8")
prob = logit.predict(va)
# El umbral se elige con entrenamiento (índice de Youden) y se evalúa en validación, para no ser optimista.
fpr_tr, tpr_tr, um_tr = roc_curve(tr[c.TARGET_CLF], logit.predict(tr))
UMBRAL_J = float(um_tr[np.argmax(tpr_tr - fpr_tr)])
fpr, tpr, um = roc_curve(va[c.TARGET_CLF], prob)
j = int(np.argmin(np.abs(um - UMBRAL_J)))
pred = (prob >= UMBRAL_J).astype(int)
vp, fp = int(((pred == 1) & (va[c.TARGET_CLF] == 1)).sum()), int(((pred == 1) & (va[c.TARGET_CLF] == 0)).sum())
fn, vn = int(((pred == 0) & (va[c.TARGET_CLF] == 1)).sum()), int(((pred == 0) & (va[c.TARGET_CLF] == 0)).sum())
log_aj = pd.Series({"n_train": int(logit.nobs), "pseudo_R2_McFadden": logit.prsquared, "AUC_val": roc_auc_score(va[c.TARGET_CLF], prob), "umbral_Youden": UMBRAL_J,
                    "sensibilidad": vp / (vp + fn), "especificidad": vn / (vn + fp), "precision": vp / (vp + fp), "exactitud": (vp + vn) / len(va),
                    "VP": vp, "FP": fp, "FN": fn, "VN": vn})
log_aj.to_csv(c.TAB_DIR / "reg_logistica_ajuste.csv", header=["valor"], encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
g = orr.iloc[1:].sort_values("OR")
axes[0].errorbar(g["OR"], g["termino"], xerr=[g["OR"] - g["OR_IC95_inf"], g["OR_IC95_sup"] - g["OR"]], fmt="o", color=c.PALETA[0], capsize=3)
axes[0].axvline(1, color="black", lw=0.8)
axes[0].set(xscale="log", title="Razones de odds (IC 95 %)", xlabel="OR (escala logarítmica)")
axes[0].tick_params(axis="y", labelsize=8)
axes[1].plot(fpr, tpr, color=c.PALETA[0], label=f"AUC = {log_aj['AUC_val']:.3f}".replace(".", ","))
axes[1].plot([0, 1], [0, 1], color="#999999", ls="--", lw=1)
axes[1].scatter(fpr[j], tpr[j], color=c.PALETA[3], zorder=3, label="Umbral elegido en entrenamiento")
axes[1].set(title="Curva ROC en validación", xlabel="1 − especificidad", ylabel="Sensibilidad"); axes[1].legend(fontsize=8)
viz.guardar(fig, "fig_28_logistica")
print(log_aj.round(4).to_string()); orr.round(3)

# %% [markdown]
# ## 7. Visualización interactiva (Plotly)
# Puntaje observado frente a predicho en validación, como mapa de densidad: con 73 000 puntos una
# dispersión se satura. Se exporta en HTML (interactivo) y PNG.

# %%
H, xe, ye = np.histogram2d(pred_va, va["punt_global"], bins=[60, 60], range=[[150, 340], [100, 420]])
figp = go.Figure(go.Heatmap(z=H.T, x=(xe[:-1] + xe[1:]) / 2, y=(ye[:-1] + ye[1:]) / 2, colorscale="Viridis", colorbar={"title": "Estudiantes"},
                            hovertemplate="Predicho: %{x:.0f}<br>Observado: %{y:.0f}<br>Estudiantes: %{z}<extra></extra>"))
figp.add_trace(go.Scatter(x=[150, 340], y=[150, 340], mode="lines", line={"color": "white", "dash": "dash"}, name="Predicción perfecta"))
r2v = evaluar(va["punt_global"], pred_va)
figp.update_layout(template="plotly_white", width=900, height=620, font={"size": 18}, title_font_size=19, showlegend=False,
                   title=f"Observado frente a predicho · R² = {r2v['R2']:.3f}".replace(".", ","),
                   xaxis_title="Puntaje global predicho", yaxis_title="Puntaje global observado")
figp.write_html(c.FIG_DIR / "fig_29_observado_predicho.html", include_plotlyjs="cdn")
figp.write_image(c.FIG_DIR / "fig_29_observado_predicho.png", scale=2)
# Rango de las predicciones: un modelo con R² bajo comprime sus predicciones hacia la media.
pd.Series({"pred_p01": float(np.quantile(pred_va, 0.01)), "pred_p99": float(np.quantile(pred_va, 0.99)), "obs_p01": float(va["punt_global"].quantile(0.01)),
           "obs_p99": float(va["punt_global"].quantile(0.99)), "desv_pred": float(np.std(pred_va)), "desv_obs": float(va["punt_global"].std())}).to_csv(
    c.TAB_DIR / "reg_rango_predicciones.csv", header=["valor"], encoding="utf-8")
