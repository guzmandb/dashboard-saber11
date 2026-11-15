# %% [markdown]
# # 02 · Preprocesamiento, ingeniería de variables y modelo base
#
# Orden de trabajo, pensado para que no haya fuga de datos:
# 1. **Partición** en entrenamiento, validación y prueba (70/15/15) antes de cualquier ajuste.
# 2. Diagnóstico del **mecanismo de faltantes** (MCAR, MAR, MNAR) y estrategia de imputación.
# 3. Tratamiento de **atípicos**.
# 4. **Variables derivadas** con sentido de dominio.
# 5. **Codificación y escalado** dentro de un `ColumnTransformer` ajustado solo con train.
# 6. **Desbalance** de clases.
# 7. **Modelos base** de regresión y clasificación.
#
# Targets: `punt_global` (regresión) y `alto_desempeno` (clasificación: cuartil superior).
# Los cinco puntajes por área no se usan como predictores porque componen el puntaje global.

# %%
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as st
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import config as c
from src import data, features, stats, viz

warnings.filterwarnings("ignore")
viz.aplicar_estilo()
df = data.cargar_limpio()
print(f"n = {len(df):,}")

# %% [markdown]
# ## 1. Partición 70 / 15 / 15
# Se estratifica por quintiles del puntaje global para que los tres conjuntos tengan la misma
# distribución del target. El umbral de `alto_desempeno` (percentil 75) se calcula **solo con
# entrenamiento** y se aplica después a validación y prueba.

# %%
estrato_split = pd.qcut(df["punt_global"], 5, labels=False, duplicates="drop")
train, resto = train_test_split(df, test_size=0.30, random_state=c.SEED, stratify=estrato_split)
val, test = train_test_split(resto, test_size=0.50, random_state=c.SEED, stratify=estrato_split.loc[resto.index])

UMBRAL = float(train["punt_global"].quantile(0.75))
train, _ = data.agregar_target_clf(train, UMBRAL)
val, _ = data.agregar_target_clf(val, UMBRAL)
test, _ = data.agregar_target_clf(test, UMBRAL)

particion = pd.DataFrame({nombre: {"n": len(d), "%": 100 * len(d) / len(df), "media_global": d["punt_global"].mean(),
                                   "desv_global": d["punt_global"].std(), "alto_desempeno_%": 100 * d[c.TARGET_CLF].mean()}
                          for nombre, d in [("train", train), ("val", val), ("test", test)]}).T
particion.to_csv(c.TAB_DIR / "pre_particion.csv", encoding="utf-8")
print(f"Umbral de alto desempeño (P75 de train): {UMBRAL:.0f}")
particion.round(2)

# %% [markdown]
# ## 2. Mecanismo de los datos faltantes
# Todo el diagnóstico usa únicamente el conjunto de entrenamiento.
#
# - **MCAR** (completamente al azar): la ausencia no depende de nada; quienes no responden son
#   una muestra aleatoria. Implicaría que el puntaje medio es igual con y sin dato.
# - **MAR** (al azar condicional): la ausencia depende de variables observadas.
# - **MNAR** (no al azar): depende del propio valor que falta; no se puede comprobar con los datos.
#
# Se aplican dos comprobaciones: (a) comparar el puntaje global de quienes tienen y no tienen el
# dato (t de Welch y d de Cohen), que es un contraste directo de MCAR; (b) ver si la ausencia
# se puede **predecir** con variables siempre observadas (naturaleza, zona, departamento, género
# y edad): un AUC claramente superior a 0,5 es evidencia de MAR.

# %%
con_falt = [v for v in ["cole_bilingue", "fami_educacionpadre", "fami_educacionmadre", "fami_tienecomputador", "estu_horassemanatrabaja",
                        "fami_estratovivienda", "fami_personashogar", "estu_dedicacionlecturadiaria", "cole_caracter",
                        "estu_inse_individual", "fami_tieneinternet", "fami_numlibros"] if train[v].isna().any()]
Xobs = pd.get_dummies(train[["cole_naturaleza", "cole_area_ubicacion", "cole_depto_ubicacion", "estu_genero"]].astype(str), drop_first=True).astype(float)
Xobs["edad"] = train["edad"].fillna(train["edad"].median()).clip(upper=30)
sub = train.sample(60000, random_state=c.SEED).index
filas = []
for v in con_falt:
    falta = train[v].isna()
    a, b = train.loc[falta, "punt_global"], train.loc[~falta, "punt_global"]
    t, p = st.ttest_ind(a, b, equal_var=False)
    m = LogisticRegression(max_iter=300).fit(Xobs.loc[sub[:40000]], falta.loc[sub[:40000]])
    auc = roc_auc_score(falta.loc[sub[40000:]], m.predict_proba(Xobs.loc[sub[40000:]])[:, 1])
    filas.append({"variable": v, "faltantes_%": 100 * falta.mean(), "global_con_dato": b.mean(), "global_sin_dato": a.mean(),
                  "diferencia": a.mean() - b.mean(), "t_Welch": t, "p": p, "d_Cohen": stats.cohen_d(a, b), "AUC_prediccion_ausencia": auc})
mec = pd.DataFrame(filas).sort_values("faltantes_%", ascending=False)
mec.to_csv(c.TAB_DIR / "pre_mecanismo_faltantes.csv", index=False, encoding="utf-8")

# ¿Faltan juntas? Si la ausencia fuera independiente entre variables, casi nadie tendría varias a la vez.
cuestionario = ["fami_estratovivienda", "fami_tieneinternet", "fami_tienecomputador", "fami_numlibros", "fami_personashogar", "estu_dedicacionlecturadiaria", "estu_horassemanatrabaja"]
n_falt = train[cuestionario].isna().sum(axis=1)
patron = pd.Series({"sin_ningun_faltante_%": 100 * (n_falt == 0).mean(), "con_1_o_2_%": 100 * n_falt.between(1, 2).mean(),
                    "con_5_o_mas_%": 100 * (n_falt >= 5).mean(), "todas_las_7_%": 100 * (n_falt == 7).mean(),
                    "corr_media_entre_indicadores": train[cuestionario].isna().astype(int).corr().where(~np.eye(7, dtype=bool)).stack().mean()})
patron.to_csv(c.TAB_DIR / "pre_patron_faltantes.csv", header=["valor"], encoding="utf-8")
print(patron.round(3).to_string()); mec.round(3)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
o = mec.sort_values("diferencia")
axes[0].barh([viz.etiqueta(v) for v in o["variable"]], o["diferencia"], color=[c.PALETA[3] if x < 0 else c.PALETA[0] for x in o["diferencia"]])
axes[0].axvline(0, color="black", lw=0.8)
axes[0].set(title="Puntaje global: sin dato − con dato", xlabel="Diferencia de medias (puntos)")
axes[0].tick_params(axis="y", labelsize=7)
sns.heatmap(train[cuestionario].isna().astype(int).corr(), cmap="viridis", vmin=0, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 6},
            xticklabels=[viz.etiqueta(v)[:14] for v in cuestionario], yticklabels=[viz.etiqueta(v)[:14] for v in cuestionario], ax=axes[1])
axes[1].set(title="Correlación entre indicadores de ausencia")
axes[1].tick_params(labelsize=6)
viz.guardar(fig, "fig_24_mecanismo_faltantes")

# %% [markdown]
# **Lectura y decisión.** Se interpreta en el informe con los valores de la tabla. La regla es:
# si el puntaje difiere entre quienes tienen y no tienen el dato, o la ausencia es predecible,
# se descarta MCAR y eliminar filas sesgaría la muestra. En ese caso:
#
# - **Categóricas:** la ausencia pasa a ser una categoría propia, "Sin dato". No se inventa un
#   valor y el modelo puede aprender lo que la no respuesta señala.
# - **INSE** (numérica): mediana de entrenamiento más un indicador binario `inse_faltante`.
# - **Edad:** mediana de entrenamiento (0,06 % de faltantes).
#
# No se usa `KNNImputer` ni `IterativeImputer`: con 350 000 filas el primero es inviable y ambos
# suponen MAR sin que la mejora compense, porque las variables que faltan lo hacen en bloque.

# %% [markdown]
# ## 3. Valores atípicos
# - **Edad:** asimetría de 8 por quienes validan el bachillerato de adultos. Son datos reales,
#   no errores, así que no se eliminan: se **winsoriza** a los percentiles 1 y 99 de entrenamiento
#   y se crea la variable `extraedad`, que conserva la información.
# - **INSE:** 0,9 % fuera de 1,5·IQR, sin valores imposibles. No se trata.
# - **Puntaje global (target):** no se modifica. Recortar el target cambiaría lo que se predice.

# %%
lim = train["edad"].quantile([0.01, 0.99])
at = stats.outliers(train, ["edad", "estu_inse_individual", "punt_global"])
at["P1_train"] = [lim.iloc[0], np.nan, np.nan]; at["P99_train"] = [lim.iloc[1], np.nan, np.nan]
at["asimetria_antes"] = [st.skew(train[v].dropna()) for v in at.index]
at.loc["edad", "asimetria_despues"] = st.skew(train["edad"].dropna().clip(*lim))
at.to_csv(c.TAB_DIR / "pre_outliers.csv", encoding="utf-8")
at.round(3)

# %% [markdown]
# ## 4. Variables derivadas
# | Variable | Definición | Justificación de dominio |
# |---|---|---|
# | `indice_tic` | internet + computador (0–2) | En 2020 la clase fue remota: importa cuántos recursos digitales hay, no cada uno por separado |
# | `educ_max_padres` | Máximo nivel educativo entre padre y madre (0–9) | El capital educativo del hogar lo fija quien más estudió; reduce los faltantes de cada variable |
# | `extraedad` | 1 si tiene más de 18 años | Marca rezago escolar o validación de adultos, un perfil distinto del bachiller regular |
# | `inse_faltante` | 1 si no tiene índice socioeconómico | La no respuesta es informativa |

# %%
train_d, val_d, test_d = features.derivar(train), features.derivar(val), features.derivar(test)
deriv = train_d.groupby("indice_tic")["punt_global"].agg(["mean", "count"]).rename_axis("indice_tic")
deriv2 = train_d.groupby("extraedad")["punt_global"].agg(["mean", "count"])
corr_d = train_d[["indice_tic", "educ_max_padres", "extraedad", "inse_faltante", "punt_global"]].corr(method="spearman")["punt_global"].drop("punt_global")
pd.concat({"indice_tic": deriv, "extraedad": deriv2}).to_csv(c.TAB_DIR / "pre_derivadas.csv", encoding="utf-8")
corr_d.to_csv(c.TAB_DIR / "pre_derivadas_corr.csv", header=["rho_spearman_con_global"], encoding="utf-8")
print(deriv.round(1)); print(deriv2.round(1)); corr_d.round(3)

# %% [markdown]
# ## 5. Codificación y escalado sin fuga
# Todo vive en un `ColumnTransformer` (`src/features.py`) que se ajusta con `fit` sobre train y
# solo se aplica (`transform`) a validación y prueba:
#
# - **Ordinal** para las variables con orden natural (estrato, educación, libros, personas, lectura, trabajo).
# - **One-Hot** para las nominales de pocas categorías.
# - **Target encoding** para el departamento (33 categorías), con validación cruzada interna de 5
#   pliegues para que ninguna fila se codifique con su propio target.
# - **StandardScaler** en numéricas y ordinales.

# %%
Xtr, Xva, Xte = train_d[features.PREDICTORES], val_d[features.PREDICTORES], test_d[features.PREDICTORES]
ytr, yva, yte = train_d["punt_global"], val_d["punt_global"], test_d["punt_global"]
ctr, cva, cte = train_d[c.TARGET_CLF], val_d[c.TARGET_CLF], test_d[c.TARGET_CLF]

pre = features.construir_preprocesador("regresion").fit(Xtr, ytr)
Ztr = pre.transform(Xtr)
nombres = pre.get_feature_names_out()
print(f"Predictores originales: {len(features.PREDICTORES)} → columnas tras codificar: {Ztr.shape[1]}")
print(f"Faltantes tras el preprocesador: {int(np.isnan(Ztr).sum())}")
print("Media y desviación de las numéricas escaladas en train:", Ztr[:, :4].mean(axis=0).round(3), Ztr[:, :4].std(axis=0).round(3))
print("Las mismas columnas en validación (no son 0 y 1 exactos porque el escalador no las vio):", pre.transform(Xva)[:, :4].mean(axis=0).round(3))

# %% [markdown]
# ## 6. Desbalance de clases
# El cuartil superior es, por construcción, cerca del 25 % de los casos: un desbalance moderado
# (3 a 1), no severo. Se usa `class_weight="balanced"`, que pondera la función de pérdida, en
# lugar de SMOTE: con 350 000 filas reales no hace falta sintetizar observaciones, y SMOTE
# interpola entre categorías codificadas que no tienen puntos intermedios con sentido.

# %%
print(ctr.value_counts(normalize=True).mul(100).round(2).to_string())

# %% [markdown]
# ## 7. Modelos base
# Fijan el piso que cualquier modelo posterior debe superar. Se evalúan en **validación**; el
# conjunto de prueba no se toca hasta el final del proyecto.

# %%
def metricas_reg(y, pred):
    return {"R2": r2_score(y, pred), "RMSE": float(np.sqrt(mean_squared_error(y, pred))), "MAE": mean_absolute_error(y, pred)}

def metricas_clf(y, prob, umbral=0.5):
    pred = (prob >= umbral).astype(int)
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred),
            "F1": f1_score(y, pred), "AUC_ROC": roc_auc_score(y, prob) if len(np.unique(prob)) > 1 else 0.5}

# Regresión: media constante, lineal simple (solo INSE) y lineal con todos los predictores.
med = float(Xtr["estu_inse_individual"].median())
inse_tr, inse_va = Xtr[["estu_inse_individual"]].fillna(med), Xva[["estu_inse_individual"]].fillna(med)
base_reg = {
    "Dummy (media de train)": DummyRegressor().fit(Xtr, ytr).predict(Xva),
    "Lineal simple (INSE)": LinearRegression().fit(inse_tr, ytr).predict(inse_va),
}
lin = Pipeline([("pre", features.construir_preprocesador("regresion")), ("modelo", LinearRegression())]).fit(Xtr, ytr)
base_reg["Lineal múltiple (todos los predictores)"] = lin.predict(Xva)
res_reg = pd.DataFrame({k: metricas_reg(yva, v) for k, v in base_reg.items()}).T
res_reg.loc["Lineal múltiple (todos los predictores)", "R2_train"] = r2_score(ytr, lin.predict(Xtr))
res_reg.to_csv(c.TAB_DIR / "pre_baseline_regresion.csv", encoding="utf-8")
res_reg.round(4)

# %%
base_clf = {
    "Dummy (clase mayoritaria)": DummyClassifier(strategy="most_frequent").fit(Xtr, ctr).predict_proba(Xva)[:, 1],
    "Dummy (estratificado)": DummyClassifier(strategy="stratified", random_state=c.SEED).fit(Xtr, ctr).predict_proba(Xva)[:, 1],
}
log_simple = LogisticRegression(class_weight="balanced", max_iter=500).fit(inse_tr, ctr)
base_clf["Logística simple (INSE)"] = log_simple.predict_proba(inse_va)[:, 1]
log = Pipeline([("pre", features.construir_preprocesador("clasificacion")), ("modelo", LogisticRegression(class_weight="balanced", max_iter=1000))]).fit(Xtr, ctr)
base_clf["Logística (todos los predictores)"] = log.predict_proba(Xva)[:, 1]
res_clf = pd.DataFrame({k: metricas_clf(cva, v) for k, v in base_clf.items()}).T
res_clf.to_csv(c.TAB_DIR / "pre_baseline_clasificacion.csv", encoding="utf-8")
res_clf.round(4)

# %%
# Se guardan las particiones ya derivadas para que los cuadernos siguientes usen exactamente las mismas filas.
proc = c.ROOT / "data" / "processed"
for nombre, d in [("train", train_d), ("val", val_d), ("test", test_d)]:
    d.assign(id_fila=d.index).to_parquet(proc / f"{nombre}.parquet", index=False)
joblib.dump({"umbral_alto_desempeno": UMBRAL, "predictores": features.PREDICTORES}, proc / "meta_particion.joblib")
print("Particiones guardadas en data/processed/")
