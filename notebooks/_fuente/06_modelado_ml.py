# %% [markdown]
# # 06 · Modelado con aprendizaje automático
#
# Dos problemas sobre el mismo conjunto de predictores (contexto socioeconómico y del colegio):
# - **Regresión:** predecir el puntaje global (0–500).
# - **Clasificación:** predecir si el estudiante queda en el cuartil superior (282 puntos o más).
#
# Modelos: regresión lineal regularizada (ElasticNet), Random Forest y XGBoost para regresión;
# regresión logística, Random Forest y XGBoost para clasificación. Los hiperparámetros se buscan
# con `RandomizedSearchCV` y validación cruzada de 5 pliegues, sobre una submuestra de entrenamiento. El preprocesamiento va dentro del
# `Pipeline`, de modo que en cada pliegue se ajusta solo con la parte de entrenamiento.
#
# El conjunto de **prueba** se usa una sola vez, al final, con los modelos ya elegidos.

# %%
import json
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from src import config as c
from src import features, viz
from src.evaluate import metricas_clasificacion, metricas_regresion

warnings.filterwarnings("ignore")
viz.aplicar_estilo()
proc = c.ROOT / "data" / "processed"
train, val, test = (pd.read_parquet(proc / f"{n}.parquet") for n in ("train", "val", "test"))
P = features.PREDICTORES
Xtr, Xva, Xte = train[P], val[P], test[P]
ytr, yva, yte = train["punt_global"], val["punt_global"], test["punt_global"]
ctr, cva, cte = train[c.TARGET_CLF], val[c.TARGET_CLF], test[c.TARGET_CLF]

# La búsqueda de hiperparámetros usa una submuestra aleatoria de entrenamiento: con 353 000 filas,
# cinco pliegues por cada combinación serían horas de cómputo sin cambiar la elección.
# El modelo elegido se reentrena después con TODO el entrenamiento.
N_BUSQUEDA = 40_000
# El equipo de trabajo no soporta muchos procesos en paralelo: la búsqueda va en serie
# y los modelos de árboles usan dos hilos.
N_HILOS = 2
sub = train.sample(N_BUSQUEDA, random_state=c.SEED).index
Xs, ys, cs = Xtr.loc[sub], ytr.loc[sub], ctr.loc[sub]
print(f"train {len(train):,} | val {len(val):,} | test {len(test):,} | submuestra de búsqueda {N_BUSQUEDA:,} | predictores {len(P)}")

def tubo(modelo, tarea="regresion"):
    return Pipeline([("pre", features.construir_preprocesador(tarea)), ("modelo", modelo)])

# Cada búsqueda se guarda al terminar: si la ejecución se interrumpe, al relanzar no se repite.
CACHE = proc / "cache_busqueda.json"

def buscar(clave, tuberia, espacio, n_iter, cv, scoring, X, y):
    clave = f"{clave}|n={len(X)}|iter={n_iter}"
    guardado = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    if clave not in guardado:
        t0 = time.perf_counter()
        b = RandomizedSearchCV(tuberia, espacio, n_iter=n_iter, cv=cv, scoring=scoring, random_state=c.SEED, n_jobs=1, refit=False).fit(X, y)
        i = b.best_index_
        params = {k: (v.item() if hasattr(v, "item") else v) for k, v in b.best_params_.items()}
        params = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in params.items()}
        guardado[clave] = {"params": params, "media": float(b.cv_results_["mean_test_score"][i]), "desv": float(b.cv_results_["std_test_score"][i]),
                           "segundos": time.perf_counter() - t0}
        CACHE.write_text(json.dumps(guardado, ensure_ascii=False, indent=1), encoding="utf-8")
    return guardado[clave]

# %% [markdown]
# ## 1. Regresión

# %%
kf = KFold(5, shuffle=True, random_state=c.SEED)
ESPACIOS_REG = {
    "ElasticNet": (ElasticNet(max_iter=5000, random_state=c.SEED),
                   {"modelo__alpha": st.loguniform(1e-4, 1.0), "modelo__l1_ratio": st.uniform(0.0, 1.0)}, 10),
    "Random Forest": (RandomForestRegressor(n_estimators=120, n_jobs=N_HILOS, random_state=c.SEED),
                      {"modelo__max_depth": [8, 12, 16, None], "modelo__min_samples_leaf": [5, 20, 50, 100], "modelo__max_features": [0.3, 0.5, 0.8]}, 5),
    "XGBoost": (XGBRegressor(tree_method="hist", n_jobs=N_HILOS, random_state=c.SEED),
                {"modelo__n_estimators": [200, 400, 700], "modelo__learning_rate": [0.03, 0.06, 0.1], "modelo__max_depth": [3, 4, 6, 8],
                 "modelo__subsample": [0.7, 0.9], "modelo__colsample_bytree": [0.6, 0.8, 1.0], "modelo__min_child_weight": [1, 10, 50],
                 "modelo__reg_lambda": [1, 5, 20]}, 10),
}
busq_reg, filas = {}, []
for nombre, (est, espacio, n_iter) in ESPACIOS_REG.items():
    r = buscar(f"reg|{nombre}", tubo(est), espacio, n_iter, kf, "neg_root_mean_squared_error", Xs, ys)
    busq_reg[nombre] = r
    filas.append({"modelo": nombre, "combinaciones": n_iter, "RMSE_CV_media": -r["media"], "RMSE_CV_desv": r["desv"],
                  "mejores_hiperparametros": {k.replace("modelo__", ""): v for k, v in r["params"].items()}, "segundos_busqueda": r["segundos"]})
    print(f"{nombre}: RMSE CV = {filas[-1]['RMSE_CV_media']:.2f} ± {filas[-1]['RMSE_CV_desv']:.2f} | {filas[-1]['mejores_hiperparametros']} | {r['segundos']:.0f} s")
cv_reg = pd.DataFrame(filas)
cv_reg.to_csv(c.TAB_DIR / "ml_cv_regresion.csv", index=False, encoding="utf-8")

# %%
# Reentrenamiento con todo el entrenamiento y evaluación en train y validación.
modelos_reg = {"Lineal (línea base)": tubo(LinearRegression())}
for nombre, (est, *_ ) in ESPACIOS_REG.items():
    modelos_reg[nombre] = tubo(est).set_params(**busq_reg[nombre]["params"])
res_reg, pred_reg = [], {}
for nombre, m in modelos_reg.items():
    t0 = time.perf_counter()
    m.fit(Xtr, ytr)
    seg = time.perf_counter() - t0
    pred_reg[nombre] = {"train": m.predict(Xtr), "val": m.predict(Xva)}
    for parte, y in (("train", ytr), ("val", yva)):
        res_reg.append({"modelo": nombre, "conjunto": parte, **metricas_regresion(y, pred_reg[nombre][parte]), "segundos_ajuste": seg})
res_reg = pd.DataFrame(res_reg)
res_reg.to_csv(c.TAB_DIR / "ml_regresion_train_val.csv", index=False, encoding="utf-8")
res_reg.pivot(index="modelo", columns="conjunto", values=["R2", "RMSE", "MAE"]).round(4)

# %% [markdown]
# ## 2. Clasificación
# El desbalance (25 % de positivos) se atiende con `class_weight="balanced"` en logística y Random
# Forest, y con `scale_pos_weight` en XGBoost. La métrica de selección es el AUC-ROC, que no depende
# del umbral.

# %%
skf = StratifiedKFold(5, shuffle=True, random_state=c.SEED)
peso = float((ctr == 0).sum() / (ctr == 1).sum())
ESPACIOS_CLF = {
    "Logística": (LogisticRegression(class_weight="balanced", max_iter=2000), {"modelo__C": st.loguniform(1e-3, 10)}, 6),
    "Random Forest": (RandomForestClassifier(n_estimators=120, class_weight="balanced", n_jobs=N_HILOS, random_state=c.SEED),
                      {"modelo__max_depth": [8, 12, 16, None], "modelo__min_samples_leaf": [5, 20, 50, 100], "modelo__max_features": [0.3, 0.5, 0.8]}, 5),
    "XGBoost": (XGBClassifier(tree_method="hist", n_jobs=N_HILOS, random_state=c.SEED, scale_pos_weight=peso, eval_metric="logloss"),
                {"modelo__n_estimators": [200, 400, 700], "modelo__learning_rate": [0.03, 0.06, 0.1], "modelo__max_depth": [3, 4, 6, 8],
                 "modelo__subsample": [0.7, 0.9], "modelo__colsample_bytree": [0.6, 0.8, 1.0], "modelo__min_child_weight": [1, 10, 50],
                 "modelo__reg_lambda": [1, 5, 20]}, 10),
}
busq_clf, filas = {}, []
for nombre, (est, espacio, n_iter) in ESPACIOS_CLF.items():
    r = buscar(f"clf|{nombre}", tubo(est, "clasificacion"), espacio, n_iter, skf, "roc_auc", Xs, cs)
    busq_clf[nombre] = r
    filas.append({"modelo": nombre, "combinaciones": n_iter, "AUC_CV_media": r["media"], "AUC_CV_desv": r["desv"],
                  "mejores_hiperparametros": {k.replace("modelo__", ""): v for k, v in r["params"].items()}, "segundos_busqueda": r["segundos"]})
    print(f"{nombre}: AUC CV = {filas[-1]['AUC_CV_media']:.4f} ± {filas[-1]['AUC_CV_desv']:.4f} | {filas[-1]['mejores_hiperparametros']} | {r['segundos']:.0f} s")
cv_clf = pd.DataFrame(filas)
cv_clf.to_csv(c.TAB_DIR / "ml_cv_clasificacion.csv", index=False, encoding="utf-8")

# %%
modelos_clf = {nombre: tubo(est, "clasificacion").set_params(**busq_clf[nombre]["params"]) for nombre, (est, *_ ) in ESPACIOS_CLF.items()}
res_clf, prob_clf = [], {}
for nombre, m in modelos_clf.items():
    m.fit(Xtr, ctr)
    prob_clf[nombre] = {"train": m.predict_proba(Xtr)[:, 1], "val": m.predict_proba(Xva)[:, 1]}
    for parte, y in (("train", ctr), ("val", cva)):
        res_clf.append({"modelo": nombre, "conjunto": parte, **metricas_clasificacion(y, prob_clf[nombre][parte])})
res_clf = pd.DataFrame(res_clf)
res_clf.to_csv(c.TAB_DIR / "ml_clasificacion_train_val.csv", index=False, encoding="utf-8")
res_clf.pivot(index="modelo", columns="conjunto", values=["AUC_ROC", "F1", "Recall", "Precision", "Accuracy"]).round(4)

# %% [markdown]
# ## 3. Sobreajuste: entrenamiento frente a validación

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
for ax, tabla, metrica, titulo in ((axes[0], res_reg, "R2", "Regresión: R²"), (axes[1], res_clf, "AUC_ROC", "Clasificación: AUC-ROC")):
    t = tabla.pivot(index="modelo", columns="conjunto", values=metrica).loc[list(dict.fromkeys(tabla["modelo"]))]
    x = np.arange(len(t))
    ax.bar(x - 0.2, t["train"], 0.4, color=c.PALETA[5], label="Entrenamiento")
    ax.bar(x + 0.2, t["val"], 0.4, color=c.PALETA[0], label="Validación")
    for i, (a, b) in enumerate(zip(t["train"], t["val"])):
        ax.text(i - 0.2, a + 0.005, f"{a:.3f}".replace(".", ","), ha="center", fontsize=7)
        ax.text(i + 0.2, b + 0.005, f"{b:.3f}".replace(".", ","), ha="center", fontsize=7)
    ax.set(title=titulo, xticks=x, ylim=(0, t.to_numpy().max() * 1.15))
    ax.set_xticklabels(t.index, fontsize=8)
    ax.legend(fontsize=8, loc="upper left")
viz.guardar(fig, "fig_30_sobreajuste")

# %% [markdown]
# ## 4. Curvas ROC y precisión-sensibilidad, y matrices de confusión (validación)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for (nombre, pr), color in zip(prob_clf.items(), (c.PALETA[7], c.PALETA[2], c.PALETA[3])):
    fpr, tpr, _ = roc_curve(cva, pr["val"])
    pre, rec, _ = precision_recall_curve(cva, pr["val"])
    m = metricas_clasificacion(cva, pr["val"])
    axes[0].plot(fpr, tpr, color=color, label=f"{nombre} (AUC = {m['AUC_ROC']:.3f})".replace(".", ","))
    axes[1].plot(rec, pre, color=color, label=f"{nombre} (AP = {m['AUC_PR']:.3f})".replace(".", ","))
axes[0].plot([0, 1], [0, 1], color="#999999", ls="--", lw=1)
axes[0].set(title="Curva ROC", xlabel="1 − especificidad", ylabel="Sensibilidad"); axes[0].legend(fontsize=8)
axes[1].axhline(cva.mean(), color="#999999", ls="--", lw=1, label=f"Azar ({cva.mean():.2f})".replace(".", ","))
axes[1].set(title="Curva precisión-sensibilidad", xlabel="Sensibilidad", ylabel="Precisión"); axes[1].legend(fontsize=8)
viz.guardar(fig, "fig_31_roc_pr")

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
matrices = {}
for ax, (nombre, pr) in zip(axes, prob_clf.items()):
    cm = confusion_matrix(cva, (pr["val"] >= 0.5).astype(int))
    matrices[nombre] = {"VN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "VP": cm[1, 1]}
    ConfusionMatrixDisplay(cm, display_labels=["Resto", "Cuartil superior"]).plot(ax=ax, cmap="Blues", colorbar=False, values_format=",d")
    ax.set(title=nombre, xlabel="Predicho", ylabel="Real"); ax.grid(False)
pd.DataFrame(matrices).T.to_csv(c.TAB_DIR / "ml_matrices_confusion.csv", encoding="utf-8")
viz.guardar(fig, "fig_32_matrices_confusion")

# %% [markdown]
# ## 5. Análisis de residuos y de errores del mejor modelo de regresión

# %%
mejor_reg = res_reg[res_reg["conjunto"] == "val"].sort_values("RMSE").iloc[0]["modelo"]
pv = pred_reg[mejor_reg]["val"]
resid = yva.to_numpy() - pv
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
hb = axes[0].hexbin(pv, yva, gridsize=45, cmap="viridis", bins="log", mincnt=1)
axes[0].plot([150, 380], [150, 380], color="white", ls="--", lw=1)
axes[0].set(title=f"{mejor_reg}: observado vs. predicho", xlabel="Puntaje predicho", ylabel="Puntaje observado")
axes[1].hist(resid, bins=80, color=c.PALETA[5])
axes[1].set(title="Distribución de los residuos", xlabel="Residuo (observado − predicho)", ylabel="Estudiantes")
dec = pd.qcut(yva, 10, labels=False)
sesgo = pd.Series(resid).groupby(dec.to_numpy()).mean()
axes[2].bar(sesgo.index + 1, sesgo.values, color=[c.PALETA[3] if v < 0 else c.PALETA[2] for v in sesgo.values])
axes[2].axhline(0, color="black", lw=0.8)
axes[2].set(title="Sesgo por decil del puntaje real", xlabel="Decil del puntaje observado", ylabel="Residuo medio")
viz.guardar(fig, "fig_33_residuos_mejor_modelo")

va_e = val.assign(error_abs=np.abs(resid), residuo=resid)
errores = pd.concat({
    "Estrato": va_e.groupby("fami_estratovivienda")["error_abs"].agg(MAE="mean", n="count"),
    "Naturaleza": va_e.groupby("cole_naturaleza")["error_abs"].agg(MAE="mean", n="count"),
    "Zona": va_e.groupby("cole_area_ubicacion")["error_abs"].agg(MAE="mean", n="count"),
    "Extraedad": va_e.groupby("extraedad")["error_abs"].agg(MAE="mean", n="count"),
})
errores.to_csv(c.TAB_DIR / "ml_errores_por_grupo.csv", encoding="utf-8")
pd.Series({"mejor_modelo_regresion": mejor_reg, "sesgo_decil_1": sesgo.iloc[0], "sesgo_decil_10": sesgo.iloc[-1], "asimetria_residuos": st.skew(resid),
           "pred_p01": np.quantile(pv, 0.01), "pred_p99": np.quantile(pv, 0.99), "desv_pred": pv.std(), "desv_obs": yva.std()}).to_csv(
    c.TAB_DIR / "ml_residuos_resumen.csv", header=["valor"], encoding="utf-8")
errores.round(2)

# %% [markdown]
# ## 6. Interpretabilidad: SHAP e importancia de variables
# SHAP reparte la predicción de cada estudiante entre los predictores; el promedio de los valores
# absolutos mide la importancia global. Se contrasta con la importancia por permutación, que mide
# cuánto empeora el error al desordenar cada variable original en validación.

# %%
NOMBRES = {"estu_inse_individual": "Índice INSE", "edad": "Edad", "indice_tic": "Índice TIC", "educ_max_padres": "Educación máx. de los padres",
           "fami_estratovivienda": "Estrato", "fami_educacionpadre": "Educación del padre", "fami_educacionmadre": "Educación de la madre",
           "fami_numlibros": "Libros en el hogar", "fami_personashogar": "Personas en el hogar", "estu_dedicacionlecturadiaria": "Lectura diaria",
           "estu_horassemanatrabaja": "Horas de trabajo", "cole_depto_ubicacion": "Departamento", "extraedad": "Extraedad", "inse_faltante": "INSE faltante",
           "estu_genero": "Género", "fami_tieneinternet": "Internet", "fami_tienecomputador": "Computador", "cole_naturaleza": "Naturaleza",
           "cole_area_ubicacion": "Zona", "cole_jornada": "Jornada", "cole_caracter": "Carácter", "cole_calendario": "Calendario", "cole_bilingue": "Bilingüe"}

def nombre_legible(col: str) -> str:
    for base, etq in sorted(NOMBRES.items(), key=lambda kv: -len(kv[0])):
        if col == base:
            return etq
        if col.startswith(base + "_"):
            return f"{etq}: {col[len(base) + 1:].capitalize()}"
    return col

xgb_reg = modelos_reg["XGBoost"]
pre, booster = xgb_reg.named_steps["pre"], xgb_reg.named_steps["modelo"]
muestra = Xva.sample(5000, random_state=c.SEED)
Z = pd.DataFrame(pre.transform(muestra), columns=[nombre_legible(n) for n in pre.get_feature_names_out()])
valores = shap.TreeExplainer(booster).shap_values(Z)
imp_shap = pd.Series(np.abs(valores).mean(axis=0), index=Z.columns).sort_values(ascending=False)
imp_shap.to_csv(c.TAB_DIR / "ml_shap_importancia.csv", header=["shap_medio_abs"], encoding="utf-8")

# SHAP ajusta la figura con tight_layout, que choca con el constrained_layout del estilo del proyecto:
# se desactiva solo para esta figura y se guarda sin la nota de fuente automática.
with plt.rc_context({"figure.constrained_layout.use": False}):
    shap.summary_plot(valores, Z, max_display=14, show=False, plot_size=(8, 4.6))
    fig = plt.gcf()
    plt.xlabel("Valor SHAP (puntos que suma o resta a la predicción)")
    plt.title("XGBoost: contribución de cada predictor (5 000 estudiantes de validación)", fontsize=10, fontweight="bold")
    fig.savefig(c.FIG_DIR / "fig_34_shap_resumen.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
imp_shap.head(12).round(3)

# %%
perm = permutation_importance(xgb_reg, Xva.iloc[:20000], yva.iloc[:20000], scoring="neg_root_mean_squared_error", n_repeats=3, random_state=c.SEED, n_jobs=1)
imp_perm = pd.DataFrame({"aumento_RMSE": perm.importances_mean, "desv": perm.importances_std}, index=[NOMBRES[p] for p in P]).sort_values("aumento_RMSE", ascending=False)
imp_perm.to_csv(c.TAB_DIR / "ml_importancia_permutacion.csv", encoding="utf-8")
# SHAP agregado por variable original (se suman las columnas one-hot de una misma variable).
shap_var = imp_shap.groupby(lambda n: n.split(":")[0]).sum().sort_values(ascending=False)
shap_var.to_csv(c.TAB_DIR / "ml_shap_por_variable.csv", header=["shap_medio_abs"], encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
g = shap_var.head(12).iloc[::-1]
axes[0].barh(g.index, g.values, color=c.PALETA[0]); axes[0].set(title="SHAP medio absoluto por variable", xlabel="Puntos")
g = imp_perm.head(12).iloc[::-1]
axes[1].barh(g.index, g["aumento_RMSE"], xerr=g["desv"], color=c.PALETA[2]); axes[1].set(title="Importancia por permutación", xlabel="Aumento del RMSE (puntos)")
for ax in axes: ax.tick_params(axis="y", labelsize=8)
viz.guardar(fig, "fig_35_importancias")
imp_perm.head(10).round(3)

# %% [markdown]
# ## 7. Evaluación final en el conjunto de prueba
# Primera y única vez que se usa el conjunto de prueba. Los modelos ya están elegidos y entrenados.

# %%
final_reg = pd.DataFrame({n: metricas_regresion(yte, m.predict(Xte)) for n, m in modelos_reg.items()}).T
final_reg.to_csv(c.TAB_DIR / "ml_regresion_test.csv", encoding="utf-8")
final_clf = pd.DataFrame({n: metricas_clasificacion(cte, m.predict_proba(Xte)[:, 1]) for n, m in modelos_clf.items()}).T
final_clf.to_csv(c.TAB_DIR / "ml_clasificacion_test.csv", encoding="utf-8")
print(final_reg.round(4)); final_clf.round(4)

# %%
# Modelos serializados para el tablero (predicción en vivo) y predicciones de validación para el cuaderno 07.
destino = c.ROOT / "dashboard" / "modelos"
destino.mkdir(parents=True, exist_ok=True)
joblib.dump(modelos_reg["XGBoost"], destino / "regresor_xgboost.joblib", compress=3)
joblib.dump(modelos_clf["XGBoost"], destino / "clasificador_xgboost.joblib", compress=3)
pd.DataFrame({"y": yva.to_numpy(), **{f"pred_{k}": v["val"] for k, v in pred_reg.items()}}).to_parquet(proc / "pred_val_regresion.parquet", index=False)
print("Modelos guardados en dashboard/modelos/")
