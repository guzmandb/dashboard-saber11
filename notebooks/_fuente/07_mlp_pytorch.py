# %% [markdown]
# # 07 · Perceptrón multicapa (MLP) en PyTorch
#
# Se entrena una red neuronal densa para los dos problemas del proyecto —regresión del puntaje
# global y clasificación del cuartil superior— con los mismos predictores y la misma partición que
# los modelos de `06_modelado_ml.ipynb`, y se compara con ellos.
#
# - **Entrada:** las columnas que produce el preprocesador (`src/features.py`), ajustado solo con train.
# - **Arquitectura:** tres capas ocultas (128, 64, 32) con ReLU y *dropout* de 0,1.
# - **Entrenamiento:** AdamW, lotes de 1 024, *early stopping* sobre la pérdida de validación
#   (paciencia de 6 épocas) y restauración de los mejores pesos.
# - El conjunto de prueba se evalúa una sola vez, al final.

# %%
import copy
import random
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config as c
from src import features, viz
from src.evaluate import metricas_clasificacion, metricas_regresion

warnings.filterwarnings("ignore")
viz.aplicar_estilo()
random.seed(c.SEED); np.random.seed(c.SEED); torch.manual_seed(c.SEED)
torch.use_deterministic_algorithms(True)
torch.set_num_threads(2)  # el equipo no soporta carga alta sostenida en todos los núcleos

proc = c.ROOT / "data" / "processed"
train, val, test = (pd.read_parquet(proc / f"{n}.parquet") for n in ("train", "val", "test"))
P = features.PREDICTORES

def tensores(tarea: str):
    """Preprocesa con un transformador ajustado solo en train y devuelve tensores."""
    objetivo = "punt_global" if tarea == "regresion" else c.TARGET_CLF
    pre = features.construir_preprocesador(tarea).fit(train[P], train[objetivo])
    X = [torch.tensor(pre.transform(d[P]), dtype=torch.float32) for d in (train, val, test)]
    y = [d[objetivo].to_numpy(dtype="float32") for d in (train, val, test)]
    return X, y

class MLP(nn.Module):
    def __init__(self, n_entrada: int, ocultas=(128, 64, 32), dropout: float = 0.1) -> None:
        super().__init__()
        capas, anterior = [], n_entrada
        for n in ocultas:
            capas += [nn.Linear(anterior, n), nn.ReLU(), nn.Dropout(dropout)]
            anterior = n
        capas.append(nn.Linear(anterior, 1))
        self.red = nn.Sequential(*capas)

    def forward(self, x):
        return self.red(x).squeeze(1)

def entrenar(Xtr, ytr, Xva, yva, perdida, max_epocas=60, paciencia=6, lr=1e-3):
    modelo = MLP(Xtr.shape[1])
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(c.SEED)
    lotes = DataLoader(TensorDataset(Xtr, ytr), batch_size=1024, shuffle=True, generator=gen)
    historia, mejor, mejores_pesos, sin_mejora = [], float("inf"), None, 0
    for epoca in range(1, max_epocas + 1):
        modelo.train()
        suma = 0.0
        for xb, yb in lotes:
            opt.zero_grad()
            l = perdida(modelo(xb), yb)
            l.backward()
            opt.step()
            suma += l.item() * len(xb)
        modelo.eval()
        with torch.no_grad():
            # La pérdida de train se recalcula sin dropout para que sea comparable con la de validación.
            l_tr = perdida(modelo(Xtr), ytr).item()
            l_va = perdida(modelo(Xva), yva).item()
        historia.append({"epoca": epoca, "train_durante_la_epoca": suma / len(Xtr), "train": l_tr, "val": l_va})
        if l_va < mejor - 1e-5:
            mejor, mejores_pesos, sin_mejora = l_va, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    modelo.load_state_dict(mejores_pesos)
    modelo.eval()
    return modelo, pd.DataFrame(historia)

# %% [markdown]
# ## 1. Regresión
# El puntaje se estandariza con la media y la desviación de train para que la pérdida (error
# cuadrático medio) esté en una escala cómoda para el optimizador; las métricas se reportan en puntos.

# %%
X, y = tensores("regresion")
mu, sd = float(y[0].mean()), float(y[0].std())
yt = [torch.tensor((v - mu) / sd) for v in y]
t0 = time.perf_counter()
mlp_reg, hist_reg = entrenar(X[0], yt[0], X[1], yt[1], nn.MSELoss())
seg_reg = time.perf_counter() - t0
with torch.no_grad():
    pred = [mlp_reg(x).numpy() * sd + mu for x in X]
res_mlp_reg = pd.DataFrame({parte: metricas_regresion(y[i], pred[i]) for i, parte in enumerate(("train", "val", "test"))}).T
hist_reg.to_csv(c.TAB_DIR / "mlp_historia_regresion.csv", index=False, encoding="utf-8")
n_param = sum(p.numel() for p in mlp_reg.parameters())
print(f"Entradas: {X[0].shape[1]} | parámetros: {n_param:,} | épocas: {len(hist_reg)} (mejor: {int(hist_reg['val'].idxmin()) + 1}) | {seg_reg:.0f} s")
res_mlp_reg.round(4)

# %% [markdown]
# ## 2. Clasificación
# Pérdida de entropía cruzada binaria con `pos_weight` igual a la razón entre negativos y positivos
# de train, el equivalente a `class_weight="balanced"`.

# %%
Xc, yc = tensores("clasificacion")
yct = [torch.tensor(v) for v in yc]
peso = torch.tensor(float((yc[0] == 0).sum() / (yc[0] == 1).sum()))
t0 = time.perf_counter()
mlp_clf, hist_clf = entrenar(Xc[0], yct[0], Xc[1], yct[1], nn.BCEWithLogitsLoss(pos_weight=peso))
seg_clf = time.perf_counter() - t0
with torch.no_grad():
    prob = [torch.sigmoid(mlp_clf(x)).numpy() for x in Xc]
res_mlp_clf = pd.DataFrame({parte: metricas_clasificacion(yc[i], prob[i]) for i, parte in enumerate(("train", "val", "test"))}).T
hist_clf.to_csv(c.TAB_DIR / "mlp_historia_clasificacion.csv", index=False, encoding="utf-8")
print(f"épocas: {len(hist_clf)} (mejor: {int(hist_clf['val'].idxmin()) + 1}) | {seg_clf:.0f} s")
res_mlp_clf.round(4)

# %% [markdown]
# ## 3. Curvas de pérdida

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
for ax, h, titulo, ylab in ((axes[0], hist_reg, "Regresión (error cuadrático medio estandarizado)", "Pérdida (MSE)"),
                            (axes[1], hist_clf, "Clasificación (entropía cruzada ponderada)", "Pérdida (BCE)")):
    ax.plot(h["epoca"], h["train"], "o-", ms=3, color=c.PALETA[0], label="Entrenamiento")
    ax.plot(h["epoca"], h["val"], "s-", ms=3, color=c.PALETA[3], label="Validación")
    mejor = int(h["val"].idxmin())
    ax.axvline(h.loc[mejor, "epoca"], color="#777777", ls="--", lw=1, label=f"Mejor época ({h.loc[mejor, 'epoca']})")
    ax.set(title=titulo, xlabel="Época", ylabel=ylab)
    ax.legend(fontsize=8)
viz.guardar(fig, "fig_36_mlp_curvas_perdida")

# %% [markdown]
# ## 4. Comparación con los modelos de scikit-learn y XGBoost (conjunto de prueba)

# %%
reg_test = pd.read_csv(c.TAB_DIR / "ml_regresion_test.csv", index_col=0)
reg_test.loc["MLP (PyTorch)"] = res_mlp_reg.loc["test"]
clf_test = pd.read_csv(c.TAB_DIR / "ml_clasificacion_test.csv", index_col=0)
clf_test.loc["MLP (PyTorch)"] = res_mlp_clf.loc["test"]
reg_test.to_csv(c.TAB_DIR / "final_regresion_test.csv", encoding="utf-8")
clf_test.to_csv(c.TAB_DIR / "final_clasificacion_test.csv", encoding="utf-8")
resumen = pd.Series({"entradas": X[0].shape[1], "parametros": n_param, "epocas_regresion": len(hist_reg), "mejor_epoca_regresion": int(hist_reg["val"].idxmin()) + 1,
                     "epocas_clasificacion": len(hist_clf), "mejor_epoca_clasificacion": int(hist_clf["val"].idxmin()) + 1,
                     "segundos_regresion": seg_reg, "segundos_clasificacion": seg_clf,
                     "R2_train": res_mlp_reg.loc["train", "R2"], "R2_val": res_mlp_reg.loc["val", "R2"], "R2_test": res_mlp_reg.loc["test", "R2"],
                     "RMSE_val": res_mlp_reg.loc["val", "RMSE"], "AUC_train": res_mlp_clf.loc["train", "AUC_ROC"], "AUC_val": res_mlp_clf.loc["val", "AUC_ROC"],
                     "AUC_test": res_mlp_clf.loc["test", "AUC_ROC"]})
resumen.to_csv(c.TAB_DIR / "mlp_resumen.csv", header=["valor"], encoding="utf-8")
pd.concat({"MLP regresión": res_mlp_reg, "MLP clasificación": res_mlp_clf}).to_csv(c.TAB_DIR / "mlp_metricas.csv", encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
o = reg_test.sort_values("RMSE", ascending=False)
axes[0].barh(o.index, o["RMSE"], color=[c.PALETA[3] if "MLP" in i else c.PALETA[0] for i in o.index])
for i, v in enumerate(o["RMSE"]): axes[0].text(v + 0.1, i, f"{v:.2f}".replace(".", ","), va="center", fontsize=8)
axes[0].set(title="Regresión: RMSE en prueba (menor es mejor)", xlabel="RMSE (puntos)", xlim=(0, o["RMSE"].max() * 1.12))
o = clf_test.sort_values("AUC_ROC")
axes[1].barh(o.index, o["AUC_ROC"], color=[c.PALETA[3] if "MLP" in i else c.PALETA[2] for i in o.index])
for i, v in enumerate(o["AUC_ROC"]): axes[1].text(v + 0.003, i, f"{v:.3f}".replace(".", ","), va="center", fontsize=8)
axes[1].set(title="Clasificación: AUC-ROC en prueba (mayor es mejor)", xlabel="AUC-ROC", xlim=(0.5, o["AUC_ROC"].max() * 1.06))
viz.guardar(fig, "fig_37_comparacion_final")
print(reg_test.round(4)); clf_test.round(4)

# %%
torch.save({"estado": mlp_reg.state_dict(), "media_y": mu, "desv_y": sd, "n_entrada": X[0].shape[1]}, c.ROOT / "dashboard" / "modelos" / "mlp_regresion.pt")
