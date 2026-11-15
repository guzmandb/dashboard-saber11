"""Dashboard del Proyecto Integrador — Saber 11° 2020-2.

Ejecutar desde la raíz del proyecto:  streamlit run dashboard/app.py
Borrador de la Fase 1: filtros, KPI, exploración y clústeres en el plano PCA.
La predicción en vivo se añade en la Fase 5, cuando exista el modelo serializado.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "processed" / "muestra_pca_clusters.parquet"
PALETA = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
AREAS = {
    "punt_lectura_critica": "Lectura crítica",
    "punt_matematicas": "Matemáticas",
    "punt_c_naturales": "Ciencias naturales",
    "punt_sociales_ciudadanas": "Sociales y ciudadanas",
    "punt_ingles": "Inglés",
}

st.set_page_config(page_title="Saber 11° 2020-2", page_icon="📊", layout="wide")


@st.cache_data
def cargar() -> pd.DataFrame:
    df = pd.read_parquet(DATOS)
    for col in df.select_dtypes("category"):
        df[col] = df[col].astype(object).where(df[col].notna(), "Sin dato").astype(str)
    return df


df = cargar()

# ---------------------------------------------------------------- Filtros
st.sidebar.header("Filtros")
deptos = st.sidebar.multiselect("Departamento del colegio", sorted(df["cole_depto_ubicacion"].unique()))
naturaleza = st.sidebar.multiselect("Naturaleza del colegio", sorted(df["cole_naturaleza"].unique()))
zona = st.sidebar.multiselect("Zona", sorted(df["cole_area_ubicacion"].unique()))
estratos = st.sidebar.multiselect("Estrato", sorted(df["fami_estratovivienda"].unique()))
internet = st.sidebar.radio("Internet en el hogar", ["Todos", "Si", "No"], horizontal=True)
rango = st.sidebar.slider("Puntaje global", 0, 500, (0, 500), step=10)
inse = st.sidebar.slider("Índice socioeconómico (INSE)", 10, 90, (10, 90))

f = df[df["punt_global"].between(*rango)]
# El filtro de INSE conserva los registros sin índice mientras el rango esté completo.
if inse != (10, 90):
    f = f[f["estu_inse_individual"].between(*inse)]
for col, sel in [("cole_depto_ubicacion", deptos), ("cole_naturaleza", naturaleza), ("cole_area_ubicacion", zona), ("fami_estratovivienda", estratos)]:
    if sel:
        f = f[f[col].isin(sel)]
if internet != "Todos":
    f = f[f["fami_tieneinternet"] == internet]

# ---------------------------------------------------------------- Encabezado y KPI
st.title("Desempeño en el Saber 11° 2020-2")
st.caption("Muestra estratificada de 60 000 estudiantes sobre 504 872 registros. Fuente: ICFES, datos.gov.co (CC BY-SA 4.0).")

if f.empty:
    st.warning("Ningún estudiante cumple los filtros seleccionados.")
    st.stop()

media_nac = df["punt_global"].mean()
umbral = df["punt_global"].quantile(0.75)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Estudiantes", f"{len(f):,}".replace(",", " "))
k2.metric("Puntaje global medio", f"{f['punt_global'].mean():.1f}", f"{f['punt_global'].mean() - media_nac:+.1f} vs. total")
k3.metric("En el cuartil superior", f"{100 * (f['punt_global'] >= umbral).mean():.1f} %", f"{100 * (f['punt_global'] >= umbral).mean() - 100 * (df['punt_global'] >= umbral).mean():+.1f} pp")
k4.metric("Sin internet", f"{100 * (f['fami_tieneinternet'] == 'No').mean():.1f} %")
k5.metric("Colegio oficial", f"{100 * (f['cole_naturaleza'] == 'OFICIAL').mean():.1f} %")

# `?vista=brechas` o `?vista=segmentacion` abre el dashboard directamente en esa pestaña
# (sirve para enlazar una vista concreta y para las capturas de los informes).
TITULOS = {"exploracion": "Exploración", "brechas": "Brechas", "segmentacion": "Segmentación (PCA + K-Means)"}
vista = st.query_params.get("vista", "exploracion")
claves = sorted(TITULOS, key=lambda k: k != vista)
pestanas = dict(zip(claves, st.tabs([TITULOS[k] for k in claves])))
tab1, tab2, tab3 = pestanas["exploracion"], pestanas["brechas"], pestanas["segmentacion"]

# ---------------------------------------------------------------- Exploración
with tab1:
    a, b = st.columns(2)
    fig = px.histogram(f, x="punt_global", nbins=60, color_discrete_sequence=[PALETA[0]],
                       labels={"punt_global": "Puntaje global (0–500)"}, title="Distribución del puntaje global")
    fig.add_vline(x=media_nac, line_dash="dash", annotation_text="Media de la muestra")
    fig.update_layout(yaxis_title="Estudiantes", template="plotly_white")
    a.plotly_chart(fig, width="stretch")

    medias = f[list(AREAS)].mean().rename(AREAS).reset_index()
    medias.columns = ["Área", "Puntaje medio"]
    ref = df[list(AREAS)].mean().rename(AREAS)
    medias["Total de la muestra"] = medias["Área"].map(ref)
    fig = px.bar(medias.melt("Área", var_name="Grupo", value_name="Puntaje"), x="Área", y="Puntaje", color="Grupo", barmode="group",
                 color_discrete_sequence=[PALETA[0], "#BBBBBB"], title="Puntaje medio por área: selección frente al total")
    fig.update_layout(template="plotly_white", yaxis_range=[30, 75], legend_title=None)
    b.plotly_chart(fig, width="stretch")

    por_depto = f.groupby("cole_depto_ubicacion")["punt_global"].agg(Media="mean", Estudiantes="count").reset_index().sort_values("Media")
    fig = px.bar(por_depto, x="Media", y="cole_depto_ubicacion", orientation="h", hover_data=["Estudiantes"], color="Media", color_continuous_scale="Viridis",
                 labels={"cole_depto_ubicacion": "", "Media": "Puntaje global medio"}, title="Puntaje global medio por departamento", height=720)
    fig.update_layout(template="plotly_white", coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Brechas
with tab2:
    variable = st.selectbox("Comparar el puntaje global según", {
        "fami_estratovivienda": "Estrato", "cole_naturaleza": "Naturaleza del colegio", "cole_area_ubicacion": "Zona",
        "fami_tieneinternet": "Internet en el hogar", "fami_tienecomputador": "Computador en el hogar",
        "fami_educacionmadre": "Educación de la madre", "cole_jornada": "Jornada", "estu_genero": "Género"}.items(), format_func=lambda kv: kv[1])
    col = variable[0]
    orden = f.groupby(col)["punt_global"].median().sort_values().index.tolist() if col != "fami_estratovivienda" else sorted(f[col].unique(), key=lambda s: (s != "Sin Estrato", s))
    fig = px.box(f, x=col, y="punt_global", category_orders={col: orden}, color_discrete_sequence=[PALETA[0]], points=False,
                 labels={col: variable[1], "punt_global": "Puntaje global (0–500)"}, title=f"Puntaje global según {variable[1].lower()}")
    fig.add_hline(y=media_nac, line_dash="dash")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, width="stretch")
    tabla = f.groupby(col)["punt_global"].agg(Estudiantes="count", Media="mean", Mediana="median", Desv_est="std").reindex(orden).round(1)
    st.dataframe(tabla, width="stretch")

    g = f.dropna(subset=["estu_inse_individual"])
    fig = px.density_heatmap(g, x="estu_inse_individual", y="punt_global", nbinsx=40, nbinsy=40, color_continuous_scale="Viridis",
                             labels={"estu_inse_individual": "Índice socioeconómico INSE", "punt_global": "Puntaje global"},
                             title="Nivel socioeconómico y puntaje global (densidad)")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Segmentación
with tab3:
    st.markdown(
        "Tres clústeres K-Means sobre los cinco puntajes estandarizados. **CP1** resume el desempeño general "
        "(74,6 % de la varianza) y **CP2** contrasta inglés con las demás áreas (9,4 %). La silueta media es 0,31: "
        "los grupos son tramos de un continuo, no poblaciones separadas."
    )
    a, b = st.columns([3, 2])
    v = f.sample(min(len(f), 8000), random_state=42).sort_values("cluster")
    fig = px.scatter(v, x="CP1", y="CP2", color="perfil", opacity=0.5, color_discrete_sequence=PALETA, render_mode="webgl",
                     hover_data={"punt_global": True, "punt_ingles": True, "cole_naturaleza": True, "fami_estratovivienda": True, "CP1": ":.2f", "CP2": ":.2f"},
                     labels={"CP1": "CP1 · desempeño general", "CP2": "CP2 · inglés frente al resto", "perfil": "Clúster"},
                     title=f"Clústeres en el plano PCA ({len(v):,} puntos)".replace(",", " "), height=560)
    fig.update_traces(marker={"size": 4})
    fig.update_layout(template="plotly_white", legend={"orientation": "h", "y": -0.15})
    a.plotly_chart(fig, width="stretch")

    perfil = f.groupby("perfil")[list(AREAS)].mean().rename(columns=AREAS)
    fig = px.imshow(perfil.round(1), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", color_continuous_midpoint=float(df[list(AREAS)].mean().mean()),
                    title="Perfil promedio por clúster")
    fig.update_layout(template="plotly_white", coloraxis_showscale=False, height=300)
    b.plotly_chart(fig, width="stretch")
    resumen = f.groupby("perfil").agg(Estudiantes=("punt_global", "count"), Global=("punt_global", "mean"), INSE=("estu_inse_individual", "mean"),
                                      Oficial=("cole_naturaleza", lambda s: 100 * (s == "OFICIAL").mean()),
                                      Sin_internet=("fami_tieneinternet", lambda s: 100 * (s == "No").mean())).round(1)
    resumen.columns = ["Estudiantes", "Global medio", "INSE medio", "% oficial", "% sin internet"]
    b.dataframe(resumen, width="stretch")
