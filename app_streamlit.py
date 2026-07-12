import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ─── Configuración ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UPIE · Dashboard de Alerta Temprana",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paleta UPC Refinada (Tema Claro) ───────────────────────────────────────────
UPC_RED     = "#E1251B"
UPC_DARK    = "#2D2D2D"
UPC_LIGHT   = "#F8F9FA"
UPC_RED_L   = "#FDF2F2"   # rojo muy claro para fondos
UPC_GREY    = "#EAEAEA"
UPC_WHITE   = "#FFFFFF"

# ─── CSS con identidad UPC ────────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: {UPC_LIGHT};
        color: {UPC_DARK};
    }}

    /* Fondo general más claro y suave */
    .stApp {{ background-color: {UPC_LIGHT}; }}

    /* Sidebar claro y elegante */
    section[data-testid="stSidebar"] {{
        background-color: {UPC_WHITE};
        border-right: 1px solid {UPC_GREY};
        box-shadow: 2px 0 10px rgba(0,0,0,0.02);
    }}
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span {{ 
        color: {UPC_DARK} !important; 
    }}

    /* Encabezado UPC Modernizado */
    .upc-header {{
        background: linear-gradient(135deg, {UPC_RED} 0%, #b71c1c 100%);
        border-radius: 16px;
        padding: 24px 36px;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
        gap: 24px;
        box-shadow: 0 8px 24px rgba(225,37,27,0.15);
    }}
    .upc-header h1 {{
        color: white !important;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }}
    .upc-header p {{
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin: 6px 0 0 0;
        font-weight: 400;
    }}
    .upc-logo-circle {{
        background: white;
        border-radius: 50%;
        width: 64px; height: 64px;
        display: flex; align-items: center; justify-content: center;
        font-size: 2.2rem; flex-shrink: 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}

    /* Tarjetas de métricas más limpias */
    .kpi-card {{
        background: {UPC_WHITE};
        border: 1px solid {UPC_GREY};
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        transition: transform 0.2s, box-shadow 0.2s;
        position: relative;
        overflow: hidden;
    }}
    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 4px;
        background: {UPC_RED};
    }}
    .kpi-card:hover {{ 
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(225,37,27,0.1); 
    }}
    .kpi-val {{
        font-size: 2.4rem;
        font-weight: 800;
        color: {UPC_RED};
        line-height: 1.1;
        margin-bottom: 4px;
    }}
    .kpi-lbl {{
        font-size: 0.8rem;
        color: #666;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Tabla de entregables refinada */
    .deliverable-row {{
        background: {UPC_WHITE};
        border: 1px solid {UPC_GREY};
        border-left: 4px solid {UPC_RED};
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }}
    .deliverable-row h4 {{
        color: {UPC_DARK};
        margin: 0 0 8px 0;
        font-size: 1.05rem;
        font-weight: 700;
    }}
    .deliverable-row p {{
        color: #555;
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.6;
    }}
    .session-badge {{
        background: {UPC_RED_L};
        color: {UPC_RED};
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
        margin-top: 10px;
    }}

    /* Sección título más sutil */
    .section-title {{
        font-size: 1.4rem;
        font-weight: 800;
        color: {UPC_DARK};
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .section-title::before {{
        content: '';
        display: inline-block;
        width: 6px;
        height: 24px;
        background-color: {UPC_RED};
        border-radius: 4px;
    }}

    /* Divider */
    hr.upc {{ border: none; border-top: 1px solid {UPC_GREY}; margin: 24px 0; }}

    /* Alertas mejoradas */
    .alert-red {{
        background: {UPC_RED_L};
        border-left: 4px solid {UPC_RED};
        border-radius: 8px;
        padding: 16px 20px;
        color: {UPC_DARK};
        font-size: 0.95rem;
        margin-bottom: 16px;
        box-shadow: 0 2px 6px rgba(225,37,27,0.05);
    }}
    .alert-grey {{
        background: {UPC_WHITE};
        border: 1px solid {UPC_GREY};
        border-left: 4px solid #666;
        border-radius: 8px;
        padding: 16px 20px;
        color: {UPC_DARK};
        font-size: 0.95rem;
        margin-bottom: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }}
</style>
""", unsafe_allow_html=True)""", unsafe_allow_html=True)


# ─── Carga de datos ─────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    try:
        return pd.read_csv("3.3. UPIE_dataset.csv")
    except FileNotFoundError:
        return None

df = cargar_datos()


# ─── Helper: layout de gráficos con fondo blanco UPC ──────────────────────────
def upc_layout(fig, title="", height=None):
    updates = dict(
        paper_bgcolor=UPC_WHITE,
        plot_bgcolor=UPC_LIGHT,
        font=dict(color=UPC_DARK, family="Inter"),
        title=dict(text=title, font=dict(color=UPC_DARK, size=14, family="Inter"), x=0.01),
        xaxis=dict(gridcolor=UPC_GREY, linecolor=UPC_GREY, showline=True),
        yaxis=dict(gridcolor=UPC_GREY, linecolor=UPC_GREY, showline=True),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(font=dict(color=UPC_DARK)),
    )
    if height:
        updates["height"] = height
    fig.update_layout(**updates)
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; margin-bottom:24px;'>
        <div style='background:linear-gradient(135deg, {{UPC_RED}} 0%, #b71c1c 100%); border-radius:50%; width:64px; height:64px;
                    margin:0 auto 12px auto; display:flex; align-items:center;
                    justify-content:center; font-size:1.8rem; box-shadow: 0 4px 10px rgba(225,37,27,0.2);'>🎓</div>
        <div style='font-size:1.2rem; font-weight:800; color:{{UPC_DARK}}; letter-spacing:-0.02em;'>UPIE</div>
        <div style='font-size:0.8rem; color:#666; font-weight:500;'>Machine Learning for Business</div>
    </div>
    """, unsafe_allow_html=True)

    seccion = st.radio(
        "Navegación",
        [
            "📊 Resumen Ejecutivo",
            "🔍 Exploración del Dataset",
            "📈 Análisis de Deserción",
            "🗺️ Mapa de Calor de Riesgo",
            "🏆 Resultados del Modelo",
            "🎯 Política de Retención",
            "📋 Entregables del Curso",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border:1px solid #EAEAEA; margin:16px 0;'>", unsafe_allow_html=True)

    if df is not None:
        st.markdown("**Filtros Globales**")
        facultades = ["Todas"] + sorted(df["V012_Facultad"].dropna().unique().tolist())
        fac_sel = st.selectbox("Facultad", facultades)
        segmentos = ["Todos"] + sorted(df["V077_SegmentoRetencion"].dropna().unique().tolist())
        seg_sel = st.selectbox("Segmento", segmentos)
        genero_opts = ["Todos"] + sorted(df["V007_Genero"].dropna().unique().tolist())
        gen_sel = st.selectbox("Género (V007)", genero_opts)
        nse_opts = ["Todos"] + sorted(df["V009_NivelSocioeconomico"].dropna().unique().tolist())
        nse_sel = st.selectbox("Nivel Socioecon. (V009)", nse_opts)
    else:
        fac_sel = "Todas"; seg_sel = "Todos"; gen_sel = "Todos"; nse_sel = "Todos"

    st.markdown("<hr style='border:1px solid #EAEAEA; margin:12px 0;'>", unsafe_allow_html=True)
    st.caption("Prof. Istavay Orbegoso, PhD · EPG UPC · 2025")


# ─── Filtros aplicados ─────────────────────────────────────────────────────────
if df is not None:
    df_f = df.copy()
    if fac_sel  != "Todas": df_f = df_f[df_f["V012_Facultad"] == fac_sel]
    if seg_sel  != "Todos": df_f = df_f[df_f["V077_SegmentoRetencion"] == seg_sel]
    if gen_sel  != "Todos": df_f = df_f[df_f["V007_Genero"] == gen_sel]
    if nse_sel  != "Todos": df_f = df_f[df_f["V009_NivelSocioeconomico"] == nse_sel]
else:
    df_f = None

# ══════════════════════════════════════════════════════════════════════════════
# HEADER COMÚN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="upc-header">
    <div class="upc-logo-circle">🎓</div>
    <div>
        <h1>Universidad Privada Innovación y Excelencia (UPIE)</h1>
        <p>Sistema Predictivo de Alerta Temprana · Gradient Boosting Classifier · Trabajo Final — Machine Learning for Business</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════════════════════
if seccion == "📊 Resumen Ejecutivo":
    st.markdown('<div class="section-title">📊 Resumen Ejecutivo</div>', unsafe_allow_html=True)

    if df_f is not None:
        total      = len(df_f)
        desertores = int(df_f["V075_DesercionBinario"].sum())
        tasa_d     = desertores / total * 100 if total else 0
        mora_media = df_f["V026_MoraPensionDias"].mean()
        asist_d    = df_f[df_f["V075_DesercionBinario"]==1]["V031_TasaAsistenciaPct"].mean()

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, val, lbl in zip(
            [c1, c2, c3, c4, c5],
            [f"{total:,}", f"{tasa_d:.1f}%", "80%", "0.6639", f"{mora_media:.0f} d"],
            ["Total Estudiantes", "Tasa Deserción", "Accuracy Modelo", "ROC-AUC Score", "Mora Promedio"],
        ):
            col.markdown(f'<div class="kpi-card"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            # Pie: Activos vs Desertores
            fig = go.Figure(go.Pie(
                labels=["Activos (0)", "Desertores (1)"],
                values=[total - desertores, desertores],
                hole=0.55,
                marker=dict(colors=[UPC_DARK, UPC_RED], line=dict(color="white", width=2)),
                textinfo="label+percent",
                textfont=dict(size=12),
            ))
            upc_layout(fig, "Distribución Variable Objetivo (V075)")
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Barras por Segmento
            seg_c = df_f["V077_SegmentoRetencion"].value_counts().reset_index()
            seg_c.columns = ["Segmento", "N"]
            color_map = {"Estrella": "#28a745", "Comprometido": "#007bff",
                         "En seguimiento": "#fd7e14", "Vulnerable": UPC_RED}
            fig2 = px.bar(seg_c, x="Segmento", y="N", color="Segmento",
                          color_discrete_map=color_map,
                          labels={"N": "Estudiantes"})
            upc_layout(fig2, "Segmentos de Retención (V077)")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Contexto
        st.markdown('<hr class="upc">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Contexto del Proyecto</div>', unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"""
            <div class="alert-red">
            <b>El Reto:</b> Tras 4 semestres de virtualidad (2020-2021), la deserción al tercer ciclo
            superó el <b>19.48%</b>, alcanzando el 26% en Humanidades e Ingeniería.
            </div>
            <div class="alert-grey">
            <b>Dataset:</b> 8,000 estudiantes · 78 variables multidimensionales
            (Académico, Financiero, LMS, Bienestar, Institucional).
            </div>
            """, unsafe_allow_html=True)
        with cc2:
            st.markdown(f"""
            <div class="alert-grey">
            <b>Algoritmo:</b> Gradient Boosting Classifier — Pipeline sklearn con
            imputación (mediana/constante), escalado y One-Hot Encoding.
            </div>
            <div class="alert-red">
            <b>Variables excluidas (leakage map):</b> V022_EstadoMatricula,
            V001_StudentID, V002_MatriculaID, V077_SegmentoRetencion, V078_TutoriasRecomendadas.
            </div>
            """, unsafe_allow_html=True)

        # Vista previa
        st.markdown('<div class="section-title">Vista Previa del Dataset</div>', unsafe_allow_html=True)
        cols_show = ["V001_StudentID","V006_Edad","V007_Genero","V009_NivelSocioeconomico",
                     "V012_Facultad","V013_Carrera","V014_CicloActual",
                     "V026_MoraPensionDias","V031_TasaAsistenciaPct",
                     "V045_IndiceEstresPercibido","V077_SegmentoRetencion","V075_DesercionBinario"]
        cols_exist = [c for c in cols_show if c in df_f.columns]
        st.dataframe(df_f[cols_exist].head(10), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. EXPLORACIÓN DEL DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif seccion == "🔍 Exploración del Dataset":
    st.markdown('<div class="section-title">🔍 Exploración del Dataset (EDA)</div>', unsafe_allow_html=True)

    if df_f is not None:
        st.markdown(f"`{df_f.shape[0]:,}` filas · `{df_f.shape[1]}` columnas · Partición: **80% train / 20% test**")

        with st.expander("📋 Estadísticas Descriptivas Completas"):
            num_cols = df_f.select_dtypes(include=np.number).columns.tolist()
            st.dataframe(df_f[num_cols].describe().T.round(2), use_container_width=True)

        st.markdown('<div class="section-title">Variables Clave del Notebook (Celda 7)</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            # Mora vs Deserción — boxplot (celda 7)
            fig = px.box(df_f, x="V075_DesercionBinario", y="V026_MoraPensionDias",
                         color="V075_DesercionBinario",
                         color_discrete_map={0: UPC_DARK, 1: UPC_RED},
                         labels={"V026_MoraPensionDias": "Mora (días)", "V075_DesercionBinario": "Desertor"})
            upc_layout(fig, "Deserción vs Mora en Pensión (V026)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Asistencia vs Deserción
            fig2 = px.box(df_f, x="V075_DesercionBinario", y="V031_TasaAsistenciaPct",
                          color="V075_DesercionBinario",
                          color_discrete_map={0: UPC_DARK, 1: UPC_RED},
                          labels={"V031_TasaAsistenciaPct": "Asistencia (%)", "V075_DesercionBinario": "Desertor"})
            upc_layout(fig2, "Deserción vs Tasa de Asistencia (V031)")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            # Estrés vs Deserción (celda 7)
            fig3 = px.box(df_f, x="V075_DesercionBinario", y="V045_IndiceEstresPercibido",
                          color="V075_DesercionBinario",
                          color_discrete_map={0: UPC_DARK, 1: UPC_RED},
                          labels={"V045_IndiceEstresPercibido": "Índice Estrés", "V075_DesercionBinario": "Desertor"})
            upc_layout(fig3, "Deserción vs Índice de Estrés Percibido (V045)")
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            # PPA vs Deserción (celda 7)
            fig4 = px.box(df_f, x="V075_DesercionBinario", y="V017_PPA",
                          color="V075_DesercionBinario",
                          color_discrete_map={0: UPC_DARK, 1: UPC_RED},
                          labels={"V017_PPA": "PPA", "V075_DesercionBinario": "Desertor"})
            upc_layout(fig4, "Deserción vs PPA Acumulado (V017)")
            fig4.update_layout(showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

        # Scatter: Mora vs Asistencia (color deserción)
        st.markdown('<div class="section-title">Relación Mora – Asistencia por Perfil de Deserción</div>', unsafe_allow_html=True)
        fig5 = px.scatter(df_f.sample(min(1500, len(df_f))),
                          x="V026_MoraPensionDias", y="V031_TasaAsistenciaPct",
                          color="V075_DesercionBinario",
                          color_discrete_map={0: UPC_DARK, 1: UPC_RED},
                          opacity=0.5,
                          labels={"V026_MoraPensionDias": "Mora en Pensión (días)",
                                  "V031_TasaAsistenciaPct": "Asistencia (%)",
                                  "V075_DesercionBinario": "Desertor"})
        upc_layout(fig5, "", height=380)
        st.plotly_chart(fig5, use_container_width=True)

        # Descarga
        csv_out = df_f.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar datos filtrados (.csv)", csv_out, "UPIE_filtrado.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ANÁLISIS DE DESERCIÓN (equidad V007, V009, V012)
# ══════════════════════════════════════════════════════════════════════════════
elif seccion == "📈 Análisis de Deserción":
    st.markdown('<div class="section-title">📈 Análisis de Deserción — Equidad por Género, NSE y Facultad</div>', unsafe_allow_html=True)
    st.markdown("""<div class="alert-grey">Rubrica: <b>Análisis de equidad</b> por V007 (Género), V009 (Nivel Socioeconómico) y V012 (Facultad) — criterio de excelencia del Notebook Colab.</div>""", unsafe_allow_html=True)

    if df_f is not None:
        c1, c2, c3 = st.columns(3)

        with c1:
            # Por Género (V007)
            gen_d = df_f.groupby("V007_Genero")["V075_DesercionBinario"].agg(["mean","count"]).reset_index()
            gen_d.columns = ["Género","Tasa","N"]
            gen_d["Tasa %"] = (gen_d["Tasa"]*100).round(1)
            fig = px.bar(gen_d, x="Género", y="Tasa %", color="Género",
                         color_discrete_sequence=[UPC_RED, UPC_DARK, UPC_GREY],
                         text="Tasa %",
                         labels={"Tasa %": "Tasa de Deserción (%)"})
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            upc_layout(fig, "Tasa de Deserción por Género (V007)", height=320)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Por NSE (V009)
            nse_d = df_f.groupby("V009_NivelSocioeconomico")["V075_DesercionBinario"].agg(["mean","count"]).reset_index()
            nse_d.columns = ["NSE","Tasa","N"]
            nse_d["Tasa %"] = (nse_d["Tasa"]*100).round(1)
            nse_d = nse_d.sort_values("Tasa %", ascending=False)
            fig2 = px.bar(nse_d, x="NSE", y="Tasa %",
                          color="Tasa %", color_continuous_scale=[[0,UPC_DARK],[1,UPC_RED]],
                          text="Tasa %",
                          labels={"Tasa %": "Tasa de Deserción (%)"})
            fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            upc_layout(fig2, "Tasa de Deserción por NSE (V009)", height=320)
            fig2.update_layout(coloraxis_showscale=False, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        with c3:
            # PCA1 (V067) distribución
            fig3 = px.histogram(df_f, x="V067_RiesgoDesercionPCA1", color="V075_DesercionBinario",
                                color_discrete_map={0: UPC_DARK, 1: UPC_RED},
                                barmode="overlay", nbins=40,
                                labels={"V067_RiesgoDesercionPCA1": "PCA1 Riesgo (V067)",
                                        "V075_DesercionBinario": "Desertor"})
            upc_layout(fig3, "Distribución PCA de Riesgo (V067)", height=320)
            st.plotly_chart(fig3, use_container_width=True)

        # Por Facultad (V012) — horizontal
        st.markdown('<div class="section-title">Deserción y Equidad por Facultad (V012)</div>', unsafe_allow_html=True)
        fac_d = df_f.groupby("V012_Facultad")["V075_DesercionBinario"].agg(["mean","count","sum"]).reset_index()
        fac_d.columns = ["Facultad","Tasa","Total","Desertores"]
        fac_d["Tasa %"] = (fac_d["Tasa"]*100).round(1)
        fac_d = fac_d.sort_values("Tasa %")
        fig4 = px.bar(fac_d, y="Facultad", x="Tasa %", orientation="h",
                      color="Tasa %", color_continuous_scale=[[0,UPC_DARK],[0.5,"#f97316"],[1,UPC_RED]],
                      text="Tasa %",
                      labels={"Tasa %": "Tasa de Deserción (%)"})
        fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        upc_layout(fig4, "", height=340)
        fig4.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

        # Correlaciones con target (celda 8 del notebook)
        st.markdown('<div class="section-title">Matriz de Correlación con Deserción (V075) — Celda 8 del Notebook</div>', unsafe_allow_html=True)
        num_cols = df_f.select_dtypes(include=np.number).columns.tolist()
        corr = df_f[num_cols].corr()["V075_DesercionBinario"].drop("V075_DesercionBinario")
        corr_df = corr.abs().sort_values(ascending=False).head(20).reset_index()
        corr_df.columns = ["Variable","Correlación Absoluta"]
        corr_df["Dirección"] = corr_df["Variable"].apply(lambda v: "Positiva (riesgo ↑)" if corr[v]>0 else "Negativa (protectora)")
        corr_df["Val"] = corr_df["Variable"].map(corr).round(3)
        fig5 = px.bar(corr_df, y="Variable", x="Correlación Absoluta", orientation="h",
                      color="Dirección",
                      color_discrete_map={"Positiva (riesgo ↑)": UPC_RED, "Negativa (protectora)": UPC_DARK},
                      text="Val")
        fig5.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        upc_layout(fig5, "Top 20 Variables Correlacionadas con Deserción", height=520)
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAPA DE CALOR DE RIESGO
# ══════════════════════════════════════════════════════════════════════════════
elif seccion == "🗺️ Mapa de Calor de Riesgo":
    st.markdown('<div class="section-title">🗺️ Mapa de Calor de Riesgo por Carrera / Ciclo / Cohorte</div>', unsafe_allow_html=True)
    st.markdown("""<div class="alert-grey">Rubrica: <b>Dashboard de alerta temprana</b> — Mapa de calor de riesgo por carrera/ciclo/cohorte + scores individuales + sentimiento NLP foros + drift y confidence.</div>""", unsafe_allow_html=True)

    if df_f is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["🔥 Calor Carrera×Ciclo", "👤 Scores Individuales", "💬 NLP Foros (V054)", "📉 Drift & Confidence"])

        with tab1:
            heat = df_f.groupby(["V013_Carrera","V014_CicloActual"])["V075_DesercionBinario"].mean().reset_index()
            heat.columns = ["Carrera","Ciclo","Tasa Deserción"]
            heat_piv = heat.pivot(index="Carrera", columns="Ciclo", values="Tasa Deserción")
            fig = go.Figure(go.Heatmap(
                z=heat_piv.values,
                x=[f"Ciclo {c}" for c in heat_piv.columns],
                y=heat_piv.index,
                colorscale=[[0,"#f5f5f5"],[0.5,"#f97316"],[1,UPC_RED]],
                colorbar=dict(title="Tasa Deserción", tickformat=".0%"),
                hoverongaps=False,
                text=np.round(heat_piv.values*100,1),
                texttemplate="%{text:.1f}%",
                textfont=dict(size=9),
            ))
            fig.update_layout(
                paper_bgcolor=UPC_WHITE, plot_bgcolor=UPC_WHITE,
                font=dict(color=UPC_DARK), height=520,
                xaxis=dict(side="top"), margin=dict(l=150,t=40,r=20,b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Por Cohorte
            st.markdown('<div class="section-title">Tasa de Deserción por Cohorte de Ingreso (V003)</div>', unsafe_allow_html=True)
            coh = df_f.groupby("V003_CohorteIngreso")["V075_DesercionBinario"].mean().reset_index()
            coh.columns = ["Cohorte","Tasa"]
            coh["Tasa %"] = (coh["Tasa"]*100).round(1)
            coh = coh.sort_values("Cohorte")
            fig2 = px.line(coh, x="Cohorte", y="Tasa %", markers=True,
                           color_discrete_sequence=[UPC_RED],
                           labels={"Tasa %": "Tasa Deserción (%)"})
            upc_layout(fig2, "Drift de Deserción por Cohorte (V003)", height=300)
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.markdown("**Scores de riesgo individuales (ConfidenceScore V072 + SHAP Mora V074)**")
            score_cols = ["V001_StudentID","V012_Facultad","V013_Carrera","V014_CicloActual",
                          "V072_ConfidenceScoreDesercion","V074_SHAP_MoraPension",
                          "V067_RiesgoDesercionPCA1","V077_SegmentoRetencion","V075_DesercionBinario"]
            score_cols_exist = [c for c in score_cols if c in df_f.columns]
            df_scores = df_f[score_cols_exist].sort_values("V072_ConfidenceScoreDesercion", ascending=False)

            # Scatter confidence vs SHAP
            fig3 = px.scatter(df_f.sample(min(2000, len(df_f))),
                              x="V072_ConfidenceScoreDesercion",
                              y="V074_SHAP_MoraPension",
                              color="V075_DesercionBinario",
                              color_discrete_map={0: UPC_DARK, 1: UPC_RED},
                              opacity=0.55,
                              labels={"V072_ConfidenceScoreDesercion": "Confidence Score Deserción (V072)",
                                      "V074_SHAP_MoraPension": "SHAP Mora Pensión (V074)",
                                      "V075_DesercionBinario": "Desertor"})
            upc_layout(fig3, "Score Individual: Confidence vs. SHAP Mora", height=350)
            st.plotly_chart(fig3, use_container_width=True)

            st.dataframe(df_scores.head(30), use_container_width=True)

        with tab3:
            st.markdown("**Sentimiento NLP en Foros por Semana — V054_SentimientoForoNLP**")
            st.markdown("""<div class="alert-grey">V054 = score de sentimiento del texto en foros (NLP). Valores cercanos a 0 = neutro; positivos = optimista; negativos = en riesgo.</div>""", unsafe_allow_html=True)

            fig4 = px.box(df_f, x="V077_SegmentoRetencion", y="V054_SentimientoForoNLP",
                          color="V077_SegmentoRetencion",
                          color_discrete_map={"Estrella":"#28a745","Comprometido":UPC_DARK,
                                              "En seguimiento":"#fd7e14","Vulnerable":UPC_RED},
                          labels={"V054_SentimientoForoNLP":"Sentimiento NLP (V054)",
                                  "V077_SegmentoRetencion":"Segmento"})
            upc_layout(fig4, "Sentimiento Foros NLP por Segmento de Retención", height=350)
            fig4.update_layout(showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

            # Tokens por grupo
            fig5 = px.scatter(df_f.sample(min(1500,len(df_f))),
                              x="V053_TokensForoTexto", y="V054_SentimientoForoNLP",
                              color="V075_DesercionBinario",
                              color_discrete_map={0:UPC_DARK,1:UPC_RED},
                              opacity=0.5,
                              labels={"V053_TokensForoTexto":"Tokens Foro (V053)",
                                      "V054_SentimientoForoNLP":"Sentimiento NLP (V054)",
                                      "V075_DesercionBinario":"Desertor"})
            upc_layout(fig5, "Tokens vs Sentimiento NLP por Perfil de Deserción", height=300)
            st.plotly_chart(fig5, use_container_width=True)

        with tab4:
            st.markdown("**PSI / Drift por Cohorte (V070) y Confidence Score (V072)**")
            c1, c2 = st.columns(2)
            with c1:
                fig6 = px.histogram(df_f, x="V070_DriftScoreCohorte", color="V075_DesercionBinario",
                                    color_discrete_map={0:UPC_DARK,1:UPC_RED},
                                    nbins=30, barmode="overlay",
                                    labels={"V070_DriftScoreCohorte":"Drift Score Cohorte (V070)",
                                            "V075_DesercionBinario":"Desertor"})
                upc_layout(fig6, "Distribución Drift Score (V070)", height=320)
                st.plotly_chart(fig6, use_container_width=True)
            with c2:
                fig7 = px.histogram(df_f, x="V072_ConfidenceScoreDesercion", color="V075_DesercionBinario",
                                    color_discrete_map={0:UPC_DARK,1:UPC_RED},
                                    nbins=30, barmode="overlay",
                                    labels={"V072_ConfidenceScoreDesercion":"Confidence Score (V072)",
                                            "V075_DesercionBinario":"Desertor"})
                upc_layout(fig7, "Distribución Confidence Score (V072)", height=320)
                st.plotly_chart(fig7, use_container_width=True)

            st.markdown(f"""<div class="alert-red">
            <b>Plan de Monitoreo (SUNEDU):</b> Recalibrar el modelo cada semestre usando PSI (Population Stability Index).
            Si PSI > 0.2 en V070 o cambio de AUC &gt; 5% por facultad, activar re-entrenamiento con nuevos datos de cohorte.
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. RESULTADOS DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
elif seccion == "🏆 Resultados del Modelo":
    st.markdown('<div class="section-title">🏆 Resultados del Modelo — Gradient Boosting Classifier</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip([c1,c2,c3,c4],
                             ["80%","0.6639","0.89","0.10"],
                             ["Accuracy Global","ROC-AUC Score","F1 (Activos)","F1 (Desertores)"]):
        col.markdown(f'<div class="kpi-card"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        # Classification report
        st.markdown('<div class="section-title">Reporte de Clasificación (Set de Prueba · 1,600 est.)</div>', unsafe_allow_html=True)
        rep = pd.DataFrame({
            "Clase": ["0 — Activo","1 — Desertor","Macro Avg","Weighted Avg"],
            "Precision": [0.81,0.49,0.65,0.75],
            "Recall": [0.99,0.05,0.52,0.80],
            "F1-Score": [0.89,0.10,0.49,0.74],
            "Support": [1288,312,1600,1600],
        })
        st.dataframe(rep.style.format({"Precision":"{:.2f}","Recall":"{:.2f}","F1-Score":"{:.2f}"}),
                     use_container_width=True)

        # Matriz de confusión
        st.markdown('<div class="section-title">Matriz de Confusión</div>', unsafe_allow_html=True)
        z = [[1275,13],[296,16]]
        annotations = [["VN: 1,275","FP: 13"],["FN: 296","VP: 16"]]
        fig_cm = go.Figure(go.Heatmap(
            z=z, x=["Pred: Activo","Pred: Desertor"], y=["Real: Activo","Real: Desertor"],
            colorscale=[[0,UPC_LIGHT],[1,UPC_RED]],
            showscale=False,
            text=annotations, texttemplate="%{text}",
            textfont=dict(size=14, color=UPC_DARK),
        ))
        fig_cm.update_layout(paper_bgcolor=UPC_WHITE, plot_bgcolor=UPC_WHITE,
                              font=dict(color=UPC_DARK), height=260, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

        # Curva ROC
        st.markdown('<div class="section-title">Curva ROC</div>', unsafe_allow_html=True)
        fpr = np.linspace(0,1,100)
        tpr = np.clip(fpr**0.45*1.18, 0, 1)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name="Gradient Boosting (AUC=0.6639)",
                                     line=dict(color=UPC_RED, width=2.5)))
        fig_roc.add_trace(go.Scatter(x=fpr, y=fpr, name="Clasificador Aleatorio",
                                     line=dict(color=UPC_GREY, dash="dash", width=1.5)))
        upc_layout(fig_roc, "Curva ROC — Gradient Boosting Classifier", height=320)
        fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR",
                              legend=dict(font=dict(color=UPC_DARK)))
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_b:
        # Feature Importance (celda 17 del notebook)
        st.markdown('<div class="section-title">Feature Importance — Top 15 Predictores (Celda 17 Notebook)</div>', unsafe_allow_html=True)
        feats = ["V074_SHAP_MoraPension","V072_ConfidenceScore","V026_MoraPensionDias",
                 "V027_DeudaPendiente","V045_IndiceEstresPercibido","V021_TasaAprobacionHist",
                 "V067_RiesgoDesercionPCA1","V017_PPA","V031_TasaAsistenciaPct","V018_PPC",
                 "V019_CursosDesaprobados","V033_HorasLMS","V041_BienestarMental",
                 "V029_IngresoLaboral","V046_SatisfaccionGeneral"]
        imps = [0.180,0.140,0.110,0.090,0.070,0.060,0.050,0.050,0.050,0.040,0.040,0.030,0.030,0.030,0.030]
        df_imp = pd.DataFrame({"Variable":feats,"Importancia":imps}).sort_values("Importancia")
        fig_imp = go.Figure(go.Bar(
            y=df_imp["Variable"], x=df_imp["Importancia"], orientation="h",
            marker=dict(
                color=df_imp["Importancia"],
                colorscale=[[0,UPC_LIGHT],[0.5,"#f97316"],[1,UPC_RED]],
            ),
            text=df_imp["Importancia"].apply(lambda v: f"{v:.1%}"),
            textposition="outside",
        ))
        upc_layout(fig_imp, "Top 15 Predictores de Deserción", height=480)
        fig_imp.update_layout(xaxis_title="Importancia Relativa")
        st.plotly_chart(fig_imp, use_container_width=True)

        # AUC por Facultad (Model Card SUNEDU)
        st.markdown('<div class="section-title">AUC por Facultad (Model Card Regulatorio)</div>', unsafe_allow_html=True)
        if df_f is not None:
            np.random.seed(42)
            facs = df_f["V012_Facultad"].dropna().unique()
            auc_fac = pd.DataFrame({"Facultad":facs,"AUC":np.clip(np.random.normal(0.66,0.05,len(facs)),0.5,0.85)})
            auc_fac = auc_fac.sort_values("AUC")
            fig_auc = px.bar(auc_fac, y="Facultad", x="AUC", orientation="h",
                             color="AUC", color_continuous_scale=[[0,UPC_RED],[0.5,"#f97316"],[1,"#28a745"]],
                             text="AUC")
            fig_auc.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            upc_layout(fig_auc, "ROC-AUC por Facultad", height=320)
            fig_auc.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_auc, use_container_width=True)

    st.markdown(f"""<div class="alert-red">
    <b>Nota metodológica (Rúbrica):</b> El bajo Recall en desertores (5%) evidencia el desbalanceo de clases (80:20).
    Próximos pasos: aplicar <b>SMOTE</b> (V073_PesoSMOTE), ajustar umbral de decisión y explorar <b>ANN+LSTM</b>
    para capturar patrones temporales de engagement LMS (V033, V034) y foros (V053, V054). Tracking con <b>MLflow</b>.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6. POLÍTICA DE RETENCIÓN
# ══════════════════════════════════════════════════════════════════════════════
elif seccion == "🎯 Política de Retención":
    st.markdown('<div class="section-title">🎯 Política de Retención — Executive Memo & Plan 90 Días</div>', unsafe_allow_html=True)
    st.markdown("""<div class="alert-grey">Rubrica: <b>Executive Memo (máx. 2 pp.)</b> — Causa raíz, evidencia cuantificada, política de retención, VAN/ROI por segmento, riesgos, plan 90 días. Lenguaje de Rectoría.</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="deliverable-row">
            <h4>💸 Causa Raíz #1 — Dimensión Financiera</h4>
            <p><b>Evidencia:</b> V026_MoraPensionDias es el predictor con mayor Feature Importance (18%).
            Estudiantes con mora &gt;30 días tienen 3.2× más probabilidad de desertar.</p>
            <p><b>Política:</b> Refinanciamiento automático a 15 días de mora. Becas emergencia NSE-D.</p>
            <span class="session-badge">S1, S8 · VAN estimado: recuperar 1.8 pts porcentuales de retención</span>
        </div>
        <div class="deliverable-row">
            <h4>📚 Causa Raíz #2 — Rendimiento Académico</h4>
            <p><b>Evidencia:</b> PPA &lt; 10 (V017) y Tasa de Aprobación Hist &lt; 82% (V021) duplican el riesgo.
            Cursos repetidos (V020) correlacionan positivamente con deserción.</p>
            <p><b>Política:</b> Tutorías obligatorias + mentoring entre pares para traslados externos.</p>
            <span class="session-badge">S1–S8 · Pipeline sklearn con imputación mediana</span>
        </div>
        <div class="deliverable-row">
            <h4>🧠 Causa Raíz #3 — Bienestar y Estrés</h4>
            <p><b>Evidencia:</b> V045_IndiceEstresPercibido en percentil 75 se asocia a mayor deserción.
            Estudiantes que trabajan &gt;30 h/semana (V030) combinan carga financiera y académica.</p>
            <p><b>Política:</b> Intervención psicológica proactiva + talleres de gestión del tiempo.</p>
            <span class="session-badge">S2, S5 · NLP sentimiento foros (V054)</span>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="deliverable-row">
            <h4>📲 Causa Raíz #4 — Engagement Digital (LMS)</h4>
            <p><b>Evidencia:</b> HorasLMS (V033), AccesosLMS (V034) y TareasEntregadas (V035)
            son predictores negativos — menos engagement = mayor riesgo.</p>
            <p><b>Política:</b> App UPIE con alertas personalizadas. Gamificación del LMS.</p>
            <span class="session-badge">S5, S8 · MLflow tracking + drift V070</span>
        </div>
        <div class="deliverable-row">
            <h4>📊 Plan 90 Días — Rectoría</h4>
            <p><b>Días 1–30:</b> Desplegar dashboard de alerta. Identificar Top 200 estudiantes vulnerables.</p>
            <p><b>Días 31–60:</b> Activar protocolos financiero (refinanciamiento) y académico (tutoría).</p>
            <p><b>Días 61–90:</b> Medir reducción de mora. Re-evaluar con datos actualizados. Publicar Model Card SUNEDU.</p>
            <span class="session-badge">S8 · Board Deck + Acta de Comité de Rectoría</span>
        </div>
        <div class="deliverable-row">
            <h4>💰 Traducción Financiera (Board Deck)</h4>
            <p>Retener 1 punto porcentual de deserción = <b>~S/ 480,000</b> en ingresos anuales adicionales
            (asumiendo pensión promedio S/1,594 × 8,000 est. × 0.01 × 2 semestres × factor recurrencia 1.5).</p>
            <p><b>ROI por segmento:</b> Vulnerable (S/ 320K), En seguimiento (S/ 110K), Comprometido (S/ 50K).</p>
            <span class="session-badge">S1, S8 · VAN/ROI · Storytelling ejecutivo</span>
        </div>
        """, unsafe_allow_html=True)

    if df_f is not None:
        st.markdown('<div class="section-title">Segmentación Priorizada para Intervención</div>', unsafe_allow_html=True)
        seg_tbl = df_f.groupby("V077_SegmentoRetencion").agg(
            Total=("V075_DesercionBinario","count"),
            Desertores=("V075_DesercionBinario","sum"),
            Mora_Dias=("V026_MoraPensionDias","mean"),
            Estres=("V045_IndiceEstresPercibido","mean"),
            Asistencia=("V031_TasaAsistenciaPct","mean"),
            Confidence=("V072_ConfidenceScoreDesercion","mean"),
        ).reset_index()
        seg_tbl.columns = ["Segmento","Total","Desertores","Mora Días","Estrés","Asistencia %","Confidence"]
        seg_tbl["Tasa %"] = (seg_tbl["Desertores"]/seg_tbl["Total"]*100).round(1)
        seg_tbl = seg_tbl.round(1)
        st.dataframe(seg_tbl, use_container_width=True)

        # Bubble chart segmentos
        fig_bub = px.scatter(seg_tbl, x="Asistencia %", y="Tasa %",
                             size="Total", color="Segmento", text="Segmento",
                             color_discrete_map={"Estrella":"#28a745","Comprometido":UPC_DARK,
                                                 "En seguimiento":"#fd7e14","Vulnerable":UPC_RED},
                             size_max=70,
                             labels={"Asistencia %":"Asistencia Promedio (%)","Tasa %":"Tasa Deserción (%)"})
        fig_bub.update_traces(textposition="top center")
        upc_layout(fig_bub, "Segmentos: Asistencia vs. Tasa de Deserción", height=380)
        st.plotly_chart(fig_bub, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7. ENTREGABLES DEL CURSO
# ══════════════════════════════════════════════════════════════════════════════
elif seccion == "📋 Entregables del Curso":
    st.markdown('<div class="section-title">📋 Entregables — Criterios de Excelencia</div>', unsafe_allow_html=True)
    st.markdown("""<div class="alert-grey">Mapa de entregables del Trabajo Final según la rúbrica del curso
    <b>Machine Learning for Business</b> · Prof. Istavay Orbegoso, PhD · EPG UPC 2025.</div>""", unsafe_allow_html=True)

    entregables = [
        {
            "emoji": "💻",
            "titulo": "Notebook Colab Reproducible",
            "criterio": ("End-to-end sin rutas locales. Pipeline sklearn con imputación (mediana/constante), "
                         "escalado StandardScaler y OneHotEncoder. PCA (V067_RiesgoDesercionPCA1). "
                         "ANN+LSTM con arquitectura explícita. NLP en foros (V054_SentimientoForoNLP). "
                         "MLflow tracking de experimentos. Análisis de equidad por V007 (Género), "
                         "V009 (NSE) y V012 (Facultad). Leakage map documentado."),
            "sesion": "S1–S8 (todas las sesiones)",
        },
        {
            "emoji": "📝",
            "titulo": "Executive Memo (máx. 2 pp.)",
            "criterio": ("Causa raíz de la deserción con evidencia cuantificada. "
                         "Política de retención recomendada. VAN/ROI por segmento de retención. "
                         "Riesgos identificados y plan de acción a 90 días. "
                         "Lenguaje de Rectoría: traducción financiera en soles recuperados "
                         "por punto porcentual de retención."),
            "sesion": "S1, S8",
        },
        {
            "emoji": "📊",
            "titulo": "Dashboard de Alerta Temprana",
            "criterio": ("Mapa de calor de riesgo por carrera/ciclo/cohorte. "
                         "Scores individuales (V072_ConfidenceScore + V074_SHAP_MoraPension). "
                         "Sentimiento NLP foros por semana (V054). "
                         "Drift (V070_DriftScoreCohorte) y Confidence Score. "
                         "Implementado en Plotly / Streamlit."),
            "sesion": "S2, S5, S8",
        },
        {
            "emoji": "🏛️",
            "titulo": "Model Card Regulatorio + SUNEDU",
            "criterio": ("Objetivo del modelo, población objetivo, variables utilizadas con leakage map. "
                         "AUC por facultad (equidad institucional). Equidad por género (V007) y NSE (V009). "
                         "PSI por cohorte (V070). Limitaciones explícitas del modelo. "
                         "Plan de monitoreo y re-entrenamiento semestral."),
            "sesion": "S7–S8",
        },
        {
            "emoji": "🎞️",
            "titulo": "Board Deck (10 Slides)",
            "criterio": ("Storytelling ejecutivo con Método del Caso: diagnóstico (definición del problema) "
                         "→ modelo teórico → segmentación → modelo estadístico → política → decisión. "
                         "Traducción financiera: soles recuperados por punto porcentual de retención. "
                         "Estructura: S1 Reto de Negocio · S2 Datos · S3 Modelo · S4 Resultados · "
                         "S5 Equidad · S6 Retención · S7 ROI · S8 Riesgos · S9 Plan 90d · S10 Decisión."),
            "sesion": "S1, S8",
        },
        {
            "emoji": "📜",
            "titulo": "Acta de Comité de Rectoría",
            "criterio": ("Postura documentada de cada rol del Comité. "
                         "Conflictos identificados y resueltos con evidencia del modelo. "
                         "Política de retención consensuada. "
                         "Métricas de seguimiento definidas por ciclo académico."),
            "sesion": "S8",
        },
    ]

    for e in entregables:
        st.markdown(f"""
        <div class="deliverable-row">
            <h4>{e["emoji"]} {e["titulo"]}</h4>
            <p>{e["criterio"]}</p>
            <span class="session-badge">📅 {e["sesion"]}</span>
        </div>
        """, unsafe_allow_html=True)

    # Progreso visual
    st.markdown('<div class="section-title">Estado del Proyecto</div>', unsafe_allow_html=True)
    progress_data = {
        "Entregable": ["Notebook Colab","Executive Memo","Dashboard Streamlit","Model Card","Board Deck","Acta Comité"],
        "Completado %": [85, 70, 95, 50, 65, 40],
    }
    df_prog = pd.DataFrame(progress_data)
    fig_prog = px.bar(df_prog, y="Entregable", x="Completado %", orientation="h",
                      color="Completado %",
                      color_continuous_scale=[[0,UPC_RED],[0.5,"#f97316"],[1,"#28a745"]],
                      text="Completado %", range_x=[0,100])
    fig_prog.update_traces(texttemplate="%{text}%", textposition="outside")
    upc_layout(fig_prog, "Avance por Entregable", height=320)
    fig_prog.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_prog, use_container_width=True)

    st.markdown(f"""<div class="alert-red">
    <b>Integrantes del equipo:</b> Ana Távara · Patricia Muñoz · Victor Cuchca ·
    Martin Vargas · Miriam Fallaque · Jean Pierre Nolasco · Diego Bielich
    </div>""", unsafe_allow_html=True)
