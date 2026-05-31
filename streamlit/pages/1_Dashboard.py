import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from preprocessing import load_data

st.set_page_config(page_title="Dashboard analítico", page_icon="📊", layout="wide")

OBESITY_LABELS = {
    "Insufficient_Weight": "Peso insuficiente",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso I",
    "Overweight_Level_II": "Sobrepeso II",
    "Obesity_Type_I": "Obesidade I",
    "Obesity_Type_II": "Obesidade II",
    "Obesity_Type_III": "Obesidade III",
}
CLINICAL_ORDER = list(OBESITY_LABELS.values())

FAF_LABELS = {0: "Nenhuma", 1: "Baixa (1x/sem)", 2: "Moderada (2-3x)", 3: "Alta (4-5x+)"}
AGE_BINS = [13, 20, 30, 40, 200]
AGE_LABELS = ["14-20", "21-30", "31-40", "41+"]

COLOR_RISK = "#d62728"
COLOR_SAFE = "#2ca02c"
COLOR_NEUTRAL = "#4c78a8"


@st.cache_data
def get_data() -> pd.DataFrame:
    df = load_data()
    df["ObesityLabel"] = df["Obesity"].map(OBESITY_LABELS)
    df["is_obese"] = df["Obesity"].str.startswith("Obesity_")
    return df


def prevalence(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return frame["is_obese"].mean() * 100


df = get_data()

st.title("Dashboard analítico — Obesidade")
st.caption("Visão populacional para apoiar triagem, prevenção e orientação da equipe médica.")

with st.sidebar:
    st.header("Filtros")
    gender_filter = st.multiselect("Gênero", sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique()))
    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.slider("Faixa etária", age_min, age_max, (age_min, age_max))
    transport_filter = st.multiselect("Transporte", sorted(df["MTRANS"].unique()), default=sorted(df["MTRANS"].unique()))

filtered = df[
    df["Gender"].isin(gender_filter)
    & df["Age"].between(age_range[0], age_range[1])
    & df["MTRANS"].isin(transport_filter)
].copy()

if filtered.empty:
    st.warning("Nenhum registro com os filtros selecionados. Ajuste os filtros na barra lateral.")
    st.stop()

base_prev = prevalence(filtered)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros", len(filtered))
col2.metric("Prevalência de obesidade", f"{base_prev:.1f}%")
col3.metric("IMC médio", f"{filtered['BMI'].mean():.1f}")
col4.metric("Histórico familiar (+)", f"{(filtered['family_history'] == 'yes').mean() * 100:.1f}%")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Distribuição dos níveis")
    dist = (
        filtered["ObesityLabel"]
        .value_counts()
        .reindex(CLINICAL_ORDER)
        .fillna(0)
        .reset_index()
    )
    dist.columns = ["Nível", "Quantidade"]
    fig_dist = px.bar(
        dist,
        x="Quantidade",
        y="Nível",
        orientation="h",
        color="Quantidade",
        color_continuous_scale="Reds",
    )
    fig_dist.update_layout(
        yaxis=dict(categoryorder="array", categoryarray=CLINICAL_ORDER[::-1]),
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption("Quase metade da base já está em algum grau de obesidade — não é um problema de borda.")

with right:
    st.subheader("Fatores de risco: prevalência com × sem")
    risk_specs = [
        ("Histórico familiar", filtered["family_history"] == "yes"),
        ("Consumo frequente de calóricos", filtered["FAVC"] == "yes"),
        ("Sedentarismo (atividade ≤ 1x/sem)", filtered["FAF"].round() <= 1),
        ("Transporte ativo (a pé / bike)", filtered["MTRANS"].isin(["Walking", "Bike"])),
    ]
    rows = []
    for label, mask in risk_specs:
        rows.append(
            {
                "fator": label,
                "com": prevalence(filtered[mask]),
                "sem": prevalence(filtered[~mask]),
                "n_com": int(mask.sum()),
            }
        )
    rf = pd.DataFrame(rows).sort_values("com")

    fig_rf = go.Figure()
    for _, r in rf.iterrows():
        fig_rf.add_trace(
            go.Scatter(
                x=[r["sem"], r["com"]],
                y=[r["fator"], r["fator"]],
                mode="lines",
                line=dict(color="#c7c7c7", width=4),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig_rf.add_trace(
        go.Scatter(
            x=rf["sem"],
            y=rf["fator"],
            mode="markers",
            name="Sem o fator",
            marker=dict(color=COLOR_SAFE, size=13),
            hovertemplate="Sem o fator: %{x:.1f}%<extra></extra>",
        )
    )
    fig_rf.add_trace(
        go.Scatter(
            x=rf["com"],
            y=rf["fator"],
            mode="markers",
            name="Com o fator",
            marker=dict(color=COLOR_RISK, size=13),
            customdata=rf[["n_com"]],
            hovertemplate="Com o fator: %{x:.1f}%  (n=%{customdata[0]})<extra></extra>",
        )
    )
    fig_rf.update_layout(
        xaxis_title="Prevalência de obesidade (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_rf, use_container_width=True)
    st.caption("A distância entre os pontos é o tamanho do efeito: histórico familiar e dieta calórica dominam.")

low, lright = st.columns(2)

with low:
    st.subheader("Atividade física × obesidade")
    faf = filtered.assign(FAFr=filtered["FAF"].round().astype(int).clip(0, 3))
    faf_prev = (
        faf.groupby("FAFr")
        .agg(prev=("is_obese", lambda s: s.mean() * 100), n=("is_obese", "size"))
        .reindex([0, 1, 2, 3])
        .dropna()
        .reset_index()
    )
    faf_prev["Nível de atividade"] = faf_prev["FAFr"].map(FAF_LABELS)
    fig_faf = px.bar(
        faf_prev,
        x="Nível de atividade",
        y="prev",
        text=faf_prev["prev"].round(0).astype(int).astype(str) + "%",
        color="prev",
        color_continuous_scale="RdYlGn_r",
        labels={"prev": "Prevalência de obesidade (%)"},
    )
    fig_faf.update_traces(textposition="outside")
    fig_faf.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[0, 70])
    st.plotly_chart(fig_faf, use_container_width=True)
    st.caption("Gradiente claro: quanto mais atividade física, menor a prevalência — alvo direto de intervenção.")

with lright:
    st.subheader("Faixa etária × obesidade e IMC")
    aged = filtered.assign(faixa=pd.cut(filtered["Age"], AGE_BINS, labels=AGE_LABELS))
    age_prev = (
        aged.groupby("faixa", observed=True)
        .agg(prev=("is_obese", lambda s: s.mean() * 100), imc=("BMI", "mean"), n=("is_obese", "size"))
        .reset_index()
    )
    fig_age = make_subplots(specs=[[{"secondary_y": True}]])
    fig_age.add_trace(
        go.Bar(
            x=age_prev["faixa"],
            y=age_prev["prev"],
            name="Prevalência (%)",
            marker_color=COLOR_NEUTRAL,
            text=age_prev["prev"].round(0).astype(int).astype(str) + "%",
            textposition="outside",
        ),
        secondary_y=False,
    )
    fig_age.add_trace(
        go.Scatter(
            x=age_prev["faixa"],
            y=age_prev["imc"],
            name="IMC médio",
            mode="lines+markers",
            line=dict(color=COLOR_RISK, width=3),
        ),
        secondary_y=True,
    )
    fig_age.update_yaxes(title_text="Prevalência (%)", range=[0, 80], secondary_y=False)
    fig_age.update_yaxes(title_text="IMC médio", range=[20, 35], secondary_y=True)
    fig_age.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_age, use_container_width=True)
    st.caption("Salto na transição para a vida adulta: prevalência mais que dobra dos 14-20 para os 21-30 anos.")

st.divider()
st.subheader("Recomendações para a equipe médica")
st.markdown(
    """
- **Triagem por histórico familiar:** a prevalência de obesidade salta de ~2% (sem histórico) para ~56% (com histórico). É o filtro de triagem mais barato e mais discriminante disponível.
- **Dieta como alvo principal:** consumo frequente de alimentos calóricos separa ~8% de ~51% de prevalência — orientação nutricional tem alto potencial de impacto.
- **Atividade física é dose-dependente:** a prevalência cai de ~54% (sedentários) para ~21% (mais ativos). Programas de movimento devem ser estruturados por nível de atividade.
- **Janela preventiva dos 20+:** a prevalência mais que dobra na transição da adolescência para a vida adulta — momento ideal para ação preventiva precoce.
- **Mobilidade ativa:** quem usa transporte ativo (a pé/bicicleta) tem prevalência muito menor, reforçando políticas de incentivo ao deslocamento ativo.
"""
)
