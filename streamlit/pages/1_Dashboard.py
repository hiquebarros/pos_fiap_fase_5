import plotly.express as px
import streamlit as st

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


@st.cache_data
def get_data():
    return load_data()


df = get_data()
df["ObesityLabel"] = df["Obesity"].map(OBESITY_LABELS)

st.title("Dashboard analítico — Obesidade")
st.caption("Visão de negócio para apoiar triagem, prevenção e orientação da equipe médica.")

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
]

obesity_classes = [c for c in df["Obesity"].unique() if c.startswith("Obesity_")]
obesity_pct = filtered["Obesity"].isin(obesity_classes).mean() * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros", len(filtered))
col2.metric("Prevalência de obesidade", f"{obesity_pct:.1f}%")
col3.metric("IMC médio", f"{filtered['BMI'].mean():.1f}")
col4.metric("Histórico familiar (+)", f"{(filtered['family_history'] == 'yes').mean() * 100:.1f}%")

left, right = st.columns(2)

with left:
    fig_dist = px.bar(
        filtered["ObesityLabel"].value_counts().reset_index(),
        x="count",
        y="ObesityLabel",
        orientation="h",
        title="Distribuição dos níveis de obesidade",
        labels={"ObesityLabel": "Nível", "count": "Quantidade"},
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    fig_family = px.histogram(
        filtered,
        x="ObesityLabel",
        color="family_history",
        barmode="group",
        title="Obesidade × histórico familiar",
        labels={"ObesityLabel": "Nível", "family_history": "Histórico familiar"},
    )
    st.plotly_chart(fig_family, use_container_width=True)

with right:
    fig_bmi = px.box(
        filtered,
        x="ObesityLabel",
        y="BMI",
        title="IMC por nível de obesidade",
        labels={"ObesityLabel": "Nível", "BMI": "IMC"},
    )
    st.plotly_chart(fig_bmi, use_container_width=True)

    fig_faf = px.box(
        filtered,
        x="ObesityLabel",
        y="FAF",
        title="Atividade física × nível de obesidade",
        labels={"ObesityLabel": "Nível", "FAF": "Atividade física (0-3)"},
    )
    st.plotly_chart(fig_faf, use_container_width=True)

fig_corr = px.imshow(
    filtered[["Age", "Height", "Weight", "BMI", "FCVC", "NCP", "CH2O", "FAF", "TUE"]].corr(),
    text_auto=".2f",
    title="Correlação entre variáveis numéricas",
    color_continuous_scale="RdBu_r",
    aspect="auto",
)
st.plotly_chart(fig_corr, use_container_width=True)

st.subheader("Recomendações para a equipe médica")
st.markdown(
    """
- **Triagem prioritária:** pacientes com histórico familiar positivo e consumo frequente de alimentos calóricos concentram-se nos níveis de sobrepeso e obesidade.
- **Intervenção comportamental:** baixa atividade física (`FAF` ≤ 1) e alto uso de dispositivos eletrônicos (`TUE` ≥ 1) aparecem com IMC mais elevado na base analisada.
- **Hábitos alimentares:** aumentar consumo de vegetais (`FCVC`) e reduzir lanches entre refeições (`CAEC`) são alvos claros de orientação nutricional.
- **Mobilidade ativa:** modos de transporte ativo (a pé, bicicleta) associam-se a perfis com menor prevalência de obesidade severa.
- **Monitoramento contínuo:** combinar IMC, hábitos e histórico familiar permite estratificar risco antes da progressão para obesidade tipo II/III.
"""
)
