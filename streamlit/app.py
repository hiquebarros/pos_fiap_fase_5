import os

import pandas as pd
import requests
import streamlit as st

from predictor import get_metrics_local, predict_local

API_URL = os.environ.get("API_URL", "").strip()

st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="🏥",
    layout="wide",
)


def obesity_label(value: str) -> str:
    labels = {
        "Insufficient_Weight": "Peso insuficiente",
        "Normal_Weight": "Peso normal",
        "Overweight_Level_I": "Sobrepeso nível I",
        "Overweight_Level_II": "Sobrepeso nível II",
        "Obesity_Type_I": "Obesidade tipo I",
        "Obesity_Type_II": "Obesidade tipo II",
        "Obesity_Type_III": "Obesidade tipo III",
    }
    return labels.get(value, value)


@st.cache_data(ttl=60)
def fetch_metrics() -> dict:
    if not API_URL:
        return get_metrics_local()
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {}


def run_prediction(payload: dict) -> dict:
    if not API_URL:
        return predict_local(payload)
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


metrics = fetch_metrics()

st.title("Sistema preditivo de obesidade")
st.caption("Apoio à decisão clínica com base em hábitos, antropometria e histórico familiar.")

if metrics:
    variant = metrics.get("production_variant", "N/A")
    st.info(
        f"Modelo em produção: **{metrics.get('production_model', metrics.get('model', 'N/A'))}** "
        f"({variant}) · Acurácia: **{metrics.get('accuracy', 0):.1%}** · "
        f"F1 macro: **{metrics.get('f1_macro', 0):.1%}**"
    )
    if metrics.get("variants"):
        with st.expander("Comparativo com BMI vs sem BMI"):
            for key, data in metrics["variants"].items():
                st.write(
                    f"**{key}**: acc {data.get('accuracy', 0):.1%} · "
                    f"f1_macro {data.get('f1_macro', 0):.1%} · modelo {data.get('model')}"
                )

st.warning(
    "Ferramenta de apoio à decisão. Não substitui avaliação médica, exames laboratoriais ou diagnóstico profissional."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dados do paciente")
    gender = st.selectbox("Gênero", ["Female", "Male"], format_func=lambda x: "Feminino" if x == "Female" else "Masculino")
    age = st.number_input("Idade (anos)", min_value=14, max_value=61, value=25)
    height = st.number_input("Altura (m)", min_value=1.45, max_value=1.98, value=1.70, step=0.01)
    weight = st.number_input("Peso (kg)", min_value=39.0, max_value=173.0, value=75.0, step=0.1)
    family_history = st.selectbox("Histórico familiar de excesso de peso", ["no", "yes"], format_func=lambda x: "Não" if x == "no" else "Sim")
    favc = st.selectbox("Consome alimentos calóricos com frequência?", ["no", "yes"], format_func=lambda x: "Não" if x == "no" else "Sim")
    smoke = st.selectbox("Fuma?", ["no", "yes"], format_func=lambda x: "Não" if x == "no" else "Sim")
    scc = st.selectbox("Monitora ingestão calórica?", ["no", "yes"], format_func=lambda x: "Não" if x == "no" else "Sim")

with col2:
    st.subheader("Hábitos e estilo de vida")
    fcvc = st.slider("Frequência de consumo de vegetais (1=raro, 3=sempre)", 1, 3, 2)
    ncp = st.slider("Refeições principais por dia", 1, 4, 3)
    caec = st.selectbox("Come entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
    ch2o = st.slider("Consumo diário de água (1=<1L, 2=1-2L, 3=>2L)", 1, 3, 2)
    faf = st.slider("Atividade física semanal (0=nenhuma, 3=5x+/sem)", 0, 3, 1)
    tue = st.slider("Uso de dispositivos eletrônicos (0=0-2h, 1=3-5h, 2=>5h)", 0, 2, 1)
    calc = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox(
        "Meio de transporte habitual",
        ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"],
        format_func=lambda x: {
            "Automobile": "Carro",
            "Motorbike": "Moto",
            "Bike": "Bicicleta",
            "Public_Transportation": "Transporte público",
            "Walking": "A pé",
        }[x],
    )

bmi = weight / (height**2)

if st.button("Prever nível de obesidade", type="primary", use_container_width=True):
    payload = {
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans,
    }

    try:
        result = run_prediction(payload)
        prediction = result["prediction"]

        st.success(f"Nível predito: **{obesity_label(prediction)}**")
        st.metric("IMC calculado", f"{result.get('bmi', bmi):.1f}")

        st.subheader("Probabilidades por classe")
        probabilities = pd.DataFrame(result["probabilities"])
        probabilities["label"] = probabilities["class"].map(obesity_label)
        st.dataframe(
            probabilities[["label", "probability"]].assign(
                probability=lambda df: (df["probability"] * 100).round(1).astype(str) + "%"
            ),
            hide_index=True,
            use_container_width=True,
        )
    except requests.RequestException as exc:
        st.error(f"Erro ao consultar API de predição ({API_URL}): {exc}")
    except Exception as exc:
        st.error(f"Erro ao gerar predição: {exc}")

else:
    st.metric("IMC calculado (prévia)", f"{bmi:.1f}")
