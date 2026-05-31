# Tech Challenge FIAP — Predição de Obesidade

Sistema de apoio à decisão clínica para estimar o nível de obesidade com base em dados antropométricos, hábitos alimentares e estilo de vida.

## Estrutura (padrão da aula)

```text
pos-tech/
├── data/
│   ├── obesity.csv
│   └── dicionario.txt
├── train/                  # treinamento do modelo
│   ├── Dockerfile
│   ├── train.py
│   ├── preprocessing.py
│   └── requirements.txt
├── api/                    # API REST de predição (Flask)
│   ├── Dockerfile
│   ├── app.py
│   ├── preprocessing.py
│   └── requirements.txt
├── streamlit/              # interface preditiva + dashboard
│   ├── Dockerfile
│   ├── app.py
│   ├── pages/1_Dashboard.py
│   ├── preprocessing.py
│   └── requirements.txt
├── models/                 # artefatos locais (dev)
├── notebooks/
├── docker-compose.yml
└── requirements.txt        # dependências para dev local
```

## Fluxo Docker

```text
trainer  →  model_volume (/model_data)
                ↓
api      →  Flask :5000  (POST /predict)
                ↓
streamlit → :8501  (consome a API + dashboard)
```

## Docker

```bash
docker compose up --build
```

| Serviço    | URL                      |
|------------|--------------------------|
| Streamlit  | http://localhost:8501    |
| API        | http://localhost:5000    |

Endpoints da API:
- `GET /health`
- `GET /metrics`
- `POST /predict`

## Desenvolvimento local (sem Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Treinar
cd train && python train.py && cd ..

# 2. API (terminal 1)
cd api && MODEL_DIR=../models python app.py

# 3. Streamlit (terminal 2)
cd streamlit && API_URL=http://localhost:5000 streamlit run app.py
```

## Deploy no Render

O app Streamlit é **autossuficiente**: sem a variável `API_URL`, ele carrega o modelo
diretamente de `models/` (versionado no repo). Não precisa subir a API Flask separada.

### Opção A — Blueprint (recomendado)

1. Suba o repositório no GitHub.
2. No Render: **New → Blueprint** e aponte para o repo. Ele lê o `render.yaml` e cria o serviço.

### Opção B — Web Service manual

Em **New → Web Service**, com o repo conectado, preencha:

| Campo | Valor |
|---|---|
| Language | `Python 3` |
| Root Directory | `streamlit` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `sh start.sh` |

O `start.sh` escuta na porta dinâmica `$PORT` exigida pelo Render. O dashboard fica em
`/Dashboard` na mesma URL.

## Métricas

Consulte `models/metrics.json` após o treinamento.
