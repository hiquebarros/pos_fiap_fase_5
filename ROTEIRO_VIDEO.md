# Roteiro do vídeo — Tech Challenge Obesidade

> Duração alvo: **6 a 8 minutos** (dentro do intervalo FIAP: 4–10 min)  
> Tom: visão de **negócio para equipe médica**, não só técnica  
> Gravação sugerida: tela (Streamlit + notebook) + narração em português

---

## Como entregar o dashboard analítico (leia antes de gravar)

Você **já tem** o dashboard implementado. Ele não é um sistema separado — fica na **mesma app Streamlit**, na página **Dashboard**.

| O quê | Onde está no projeto |
|---|---|
| App preditiva | `streamlit/app.py` (página inicial) |
| Dashboard analítico | `streamlit/pages/1_Dashboard.py` |

### Opção A — Gravar e apresentar localmente (mais rápido hoje)

```bash
# Terminal 1 — API
cd api && MODEL_DIR=../models python app.py

# Terminal 2 — Streamlit (predição + dashboard)
cd streamlit && API_URL=http://localhost:5000 streamlit run app.py
```

Abra **http://localhost:8501**  
No menu lateral esquerdo do Streamlit aparecem:
- **app** (ou nome da página principal) → predição  
- **Dashboard** → painel analítico  

**No vídeo:** mostre o menu lateral trocando de página. Isso conta como “sistema preditivo + dashboard”.

### Opção B — Entregar na FIAP com links públicos (Streamlit Cloud)

1. Suba o repositório no **GitHub** (público).
2. Garanta que `models/obesity_model.joblib` e `models/metrics.json` estão commitados.
3. Acesse [share.streamlit.io](https://share.streamlit.io/) → **New app**.
4. **Main file path:** `streamlit/app.py`
5. **Requirements:** use o `requirements.txt` da raiz (ou aponte para `streamlit/requirements.txt` + dependências da API se necessário).

**Importante para Cloud:** o Streamlit na nuvem **não sobe o Docker Compose inteiro**. Duas alternativas:

| Alternativa | Como funciona |
|---|---|
| **B1 — Só Streamlit (recomendado para prazo)** | Alterar temporariamente o Streamlit para carregar o modelo localmente (sem API). *Se ainda estiver só via API, precisa de deploy da API (Render/Railway) ou ajuste rápido no código.* |
| **B2 — Streamlit + API na nuvem** | API no Render (grátis) + Streamlit Cloud com `API_URL` apontando para a API |

**Links para `entrega/links.txt` (mesmo deploy multipage):**

```text
App Preditiva: https://SEU-APP.streamlit.app/
Dashboard Analítico: https://SEU-APP.streamlit.app/Dashboard
Repositório GitHub: https://github.com/SEU-USUARIO/SEU-REPO
```

A FIAP pede dois links — usar a **mesma URL base** com a página `/Dashboard` é aceitável na prática (são duas “visões” da mesma aplicação).

### Opção C — Demonstrar com Docker (impressiona na gravação)

```bash
docker compose up --build
```

- Streamlit: http://localhost:8501  
- Dashboard: menu lateral → **Dashboard**

---

## Estrutura do vídeo (5 blocos do SPEC)

| Bloco | Tempo | SPEC |
|---|---|---|
| 1. Contexto de negócio | ~1 min | Problema no hospital |
| 2. Estratégia de dados | ~1,5 min | EDA, features, modelo |
| 3. Sistema preditivo | ~2 min | Demo Streamlit predição |
| 4. Dashboard analítico | ~2 min | Demo Dashboard + insights |
| 5. Conclusão | ~1 min | Limitações, valor, links |

---

## Roteiro falado (script)

### [0:00 – 1:00] 1. Contexto de negócio

**Tela:** slide simples ou você na câmera (opcional).

**Fala sugerida:**

> “Olá. Este é o Tech Challenge de Machine Learning da FIAP. O cenário é um hospital que precisa apoiar médicos e médicas na identificação precoce de risco de obesidade.
>
> A obesidade é multifatorial — genética, hábitos, sedentarismo, alimentação. Nosso objetivo não é substituir o médico, e sim oferecer uma **ferramenta de apoio à decisão**: a partir de dados do paciente e de hábitos de vida, estimamos o **nível de obesidade** e entregamos uma **visão analítica** da população para orientar prevenção e triagem.”

**Palavras-chave:** apoio à decisão, prevenção, equipe médica.

---

### [1:00 – 2:30] 2. Estratégia de dados e modelo

**Tela:** `notebooks/obesity_pipeline.ipynb` ou terminal com `models/metrics.json` aberto.

**Fala sugerida:**

> “Usamos o dataset `obesity.csv`, com cerca de 2.100 registros e variáveis como idade, altura, peso, histórico familiar, alimentação, atividade física e meio de transporte. O dicionário de variáveis está documentado em `data/dicionario.txt`.
>
> Na pipeline, tratamos escalas com arredondamento — por exemplo consumo de vegetais e água — e criamos o IMC. Treinamos **duas versões do modelo**: uma **com IMC explícito** e outra **sem a coluna IMC**, para evitar que o modelo apenas ‘copie’ a tabela de IMC em vez de aprender hábitos.
>
> O modelo de **produção** ficou na variante **sem IMC**, com **cerca de 96% de acurácia** e F1 macro acima de 95%, bem acima da meta de 75% exigida. A versão com IMC chega a ~99% e fica documentada como comparação.
>
> A arquitetura segue o padrão da disciplina: **treino** → **API Flask** → **Streamlit** que consome a API na predição.”

**Mostrar na tela (10–15 s cada):**
- Distribuição da variável `Obesity` (notebook)
- Trecho do `metrics.json`: `production_variant`, `accuracy`, `variants`

---

### [2:30 – 4:30] 3. Sistema preditivo (demo)

**Tela:** Streamlit — página principal (`http://localhost:8501` ou link Cloud).

**Passo a passo na gravação:**

1. Mostrar o banner com métricas do modelo em produção.
2. Preencher um **caso exemplo** (sugestão abaixo).
3. Clicar em **“Prever nível de obesidade”**.
4. Mostrar **classe predita**, **IMC** e **probabilidades por classe**.
5. Ler o **disclaimer** (“não substitui avaliação médica”).

**Caso exemplo para demo:**

| Campo | Valor |
|---|---|
| Gênero | Masculino |
| Idade | 32 |
| Altura | 1,75 m |
| Peso | 95 kg |
| Histórico familiar | Sim |
| Alimentos calóricos | Sim |
| Atividade física | 1 (baixa) |
| Transporte | Carro |

**Fala sugerida:**

> “Na interface, a equipe insere os dados do paciente de forma simples. Ao prever, o sistema consulta a API, que carrega o modelo treinado, e devolve o nível de obesidade em linguagem clara, com probabilidades — útil para priorizar consulta nutricional ou avaliação mais detalhada, sempre como **apoio**, não diagnóstico automático.”

---

### [4:30 – 6:30] 4. Dashboard analítico (demo)

**Tela:** Mesma app Streamlit → menu lateral → **Dashboard**.

**Como mostrar que é o “painel analítico”:**

> “O painel analítico está na mesma aplicação, na página Dashboard — é a visão para gestão e equipe médica olharem a **população**, não um paciente isolado.”

**Passo a passo na gravação:**

1. Abrir **Dashboard** no menu lateral.
2. Mostrar os **4 KPIs** no topo (registros, prevalência, IMC médio, histórico familiar).
3. Usar **filtros** na barra lateral (gênero, idade, transporte).
4. Comentar **2 gráficos** com insight de negócio, por exemplo:
   - Distribuição dos níveis de obesidade
   - Obesidade × histórico familiar
   - IMC por nível ou atividade física
5. Ler **1 ou 2 bullets** da seção “Recomendações para a equipe médica”.

**Fala sugerida (insights):**

> “No dashboard, vemos que há concentração relevante nos níveis de sobrepeso e obesidade. Pacientes com histórico familiar positivo aparecem com maior prevalência nos níveis mais altos — isso sugere **triagem prioritária** e orientação nutricional.
>
> Também observamos relação entre menor atividade física e perfis com IMC mais elevado, o que reforça programas de **movimento e acompanhamento de hábitos** no hospital.”

---

### [6:30 – 7:30] 5. Conclusão

**Tela:** slide de encerramento ou README do GitHub.

**Fala sugerida:**

> “Entregamos: pipeline de ML com feature engineering, modelo acima de 75% de acurácia, API de predição, aplicação Streamlit e dashboard analítico integrado.
>
> **Limitações:** dados sintéticos/tabular, sem integração com prontuário eletrônico; modelo depende da qualidade do preenchimento; variante sem IMC é mais realista, mas ainda usa altura e peso.
>
> **Próximos passos:** integrar ao prontuário, monitorar drift do modelo, validação com profissionais de saúde.
>
> Links: app preditiva, dashboard na mesma URL, e repositório GitHub. Obrigado.”

**Slide final — texto sugerido:**

```text
Tech Challenge — Predição de Obesidade
✓ Pipeline ML + modelo > 75%
✓ API Flask + Streamlit
✓ Dashboard analítico (página Dashboard)

Links:
• App: [URL Streamlit]
• Dashboard: [URL]/Dashboard
• GitHub: [URL repo]
```

---

## Checklist antes de gravar

- [ ] `docker compose up` **ou** API + Streamlit locais rodando sem erro
- [ ] Predição de teste funcionando (botão retorna classe)
- [ ] Página **Dashboard** abre e gráficos carregam
- [ ] `models/metrics.json` com números atualizados (pós `train/train.py`)
- [ ] Microfone e resolução da tela OK (1920×1080 se possível)
- [ ] Fechar notificações / abas irrelevantes

---

## Dicas de gravação

- Grave em **um take** por bloco se errar — edita depois.
- Use zoom no navegador (110%) para o avaliador enxergar textos.
- Fale devagar nos insights clínicos — é o que diferencia nota alta.
- **Não** passe mais de 2 min só em código; priorize Streamlit.

---

## Se a banca perguntar “cadê o link do dashboard?”

Resposta pronta:

> “O dashboard analítico é a segunda página da aplicação Streamlit (`/Dashboard`), separada da tela de predição individual. São duas visões do mesmo produto: uma para **paciente** (inferência) e outra para **população** (análise e KPIs).”

---

## Próximo passo prático (hoje)

1. Subir stack: `docker compose up --build`  
2. Ensaiar o roteiro uma vez (8 min cronometrados)  
3. Gravar  
4. Preencher `entrega/links.txt` com URLs reais após deploy (se for entregar online)
