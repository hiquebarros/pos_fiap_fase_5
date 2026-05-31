# SPEC — Tech Challenge: Predição de Obesidade

| Campo | Valor |
|---|---|
| **Projeto** | Tech Challenge — Fase (Machine Learning) |
| **Versão** | 1.0 |
| **Data** | 2026-05-26 |
| **Status** | Em execução — entrega hoje |
| **Peso na nota** | 90% da nota final |
| **Formato** | Individual (com apoio de agente de IA) |

---

## 1. Contexto

Você foi contratado como **cientista de dados** de um hospital. O desafio é desenvolver um **modelo de Machine Learning** para auxiliar médicos e médicas a **prever se uma pessoa pode ter obesidade**, apoiando a tomada de decisão clínica.

A obesidade é uma condição médica caracterizada pelo acúmulo excessivo de gordura corporal, a ponto de prejudicar a saúde. É um problema cada vez mais prevalente globalmente, com causas multifatoriais: fatores genéticos, ambientais e comportamentais.

---

## 2. Problema

A equipe médica precisa de uma solução que:

1. **Prediga** o nível de obesidade com base em dados demográficos, hábitos alimentares, estilo de vida e histórico familiar.
2. **Explique** padrões relevantes nos dados para apoiar diagnóstico e prevenção.
3. **Esteja disponível** em uma aplicação acessível (Streamlit) para uso em contexto de decisão.

---

## 3. Objetivos

| ID | Objetivo | Métrica de sucesso |
|---|---|---|
| O1 | Construir pipeline completa de ML | Feature engineering + treinamento documentados e reproduzíveis |
| O2 | Modelo preditivo performático | **Assertividade (acurácia) > 75%** |
| O3 | Sistema preditivo em produção | App Streamlit deployada e funcional |
| O4 | Visão analítica para equipe médica | Painel com insights acionáveis sobre obesidade |
| O5 | Entrega formal do projeto | Links + repositório + vídeo de apresentação |

---

## 4. Escopo

### 4.1 Dentro do escopo

- Análise exploratória dos dados (`obesity.csv`)
- Feature engineering e preparação dos dados
- Treinamento, validação e seleção de modelo
- Deploy do modelo em aplicação Streamlit (predição)
- Dashboard analítico com insights para equipe médica
- Documentação e entrega na plataforma FIAP

### 4.2 Fora do escopo

- Integração com prontuário eletrônico hospitalar (EHR)
- Diagnóstico médico definitivo (o sistema é **auxiliar à decisão**, não substituto)
- Coleta de novos dados além do dataset fornecido

---

## 5. Dados

### 5.1 Fonte

- **Arquivo:** `obesity.csv`
- **Coluna alvo:** `Obesity` (nível de obesidade — no enunciado aparece como `Obesity_level`)
- **Volume:** ~2.111 registros, 17 colunas
- **Classes alvo (7):** `Insufficient_Weight`, `Normal_Weight`, `Overweight_Level_I`, `Overweight_Level_II`, `Obesity_Type_I`, `Obesity_Type_II`, `Obesity_Type_III`

### 5.2 Dicionário de dados

| Coluna | Descrição | Tipo esperado |
|---|---|---|
| `Gender` | Gênero | Categórica |
| `Age` | Idade | Numérica |
| `Height` | Altura (metros) | Numérica |
| `Weight` | Peso (kg) | Numérica |
| `family_history` | Algum membro da família sofreu ou sofre de excesso de peso? | Categórica (Sim/Não) |
| `FAVC` | Você come alimentos altamente calóricos com frequência? | Categórica |
| `FCVC` | Você costuma comer vegetais nas suas refeições? | Numérica/Categórica |
| `NCP` | Quantas refeições principais você faz diariamente? | Numérica |
| `CAEC` | Você come alguma coisa entre as refeições? | Categórica |
| `SMOKE` | Você fuma? | Categórica (Sim/Não) |
| `CH2O` | Quanta água você bebe diariamente? | Numérica |
| `SCC` | Você monitora as calorias que ingere diariamente? | Categórica (Sim/Não) |
| `FAF` | Com que frequência você pratica atividade física? | Numérica |
| `TUE` | Quanto tempo você usa dispositivos tecnológicos (celular, videogame, TV, computador etc.) — no enunciado: `TER` | Numérica |
| `CALC` | Com que frequência você bebe álcool? | Categórica |
| `MTRANS` | Qual meio de transporte você costuma usar? | Categórica |
| **`Obesity`** | **Nível de obesidade (target)** | **Categórica (7 classes)** |

### 5.3 Premissas sobre os dados

- Dataset fornecido é representativo para treinamento e validação
- Valores ausentes, outliers e inconsistências devem ser tratados na pipeline
- Variáveis derivadas (ex.: IMC = Weight / Height²) podem ser criadas no feature engineering

---

## 6. Requisitos funcionais

### RF01 — Pipeline de Machine Learning

- Demonstrar **todas as etapas** da pipeline:
  - Carregamento e inspeção dos dados
  - Análise exploratória (EDA)
  - Tratamento de missing values e outliers
  - Feature engineering (encoding, scaling, features derivadas)
  - Split treino/teste (e validação cruzada, se aplicável)
  - Treinamento de um ou mais modelos
  - Avaliação e comparação de métricas
  - Seleção e persistência do modelo final (ex.: `.pkl`, `.joblib`)

### RF02 — Modelo preditivo

- Prever `Obesity` a partir das features de entrada
- Acurácia no conjunto de teste **superior a 75%**
- Métricas complementares recomendadas: precision, recall, F1 (macro/weighted), matriz de confusão

### RF03 — Aplicação preditiva (Streamlit)

- Interface para entrada dos dados do paciente/usuário
- Botão ou fluxo para executar predição
- Exibição clara do **nível de obesidade predito**
- Mensagens de erro para entradas inválidas
- Aplicação **deployada** no Streamlit Cloud (ou equivalente)

### RF04 — Painel analítico

- Dashboard com **insights de negócio** para equipe médica, por exemplo:
  - Distribuição dos níveis de obesidade
  - Relação entre IMC, idade, hábitos alimentares e nível de obesidade
  - Fatores de risco mais frequentes
  - Comparativos por gênero, transporte, atividade física etc.
- Linguagem orientada a **decisão clínica/preventiva**, não apenas técnica

### RF05 — Entrega e documentação

- Repositório GitHub com código completo
- Arquivo `.doc` ou `.txt` contendo:
  - Link da aplicação Streamlit (predição)
  - Link do painel analítico
  - Link do repositório GitHub
- Vídeo de apresentação (4–10 min) cobrindo estratégia, sistema preditivo e dashboard em visão de negócio

---

## 7. Requisitos não funcionais

| ID | Requisito |
|---|---|
| RNF01 | Código versionado em Git com README explicando setup e execução |
| RNF02 | Pipeline reproduzível (requirements.txt ou equivalente) |
| RNF03 | Interface Streamlit intuitiva para usuários não técnicos |
| RNF04 | Tempo de resposta da predição aceitável (< 5s em condições normais) |
| RNF05 | Apresentação em vídeo com foco em valor para equipe médica |

---

## 8. Arquitetura proposta

```text
obesity.csv
    │
    ▼
[EDA + Feature Engineering]
    │
    ▼
[Treinamento / Validação / Seleção de Modelo]
    │
    ├──► modelo serializado (.pkl / .joblib)
    │
    ├──► Streamlit App — Predição
    │         (formulário → inferência → resultado)
    │
    └──► Streamlit App — Dashboard Analítico
              (gráficos, KPIs, insights)
```

### Stack sugerida

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.10+ |
| Dados | pandas, numpy |
| ML | scikit-learn (baseline), opcional: XGBoost/LightGBM |
| Visualização | plotly, matplotlib, seaborn |
| App / Deploy | Streamlit + Streamlit Community Cloud |
| Versionamento | Git + GitHub |

---

## 9. Critérios de aceite

- [ ] Pipeline de ML completa documentada (notebook e/ou scripts)
- [ ] Feature engineering implementado e justificado
- [ ] Modelo treinado com **acurácia > 75%**
- [ ] App Streamlit de predição deployada e acessível via link público
- [ ] Painel analítico deployado e acessível via link público
- [ ] Repositório GitHub público com todo o código
- [ ] Arquivo `.doc`/`.txt` com os 3 links para upload na plataforma
- [ ] Vídeo (4–10 min) apresentando estratégia, predição e dashboard em visão de negócio

---

## 10. Entregáveis

| # | Entregável | Formato |
|---|---|---|
| E1 | Pipeline de ML | Notebook (`.ipynb`) e/ou scripts (`.py`) |
| E2 | Modelo treinado | Arquivo serializado + métricas de avaliação |
| E3 | App preditiva | Streamlit deployada |
| E4 | Dashboard analítico | Streamlit (ou página separada) deployada |
| E5 | Repositório | GitHub com README |
| E6 | Links consolidados | `.doc` ou `.txt` |
| E7 | Apresentação | Vídeo 4–10 min |

---

## 11. Estrutura sugerida do repositório

```text
pos-tech/
├── data/
│   ├── obesity.csv
│   └── dicionario.txt
├── train/                  # treinamento (Docker + local)
├── api/                    # API Flask de predição
├── streamlit/              # app preditiva + dashboard
├── models/                 # artefatos locais
├── notebooks/
├── docker-compose.yml
├── requirements.txt
├── README.md
├── SPEC.md
└── IMPLEMENTATION_PLAN.md
```

---

## 12. Plano de implementação

Plano operacional para **entrega hoje**, execução **solo** com apoio de agente de IA: **[IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)**.

| Bloco | Atividade | Tempo | Saída |
|---|---|---|---|
| — | `data/obesity.csv` | ✓ feito | Dataset no lugar |
| 1 | Setup restante | ~1h | venv, pastas, GitHub |
| 2 | Pipeline ML | ~2h | Notebook + modelo > 75% |
| 3 | Streamlit | ~2h | Predição + dashboard (multipage) |
| 4 | Deploy + docs + vídeo | ~2–4h | Links, README, gravação |

**Total estimado:** 8–10 horas hoje.

---

## 13. Riscos e mitigações

| Risco | Impacto | Mitigação |
|---|---|---|
| Acurácia abaixo de 75% | Reprovação do critério | Testar múltiplos algoritmos, tuning de hiperparâmetros, melhorar features |
| Overfitting | Modelo não generaliza | Validação cruzada, hold-out test, regularização |
| Deploy indisponível | Entrega incompleta | Testar deploy cedo; usar Streamlit Cloud com requirements fixos |
| Dashboard sem valor clínico | Nota reduzida na apresentação | Focar insights acionáveis para equipe médica |

---

## 14. Apresentação em vídeo (4–10 min)

O vídeo deve cobrir:

1. **Contexto de negócio** — problema da obesidade no hospital
2. **Estratégia de dados** — EDA, features criadas, escolha do modelo
3. **Sistema preditivo** — demo da app Streamlit (entrada → predição → interpretação)
4. **Dashboard analítico** — principais insights para decisão médica
5. **Conclusão** — limitações, próximos passos e valor entregue

---

## 15. Checklist de submissão (plataforma FIAP)

```text
[ ] Link app Streamlit (predição)
[ ] Link painel analítico
[ ] Link repositório GitHub
[ ] Arquivo .doc ou .txt com os links acima
[ ] Vídeo gravado e disponibilizado (conforme orientação da disciplina)
[ ] Entrega dentro do prazo
```

---

## 16. Referências

- Dataset: `obesity.csv` (fornecido no Tech Challenge)
- Documentação Streamlit: https://docs.streamlit.io/
- Scikit-learn: https://scikit-learn.org/
