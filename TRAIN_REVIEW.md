# Revisão técnica — `train/train.py`

Data: 2026-05-26  
Arquivo analisado: `train/train.py` (com dependência direta de `train/preprocessing.py`)  
Objetivo: avaliar qualidade do script de treino, riscos, e sugerir melhorias (você decide depois o que aplicar).

---

## 1. Resumo executivo

- **Status geral**: melhorias do review **aplicadas** em `train/train.py` e `train/preprocessing.py`.
- **Duas variantes treinadas**: `with_bmi` e `no_bmi`, com métricas comparativas em `models/metrics.json`.
- **Produção**: variant **`no_bmi`** (evita atalho por IMC) — acurácia ~**95,74%**, F1 macro ~**95,62%** (acima da meta de 75%).
- **Baseline com BMI**: acurácia ~**98,82%** (documentada para comparação no vídeo/entrega).

---

## 2. O que o script faz hoje (fluxo)

### 2.1 Entrada / features / split

- Carrega dados via `load_data()` (que aplica `clean_data()`).
- Monta \(X, y\) via `get_feature_target(df)`.
- Faz `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`.

### 2.2 Modelos candidatos

- `RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample")`
- `XGBClassifier(...)` (se disponível), marcado como “needs_encoding”.

### 2.3 Avaliação e escolha

- Treina cada candidato 1 vez (sem CV).
- Mede **accuracy no test set** e escolhe o melhor.
- Se `best_accuracy < 0.75`, roda `RandomizedSearchCV` (apenas para RF).

### 2.4 Artefatos gerados

No diretório `MODEL_DIR` (default `models/`):

- `obesity_model.joblib`: pipeline completo (preprocessador + classificador)
- `metrics.json`: accuracy + relatório + matriz de confusão + tamanhos de split
- `label_encoder.joblib`: **só se** o melhor modelo exigir encoding (ex.: XGBoost)

---

## 3. Pontos fortes (mantém)

### 3.1 Pragmático e reprodutível

- `random_state` fixo no split e nos modelos.
- A saída é “self-contained”: um `.joblib` com pipeline (bom para deploy).

### 3.2 Docker-friendly

- Uso de `MODEL_DIR` via variável de ambiente.
- O script cria `MODELS_DIR` caso não exista.

### 3.3 Métricas úteis

Salvar `classification_report` + `confusion_matrix` no `metrics.json` ajuda bastante no dashboard e no vídeo.

---

## 4. Problemas e riscos (com severidade)

### P0 — Risco de “atalho” / vazamento via `BMI` (IMC)

No `train/preprocessing.py`, `BMI` é calculado e entra no conjunto de features:

- `BMI = Weight / Height²`
- `NUMERIC_FEATURES` inclui `"BMI"` junto de `Height` e `Weight`

**Por que isso é um problema?**

Em muitos datasets de obesidade, o rótulo `Obesity` é definido diretamente por faixas de IMC, ou então é extremamente correlacionado. Nesse caso, o modelo aprende uma regra quase determinística, e a acurácia vai para o teto.

**Como isso se manifesta**:

- Acurácia **muito alta** para 7 classes (98%).
- Erros concentrados em classes adjacentes (fronteiras de IMC).

**Impacto**:

- Do ponto de vista acadêmico/negócio, fica fácil argumentar que o modelo “só reproduz a tabela de IMC”, e não usa hábitos.

**Mitigação recomendada (opções)**:

- Treinar e reportar **dois modelos**:
  - **Modelo A**: com BMI (alta acurácia, baseline forte)
  - **Modelo B**: sem BMI (mais “realista” para hábitos; ainda deve passar > 75%)
- Alternativa: manter BMI, mas **remover `Weight` e `Height`** (reduz redundância, mas não elimina o atalho).

> Para apresentação: mostrar os dois resultados costuma “ganhar pontos” por honestidade metodológica.

---

### P1 — Seleção por uma única partição (sem validação cruzada)

Hoje, a seleção do melhor candidato usa **apenas 1 split treino/teste**.  
Com dataset de ~2k linhas, isso até “funciona”, mas não mede variabilidade nem estabilidade.

**Mitigação recomendada**:

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Usar média de `f1_macro` e `accuracy` no CV para comparar candidatos.
- Reservar o test set só para “final score”.

---

### P1 — Otimização (tuning) só acontece quando o modelo é ruim

O `RandomizedSearchCV` só roda se `best_accuracy < 0.75`.  
Como o modelo já está muito acima de 75%, essa parte **nunca executa**.

**Mitigação recomendada**:

- Rodar tuning leve **sempre** no melhor candidato (poucos parâmetros, poucos `n_iter`).

---

### P2 — Métrica de decisão só por `accuracy`

Embora as classes estejam razoavelmente balanceadas, `accuracy` pode esconder performance desigual.

**Mitigação recomendada**:

- Critério: `f1_macro` (principal) + `accuracy` (secundário)
- Persistir ambas no `metrics.json`.

---

### P2 — Falta de registro do desempenho dos “perdedores”

O `metrics.json` só guarda o vencedor. Para auditoria, comparações e narrativa, é útil guardar:

- score de cada candidato
- tempo de treino aproximado

---

### P3 — Pequenas melhorias de engenharia

- `RandomForestClassifier` poderia usar `n_jobs=-1`
- `best_accuracy` como `0.0` é ok; `-inf` é mais semântico
- Falta de `trained_at` (timestamp) no `metrics.json` dificulta rastreio em volume Docker

---

## 5. Sugestões de mudanças (menu para você decidir)

### Pacote A — Resolver o P0 (IMC/atalho) e reforçar narrativa

**O que fazer**

- Treinar **duas variantes** e salvar como artefatos separados:
  - `obesity_model_with_bmi.joblib`
  - `obesity_model_no_bmi.joblib`
- Salvar `metrics_with_bmi.json` e `metrics_no_bmi.json`
- No final, escolher qual vira o “produção” (ou manter “produção” como o que melhor atende o enunciado).

**Prós**

- Defesa acadêmica forte; evita críticas sobre “modelo trivial”.

**Contras**

- Mais arquivos / mais explicação.

---

### Pacote B — Validação cruzada + escolha por `f1_macro`

**O que fazer**

- Avaliar candidatos com CV (5 folds estratificados) no treino.
- Usar CV para decidir o “melhor”.
- Rodar test set uma única vez no final.

**Prós**

- Métrica mais robusta.

**Contras**

- Treino um pouco mais lento (mas ainda ok para esse dataset).

---

### Pacote C — Logging e rastreabilidade

**O que fazer**

- Incluir `trained_at` no `metrics.json`
- Guardar scores de todos os candidatos em `metrics.json["candidates"]`
- Trocar prints por `logging` (nível INFO)

**Prós**

- Melhor auditabilidade.

**Contras**

- Mais verboso.

---

### Pacote D — Pequenos ajustes de performance e limpeza

- `RandomForestClassifier(..., n_jobs=-1)`
- Extrair função helper para reduzir duplicação do bloco “fit → predict → score → report”
- Setar `np.random.seed(random_state)` se quiser consistência em qualquer parte “não sklearn”

---

## 6. Recomendações objetivas (se você quiser o “mínimo”)

Se você quiser **manter simples** e ainda ficar tecnicamente bem defendido:

1. **Rodar e reportar a variante sem BMI** (além da com BMI), mesmo que você entregue a com BMI.
2. Salvar no `metrics.json` também `f1_macro` e `f1_weighted`.
3. Adicionar `trained_at` e `n_jobs=-1` no RF.

---

## 7. Checklist de decisão rápida

- [ ] Quero um modelo “melhor métrica” (aceito BMI) **ou** um modelo “mais realista” (sem BMI)?
- [ ] Vou mostrar no vídeo a comparação “com BMI vs sem BMI”?
- [ ] Preciso de CV (mais robusto) ou “um split” é suficiente pro prazo?

---

## 8. Referências no repositório

- Script: `train/train.py`
- Preprocessamento/feature engineering: `train/preprocessing.py`
- Métricas atuais: `models/metrics.json`

