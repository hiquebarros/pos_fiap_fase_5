#!/usr/bin/env sh
set -e

# Render/Railway injetam a porta via $PORT; localmente cai em 8501.
PORT="${PORT:-8501}"

exec streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false
