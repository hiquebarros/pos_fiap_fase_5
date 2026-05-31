FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py preprocessing.py ./

ENV MODEL_DIR=/model_data
ENV DATA_PATH=/app/data/obesity.csv

CMD ["python", "train.py"]
