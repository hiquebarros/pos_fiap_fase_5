FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py preprocessing.py ./

ENV MODEL_DIR=/model_data

EXPOSE 5000

CMD ["python", "app.py"]
