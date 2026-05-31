FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py preprocessing.py predictor.py start.sh ./
COPY pages/ pages/

RUN chmod +x start.sh

ENV DATA_PATH=/app/data/obesity.csv
ENV API_URL=http://api:5000

EXPOSE 8501

CMD ["sh", "start.sh"]
