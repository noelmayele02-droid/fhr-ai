FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

# Entraîne le modèle réel au build (données déjà incluses dans data/real/CTG.csv)
RUN python models/train_real.py --epochs 150

# Par défaut : lance l'API d'inférence. Voir docker-compose.yml pour le dashboard.
CMD ["uvicorn", "serve.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
