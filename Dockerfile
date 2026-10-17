FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/srv/.hf
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt \
 && python -m spacy download en_core_web_sm
COPY src src
COPY app app
COPY data/sample_evidence.json data/
# Download the public models at build time so the container starts offline.
RUN python -c "from app.pipeline import _load; from app import config; _load(config.CROSS_MODEL); _load(config.NLI_MODEL)"
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
