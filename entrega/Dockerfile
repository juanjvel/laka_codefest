# Imagen para ejecutar generador.py (Etapa 1, CODEFEST AD ASTRA 2026) de forma
# reproducible. El índice FAISS + metadata (entrega/base_vectorial/) y el
# archivo de consultas NO se copian a la imagen -- son datos, se montan como
# volumen en tiempo de ejecución (ver README de esta carpeta / docker-compose).
FROM python:3.12-slim

WORKDIR /app

# Dependencias del sistema para faiss-cpu/torch (libgomp) y descarga de datos NLTK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Datos de NLTK para la tokenización de oraciones (dividir_en_subfragmentos).
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

COPY generador.py .

ENTRYPOINT ["python", "generador.py"]
CMD ["--indice", "base_vectorial/encoder_multilingual-e5-small", \
     "--consultas", "consultas.jsonl", \
     "--salida", "salida/resultados.jsonl"]
