# Usa una imagen base de Python oficial
FROM python:3.11-slim

# Actualiza los paquetes e instala LibreOffice y un paquete de fuentes completo (Noto)
RUN apt-get update && apt-get install -y \
    libreoffice \
    fonts-noto \
    && rm -rf /var/lib/apt/lists/*

# Copia tu código y requerimientos al contenedor
COPY requirements.txt .
COPY main.py .

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Configura el framework para ejecutar tu código
ENV PORT 8080
CMD ["functions-framework", "--target=process_document_event", "--port=8080"]