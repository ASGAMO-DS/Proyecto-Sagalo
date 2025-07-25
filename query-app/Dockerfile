# Use una imagen base de Python oficial
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo de requerimientos e instalar dependencias
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY app.py .

# Exponer el puerto que usa Streamlit
EXPOSE 8080

# Variables de entorno para Streamlit
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true

# Comando para correr la aplicación
CMD ["streamlit", "run", "app.py"]