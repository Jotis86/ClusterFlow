# Usar imagen oficial de Python
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de requisitos
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar toda la aplicación
COPY app/ .

# Crear directorio para datos
RUN mkdir -p /app/data

# Exponer puerto de Streamlit
EXPOSE 8501

# Configurar variables de entorno de Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Comando para ejecutar la aplicación refactorizada
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
