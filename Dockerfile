# Usa uma imagem Python base
FROM python:3.8-slim

# Instala dependências necessárias, incluindo libGL
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libgl1-mesa-glx

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos para o contêiner
COPY . .

# Instala as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta da aplicação
EXPOSE 8000

# Comando para rodar o Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "--timeout", "6000", "app:app"]