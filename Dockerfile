# syntax=docker/dockerfile:1.2
FROM python:latest
# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos necessários para o diretório de trabalho
COPY ./challenge /app
COPY ./data /app/data
COPY ./tests /app/tests

# Instala as dependências necessárias
RUN pip install uvicorn
RUN make install
RUN make model-test
RUN make api-test

# Informa ao Docker que a aplicação escuta na porta 8000
EXPOSE 8000

# O comando para rodar a aplicação quando o container iniciar
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]