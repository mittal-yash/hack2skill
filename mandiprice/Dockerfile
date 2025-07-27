FROM python:3.11-slim

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --require-hashes -r requirements.txt

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=8080

# ⚠️ Do NOT hardcode sensitive credentials like this in production!
# Use Secret Manager or pass OPENAI_API_KEY via Cloud Build or runtime env
ENV OPENAI_API_KEY=API_KEY

EXPOSE 8080

CMD ["python", "app.py"]
