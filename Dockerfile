# Озвучка книги — контейнер для бесплатного хостинга
# (Render, Fly.io, Koyeb, Google Cloud Run, Oracle Always Free и др.)
FROM python:3.11-slim

# ffmpeg нужен для обработки голоса «Вера» и склейки аудио
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

# Хостинг передаёт порт через переменную PORT; локально по умолчанию 7860.
ENV PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
