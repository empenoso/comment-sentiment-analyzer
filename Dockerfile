FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Устанавливаем переменные окружения для предотвращения интерактивных запросов
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Устанавливаем Python и зависимости
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip
RUN pip3 install --upgrade pip

# Устанавливаем PyTorch с поддержкой CUDA 12.4
# Для CPU будет использоваться CPU версия автоматически
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Устанавливаем HuggingFace Transformers и зависимости
RUN pip3 install --no-cache-dir \
    transformers==4.46.0 \
    sentencepiece==0.2.0

WORKDIR /app

# Копируем скрипт анализатора
COPY sentiment_analyzer.py /app/

CMD ["python3", "sentiment_analyzer.py"]

