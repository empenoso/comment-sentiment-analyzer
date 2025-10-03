#!/usr/bin/env bash

set -euo pipefail

# Загружаем конфигурацию
if [ -f config.env ]; then
    source config.env
else
    echo "❌ Файл config.env не найден!"
    exit 1
fi

echo "🚀 Запуск анализатора тональности в Docker-контейнере..."

# Проверяем, какое устройство используется
if [ "${DEVICE:-cpu}" = "cuda" ]; then
    echo "Используется GPU для ускорения вычислений."
    docker run \
      --rm \
      --gpus all \
      -v "$(pwd)":/app \
      -v "$HOME/.cache/huggingface":"/root/.cache/huggingface" \
      --env-file config.env \
      habr-sentiment-analyzer:latest
else
    echo "Используется CPU для вычислений."
    docker run \
      --rm \
      -v "$(pwd)":/app \
      -v "$HOME/.cache/huggingface":"/root/.cache/huggingface" \
      --env-file config.env \
      habr-sentiment-analyzer:latest
fi