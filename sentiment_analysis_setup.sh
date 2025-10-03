#!/usr/bin/env bash

# 🛠️ Скрипт установки анализа сентимента комментариев Habr (Docker + NVIDIA) 🛠️
#
# Этот Shell-скрипт автоматизирует подготовку системы Ubuntu для анализа
# сентимента русскоязычных комментариев с использованием модели RuBERT
# через Docker с ускорением на GPU от NVIDIA.
#
# Основные задачи:
# - Проверка системы: определяет дистрибутив и наличие драйверов NVIDIA.
# - Установка Docker: устанавливает Docker Engine и настраивает права.
# - Установка NVIDIA Container Toolkit: обеспечивает доступ к GPU из Docker.
# - Тестирование GPU в Docker: проверяет корректность настройки.
# - Создание рабочего пространства:
#   - Папка `habr_comments/` для входных JSON файлов.
#   - Глобальный кеш для моделей в `~/.cache/huggingface/`.
#   - Файл конфигурации `config.env` с настройками.
#
# Порядок использования:
# 1. Сделайте скрипт исполняемым: chmod +x sentiment_analysis_setup.sh
# 2. Запустите его: ./sentiment_analysis_setup.sh
# 3. После завершения может потребоваться перезагрузка системы.
#
# Репозиторий проекта всегда доступен тут: https://github.com/empenoso/comment-sentiment-analyzer
# 
# Автор: Михаил Шардин https://shardin.name/
# Дата создания: 04.10.2025
# Версия: 1.0
# 
# ===================================================================

## Строгий режим для bash
set -euo pipefail

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функции логирования
log()     { printf "${BLUE}[INFO]${NC} %s\n" "$1"; }
success() { printf "${GREEN}[SUCCESS]${NC} %s\n" "$1"; }
warning() { printf "${YELLOW}[WARNING]${NC} %s\n" "$1"; }
error()   { printf "${RED}[ERROR]${NC} %s\n" "$1" >&2; }

# --- Функции проверки системы ---

check_distro() {
    if ! [ -f /etc/os-release ]; then
        error "Не удалось определить операционную систему."
        exit 1
    fi
    . /etc/os-release
    if [[ "$ID" != "ubuntu" && "$ID" != "debian" ]]; then
        error "Этот скрипт предназначен для Ubuntu/Debian. Обнаружено: $PRETTY_NAME"
        exit 1
    fi
    success "Обнаружена совместимая система: $PRETTY_NAME"
}

check_gpu() {
    log "Проверка наличия NVIDIA GPU и драйверов..."
    if ! command -v nvidia-smi &> /dev/null; then
        warning "Команда 'nvidia-smi' не найдена."
        warning "Скрипт продолжит работу, но GPU не будет использоваться."
        warning "Для использования GPU установите драйверы NVIDIA:"
        printf "  sudo ubuntu-drivers autoinstall\n"
        printf "  sudo reboot\n"
        return 1
    fi
    if ! nvidia-smi &> /dev/null; then
        warning "'nvidia-smi' не отвечает. GPU будет недоступен."
        return 1
    fi
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits)
    success "Найден GPU: $GPU_INFO"
    log "Версия драйвера: $DRIVER_VERSION"
    return 0
}

# --- Функции установки компонентов ---

install_docker() {
    if command -v docker &> /dev/null && docker --version &> /dev/null; then
        success "Docker уже установлен: $(docker --version)"
    else
        log "Установка Docker Engine..."
        sudo apt-get update
        sudo apt-get install -y ca-certificates curl
        sudo install -m 0755 -d /etc/apt/keyrings
        sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
        sudo chmod a+r /etc/apt/keyrings/docker.asc

        echo \
          "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
          $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
          sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
        success "Docker успешно установлен."
    fi

    # Добавление пользователя в группу docker
    if ! groups "$USER" | grep -q '\bdocker\b'; then
        log "Добавление пользователя $USER в группу docker..."
        sudo usermod -aG docker "$USER"
        warning "Для применения изменений группы docker требуется перезагрузка или перелогин."
        log "Вы можете выполнить 'sudo reboot' после завершения установки."
    fi
}

install_nvidia_toolkit() {
    log "Установка NVIDIA Container Toolkit..."
    
    if command -v nvidia-ctk &> /dev/null; then
        success "NVIDIA Container Toolkit уже установлен."
    else
        log "Настройка репозитория NVIDIA..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
          && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
        
        log "Обновление списка пакетов и установка..."
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        success "NVIDIA Container Toolkit успешно установлен."
    fi

    log "Конфигурирование Docker для работы с NVIDIA GPU..."
    sudo nvidia-ctk runtime configure --runtime=docker
    
    log "Перезапуск Docker daemon..."
    sudo systemctl restart docker
    sleep 3
    success "Docker настроен для работы с NVIDIA GPU."
}

test_docker_gpu() {
    log "Тестирование Docker с поддержкой GPU..."
    if ! sudo docker run --rm hello-world > /dev/null 2>&1; then
        error "Базовый Docker не работает. Проверьте 'systemctl status docker'"
        exit 1
    fi
    success "Базовый тест Docker пройден."

    log "Проверка доступа к GPU из контейнера..."
    local cuda_image="nvidia/cuda:12.4.1-base-ubuntu22.04"
    log "Используем тестовый образ: $cuda_image"

    if ! sudo docker pull "$cuda_image" > /dev/null 2>&1; then
        warning "Не удалось загрузить тестовый образ. GPU будет недоступен."
        return 1
    fi

    local gpu_name_in_container
    gpu_name_in_container=$(sudo docker run --rm --gpus all "$cuda_image" nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) || true

    if [[ -n "$gpu_name_in_container" ]]; then
        success "🎉 GPU успешно обнаружен в Docker: $gpu_name_in_container"
        return 0
    else
        warning "GPU недоступен из Docker. Будет использоваться CPU."
        return 1
    fi
}

create_dockerfile() {
    log "Создание Dockerfile для анализа сентимента..."
    cat > Dockerfile << 'EOF'
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

EOF
    success "Dockerfile создан."
}

build_docker_image() {
    log "Сборка Docker образа для анализа сентимента..."
    if sudo docker build -t habr-sentiment-analyzer:latest .; then
        success "Docker образ собран успешно."
    else
        error "Не удалось собрать Docker образ."
        exit 1
    fi
}

setup_workspace() {
    log "Создание рабочих директорий и конфигурации..."
    local base_dir="."
    local output_dir="$base_dir/positive_comments"
    local cache_dir="$HOME/.cache/huggingface"
    
    # Создаем только папку для результатов и кеша
    mkdir -p "$output_dir"
    mkdir -p "$cache_dir"
    
    log "Установка прав 777 на папки..."
    sudo chmod -R 777 "$output_dir" "$cache_dir"
    
    success "Созданы директории:"
    printf "  📂 %s/  - для папок с комментариями (habr_comments, t-j_comments и т.д.)\n" "$(pwd)"
    printf "  📂 %s/  - для результатов\n" "$output_dir"
    printf "  🧠 %s          - для кеша моделей\n" "$cache_dir"

    local config_file="$base_dir/config.env"
    if [ -f "$config_file" ]; then
        success "Конфигурационный файл $config_file уже существует."
    else
        log "Создание конфигурационного файла: $config_file"
        cat > "$config_file" << 'EOF'
# Конфигурация анализатора тональности комментариев

# Модель для анализа (можно заменить на другую русскоязычную модель)
MODEL_NAME=cointegrated/rubert-tiny-sentiment-balanced

# Устройство для вычислений (cuda или cpu)
DEVICE=cuda

# Минимальный порог позитивности (0.0 - 1.0)
# Чем выше значение, тем строже фильтр
POSITIVE_THRESHOLD=0.6

# Исключить комментарии авторов (через запятую)
EXCLUDE_AUTHORS="empenoso,Михаил Шардин"

# Минимальная длина комментария в символах для анализа
MIN_COMMENT_LENGTH=20

# Максимальная длина комментария для обработки (в токенах)
MAX_TOKEN_LENGTH=512
EOF
        success "Конфигурационный файл создан: $config_file"
    fi
}

final_check() {
    log "Выполнение финальной проверки установки..."
    
    if ! command -v docker &>/dev/null; then error "Docker не найден!"; exit 1; fi
    if ! sudo docker image inspect "habr-sentiment-analyzer:latest" &>/dev/null; then 
        error "Docker образ не найден!"; exit 1
    fi
    # Проверяем только папку для результатов
    if ! [ -d "./positive_comments" ]; then error "Директория для результатов не найдена!"; exit 1; fi
    if ! [ -d "$HOME/.cache/huggingface" ]; then error "Директория кеша не найдена!"; exit 1; fi
    
    success "Все компоненты установлены и готовы к работе!"
}

show_usage() {
    printf "\n=====================================================================\n"
    printf "🎉 УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!\n"
    printf "=====================================================================\n\n"
    
    printf "🔥 СЛЕДУЮЩИЕ ШАГИ:\n\n"
    
    printf "1. 📁 Убедитесь, что ваши папки с JSON файлами находятся здесь:\n"
    printf "   - ./habr_comments/\n"
    printf "   - ./smart-lab_comments/\n"
    printf "   - ./t-j_comments/\n\n"
    
    printf "2. ⚙️  ${YELLOW}Настройте параметры${NC} (опционально):\n"
    printf "   nano ./config.env\n\n"

    if ! groups "$USER" | grep -q '\bdocker\b'; then
        printf "3. 🔄 ${YELLOW}Перезагрузите систему${NC} для применения прав Docker:\n"
        printf "   sudo reboot\n\n"
        printf "После перезагрузки:\n"
    fi
    
    printf "4. 🚀 Запустите анализ:\n"
    printf "   python3 sentiment_analyzer.py\n\n"
    
    printf "Рабочие директории:\n"
    printf "  📂 ./habr_comments/ и др.   - Входные JSON файлы\n"
    printf "  📂 ./positive_comments/     - Найденные позитивные комментарии\n"
    printf "  🧠 ~/.cache/huggingface/    - Кеш моделей\n"
    printf "  ⚙️  ./config.env             - Настройки\n\n"
    
    printf "Мониторинг GPU:\n"
    printf "  watch -n 5 nvidia-smi\n\n"
    
    printf "=====================================================================\n"
}

# --- Основная функция ---
main() {
    printf "=====================================================================\n"
    printf "💬 УСТАНОВКА АНАЛИЗАТОРА ТОНАЛЬНОСТИ КОММЕНТАРИЕВ\n"
    printf "🤖 RuBERT + Docker + NVIDIA GPU\n"
    printf "=====================================================================\n\n"
    
    check_distro
    
    local has_gpu=false
    if check_gpu; then
        has_gpu=true
    fi
    
    install_docker
    
    if [ "$has_gpu" = true ]; then
        install_nvidia_toolkit
        if test_docker_gpu; then
            log "GPU настроен и готов к использованию."
        else
            warning "GPU тест не пройден. Анализ будет выполняться на CPU."
        fi
    else
        log "Продолжаем без GPU. Анализ будет выполняться на CPU."
    fi

    create_dockerfile
    build_docker_image
    setup_workspace
    final_check
    show_usage
}

# Запуск основной функции
main