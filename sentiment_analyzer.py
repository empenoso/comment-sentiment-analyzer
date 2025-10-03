#!/usr/bin/env python3
"""
Анализатор тональности комментариев
======================================

Скрипт анализирует JSON файлы из нескольких директорий, находит позитивные
комментарии, используя модель RuBERT, и сохраняет их в единые текстовые файлы
для каждой исходной директории.

Репозиторий проекта всегда доступен тут: https://github.com/empenoso/comment-sentiment-analyzer

Автор: Михаил Шардин https://shardin.name/
Дата создания: 04.10.2025
Версия: 1.0
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, DefaultDict
from collections import defaultdict

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError as e:
    print(f"❌ Ошибка импорта библиотек: {e}", file=sys.stderr)
    print("Убедитесь, что скрипт запущен в Docker контейнере или установите зависимости:", file=sys.stderr)
    print("pip install transformers sentencepiece torch", file=sys.stderr)
    sys.exit(1)

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

class Config:
    """Конфигурация из переменных окружения и файла config.env"""

    def __init__(self):
        self.load_env_file()

        self.MODEL_NAME = os.getenv('MODEL_NAME', 'cointegrated/rubert-tiny-sentiment-balanced')
        self.DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.POSITIVE_THRESHOLD = float(os.getenv('POSITIVE_THRESHOLD', '1.0'))
        
        exclude_str = os.getenv('EXCLUDE_AUTHORS', 'empenoso,Михаил Шардин')
        exclude_str = exclude_str.strip('"').strip("'")
        self.EXCLUDE_AUTHORS: set[str] = {author.strip() for author in exclude_str.split(',') if author.strip()}
        
        self.MIN_COMMENT_LENGTH = int(os.getenv('MIN_COMMENT_LENGTH', '20'))
        self.MAX_TOKEN_LENGTH = int(os.getenv('MAX_TOKEN_LENGTH', '512'))

        self.INPUT_DIRS: List[Path] = [
            Path('./habr_comments'),
            Path('./smart-lab_comments'),
            Path('./t-j_comments')
        ]
        self.OUTPUT_DIR = Path('./positive_comments') 
        self.CACHE_DIR = Path.home() / '.cache' / 'huggingface'

    def load_env_file(self):
        """Загрузка переменных из config.env"""
        env_file = Path('./config.env')
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        value = value.strip().strip('"').strip("'")
                        os.environ[key.strip()] = value

    def print_config(self):
        """Вывод текущей конфигурации"""
        print("\n" + "="*70)
        print("⚙️  КОНФИГУРАЦИЯ")
        print("="*70)
        print(f"🤖 Модель:              {self.MODEL_NAME}")
        print(f"🖥️  Устройство:          {self.DEVICE}")
        print(f"📊 Порог позитивности:  {self.POSITIVE_THRESHOLD}")
        print(f"👤 Исключить авторов:   {list(self.EXCLUDE_AUTHORS)}")
        print(f"📏 Мин. длина:          {self.MIN_COMMENT_LENGTH} символов")
        print(f"📂 Входные папки:       {[str(d) for d in self.INPUT_DIRS]}")
        print("="*70 + "\n")

# ============================================================================
# АНАЛИЗАТОР ТОНАЛЬНОСТИ
# ============================================================================

class SentimentAnalyzer:
    """Анализатор тональности на основе RuBERT"""
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """Загрузка модели и токенизатора"""
        print(f"📥 Загрузка модели {self.config.MODEL_NAME}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME, cache_dir=self.config.CACHE_DIR)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.config.MODEL_NAME, cache_dir=self.config.CACHE_DIR)
            
            if self.config.DEVICE == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')
                print(f"✅ Модель загружена на GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.config.DEVICE = 'cpu'
                print("✅ Модель загружена на CPU")

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}", file=sys.stderr)
            sys.exit(1)

    def is_positive_comment(self, text: str, author: str) -> bool:
        """Проверка, является ли комментарий позитивным"""
        if author in self.config.EXCLUDE_AUTHORS:
            return False
            
        if len(text) < self.config.MIN_COMMENT_LENGTH:
            return False

        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=self.config.MAX_TOKEN_LENGTH).to(self.model.device)
            proba = torch.sigmoid(self.model(**inputs).logits).cpu().numpy()[0]

        label = self.model.config.id2label[proba.argmax()]
        
        if label == 'positive' and proba[2] >= self.config.POSITIVE_THRESHOLD:
            return True
        
        praise_keywords = [
            'спасибо', 'благодарю', 'отлично', 'прекрасно', 'замечательно', 'великолепно',
            'хорошо', 'молодец', 'браво', 'супер', 'класс', 'круто', 'восхищен', 'впечатлен',
            'полезно', 'помогло', 'отличная статья', 'хорошая работа'
        ]
        if label == 'neutral' and any(keyword in text.lower() for keyword in praise_keywords):
            return True
            
        return False

# ============================================================================
# ОБРАБОТКА ФАЙЛОВ
# ============================================================================

def get_comment_url(directory_name: str, article_url: str, comment_id: str) -> str:
    """Формирование URL комментария в зависимости от источника"""
    if 'habr' in directory_name.lower():
        # Для Habr извлекаем ID статьи из URL
        if 'habr.com' in article_url:
            article_id = article_url.split('/articles/')[-1].split('/')[0]
            return f"https://habr.com/ru/articles/{article_id}/comments/#comment_{comment_id}"
        return f"Unknown Habr URL: {article_url}"
    
    elif 'smart-lab' in directory_name.lower():
        # Для Smart-Lab извлекаем ID из URL
        if 'smart-lab.ru' in article_url:
            article_id = article_url.split('/blog/')[-1].split('.php')[0]
            return f"https://smart-lab.ru/blog/{article_id}.php#comment{comment_id}"
        return f"Unknown Smart-Lab URL: {article_url}"
    
    elif 't-j' in directory_name.lower() or 'tj' in directory_name.lower():
        # ИСПРАВЛЕНО: для TJ используем оригинальный URL + #c{comment_id}
        if article_url:
            # Убираем возможный финальный слэш и якорь
            base_url = article_url.rstrip('/').split('#')[0]
            return f"{base_url}/#c{comment_id}"
        return f"Unknown TJ URL (comment: {comment_id})"
    
    else:
        return f"Unknown source (url: {article_url}, comment: {comment_id})"

def process_json_file(file_path: Path, analyzer: SentimentAnalyzer, directory_name: str) -> List[Dict]:
    """Обработка одного JSON файла"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"\n⚠️  Ошибка чтения {file_path.name}: {e}")
        return []

    # ИСПРАВЛЕНО: сохраняем полный URL статьи
    article_url = data.get('url', '')
    
    comments = data.get('comments', [])
    positive_comments = []

    for comment in comments:
        text = comment.get('text', '')
        author = comment.get('author', '')
        
        if not text or not author:
            continue

        if analyzer.is_positive_comment(text, author):
            comment['article_url'] = article_url  # Сохраняем полный URL
            comment['directory_name'] = directory_name
            positive_comments.append(comment)
            
    return positive_comments

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Основная логика скрипта"""
    print("\n" + "="*70)
    print("💬 АНАЛИЗАТОР ТОНАЛЬНОСТИ КОММЕНТАРИЕВ")
    print("="*70)

    config = Config()
    config.print_config()
    
    grouped_positive_comments: DefaultDict[Path, List] = defaultdict(list)
    total_files_to_process = 0
    
    # --- Сбор и фильтрация файлов ---
    for directory in config.INPUT_DIRS:
        if directory.exists() and directory.is_dir():
            found = list(directory.glob('*.json'))
            if found:
                print(f"🔍 В '{directory}' найдено {len(found)} json-файлов.")
                total_files_to_process += len(found)
            else:
                print(f"ℹ️  В '{directory}' json-файлы не найдены.")
        else:
            print(f"⚠️  Директория '{directory}' не найдена, пропускаем.")

    if total_files_to_process == 0:
        print("\n❌ В указанных директориях не найдено ни одного JSON файла!", file=sys.stderr)
        sys.exit(1)
        
    print(f"\n➡️  Всего для обработки: {total_files_to_process} файлов.")

    analyzer = SentimentAnalyzer(config)

    print("\n" + "="*70)
    print("🚀 НАЧАЛО АНАЛИЗА")
    print("="*70 + "\n")

    # --- Основной цикл обработки ---
    processed_files = 0
    total_positive = 0
    for directory in config.INPUT_DIRS:
        if not directory.is_dir():
            continue
        
        for file_path in directory.glob('*.json'):
            processed_files += 1
            print(f"\r[{processed_files}/{total_files_to_process}] Обработка {file_path.name}...", end='')
            
            positive_comments = process_json_file(file_path, analyzer, directory.name)
            
            if positive_comments:
                grouped_positive_comments[directory].extend(positive_comments)
                total_positive += len(positive_comments)
    
    # --- Сохранение результатов ---
    print("\n\n" + "="*70)
    print("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*70)
    
    if not grouped_positive_comments:
         print("ℹ️  Позитивных комментариев для сохранения не найдено.")
    else:
        for directory, comments in grouped_positive_comments.items():
            output_file = directory / f"{directory.name}_positive_comments.txt"
            print(f"📝 Сохраняю {len(comments)} комментариев в файл: {output_file}")
            
            # Сортируем комментарии по ID для консистентности
            comments.sort(key=lambda c: c.get('id') if c.get('id') is not None else 0)

            with open(output_file, 'w', encoding='utf-8') as f:
                for idx, comment in enumerate(comments):
                    author = comment.get('author', 'Неизвестный автор')
                    datetime = comment.get('datetime', 'Дата неизвестна')
                    text = comment.get('text', '')
                    comment_id = comment.get('id', 'unknown')
                    article_url = comment.get('article_url', '')  # Используем полный URL
                    directory_name = comment.get('directory_name', '')
                    
                    # ИСПРАВЛЕНО: передаем article_url вместо article_id
                    comment_url = get_comment_url(directory_name, article_url, str(comment_id))
                    
                    # Красивое форматирование
                    f.write(f"Автор: {author}\n")
                    f.write(f"Дата: {datetime}\n")
                    f.write(f"Текст: {text}\n")
                    f.write(f"Ссылка: {comment_url}\n")
                    
                    # Разделитель между комментариями (100 знаков '=')
                    if idx < len(comments) - 1:
                        f.write("=" * 100 + "\n")

    print("\n" + "="*70)
    print("📊 СТАТИСТИКА")
    print("="*70)
    print(f"Обработано файлов:          {processed_files}")
    print(f"Найдено позитивных:         {total_positive}")
    
    if total_positive > 0:
        print(f"✅ Анализ завершен! Результаты сохранены в исходных директориях.")
    else:
        print("ℹ️  Позитивных комментариев не найдено. Попробуйте уменьшить `POSITIVE_THRESHOLD` в `config.env`.")
    print("="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Процесс прерван пользователем.", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)