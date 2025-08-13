# 📋 План превращения VantageVeoConverter в Serverless контейнер на fal.ai

## 1. Архитектура решения

Ваше приложение будет преобразовано из Gradio веб-приложения в serverless API на fal.ai с следующей структурой:

```
VantageVeoConverter (Gradio) → fal.App (Container) → Serverless API
```

## 2. Основные требования fal.ai

- ✅ **Контейнерная архитектура**: Использование Docker для упаковки всех зависимостей
- ✅ **GPU поддержка**: Автоматическое масштабирование на GPU (H100, A100, etc.)
- ✅ **Pydantic модели**: Для input/output валидации
- ✅ **Persistent storage**: `/data` volume для моделей и временных файлов
- ✅ **Autoscaling**: От 0 до N реплик с keep_alive настройками

## 3. План реализации

### Шаг 1: Подготовка окружения

```bash
# Установка fal SDK
pip install --upgrade fal

# Аутентификация (требуется enterprise доступ)
fal auth login
```

### Шаг 2: Рефакторинг архитектуры

Создать новую структуру файлов:
```
fal_vantage/
├── fal_app.py          # Основной fal.App класс
├── Dockerfile          # Контейнер с всеми зависимостями
├── models.py           # Pydantic модели для IO
├── src/                # Ваша существующая логика
└── requirements.txt    # Зависимости
```

### Шаг 3: Создание fal.App класса

```python
# fal_app.py
import fal
from fal.container import ContainerImage
from fal.toolkit import File, Video, Audio
from pydantic import BaseModel, Field
from pathlib import Path

class VideoSyncInput(BaseModel):
    video_url: str = Field(description="URL видео для синхронизации")
    audio_url: str = Field(description="URL целевого аудио")
    use_rife: bool = Field(default=True, description="Использовать RIFE интерполяцию")
    diagnostic_mode: bool = Field(default=False, description="Режим диагностики")

class VideoSyncOutput(BaseModel):
    synchronized_video: Video = Field(description="Синхронизированное видео")
    diagnostic_video: Video = Field(description="Диагностическое видео", default=None)
    timecodes: File = Field(description="Файл таймкодов", default=None)

class VantageVeoConverter(
    fal.App,
    kind="container",
    image=ContainerImage.from_dockerfile("./Dockerfile"),
    keep_alive=300,  # 5 минут между запросами
    machine_type="GPU-A100",  # или "GPU-H100" для больших моделей
    min_concurrency=0,
    max_concurrency=5
):
    def setup(self):
        """Инициализация моделей при старте контейнера"""
        import torch
        import whisper
        from src.comfy_rife import ComfyRIFE
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Загрузка моделей в persistent storage
        self.whisper_model = whisper.load_model("base", device=self.device)
        self.rife_model = ComfyRIFE(self.device)
        
        # Прогрев моделей
        self.warmup()
    
    def warmup(self):
        """Прогрев моделей для оптимизации"""
        # Тестовый запуск для кеширования
        pass
    
    @fal.endpoint("/")
    def synchronize_video(self, input: VideoSyncInput) -> VideoSyncOutput:
        """Основной endpoint для синхронизации"""
        from fal.toolkit import download_file
        from src.audio_sync import synchronize_audio_video
        from src.ai_freeze_repair import repair_freezes_with_rife
        
        # Скачивание файлов
        video_path = download_file(input.video_url)
        audio_path = download_file(input.audio_url)
        
        # Синхронизация
        result = synchronize_audio_video(
            video_path, 
            audio_path,
            use_rife=input.use_rife,
            whisper_model=self.whisper_model,
            rife_model=self.rife_model
        )
        
        # Возврат результатов
        return VideoSyncOutput(
            synchronized_video=Video.from_path(result["video"]),
            timecodes=File.from_path(result["timecodes"]) if result.get("timecodes") else None
        )
```

### Шаг 4: Dockerfile

```dockerfile
FROM python:3.11-slim

# Системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libespeak-dev \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Установка mp4fpsmod
RUN git clone https://github.com/nu774/mp4fpsmod.git /tmp/mp4fpsmod && \
    cd /tmp/mp4fpsmod && \
    ./bootstrap.sh && \
    ./configure && \
    make && \
    make install && \
    rm -rf /tmp/mp4fpsmod

# Python зависимости
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY src/ ./src/
COPY weights/ ./weights/
COPY fal_app.py models.py ./

# Использование persistent storage для моделей
ENV HF_HOME=/data/huggingface
ENV TORCH_HOME=/data/torch
```

### Шаг 5: Модификация существующего кода

Основные изменения:
1. **Убрать Gradio интерфейс** - заменить на Pydantic модели
2. **Использовать `/data` для persistent storage** моделей
3. **Добавить fal.toolkit** для работы с файлами (upload/download)
4. **Разделить на endpoints** - отдельные функции для разных режимов

### Шаг 6: Деплой

```bash
# Локальное тестирование
fal run fal_app.py::VantageVeoConverter

# Деплой в production
fal deploy fal_app.py::VantageVeoConverter --auth=private
```

## 4. Преимущества миграции на fal.ai

- 🚀 **Автомасштабирование**: От 0 до тысяч GPU автоматически
- 💰 **Pay-per-use**: Платите только за использованное время
- 🔧 **Нет инфраструктуры**: Не нужно управлять серверами
- 📊 **Мониторинг**: Встроенная observability
- 🌐 **REST API**: Автоматически генерируется API
- 🎮 **Playground**: Веб-интерфейс для тестирования

## 5. Ключевые изменения в коде

1. **app_rife_compact.py** → разделить на:
   - `fal_app.py` - основной класс приложения
   - `endpoints.py` - отдельные функции для каждого режима

2. **Gradio интерфейс** → заменить на:
   - Pydantic модели для валидации
   - REST API endpoints

3. **Файловая система** → использовать:
   - `fal.toolkit.download_file()` для входных файлов
   - `fal.toolkit.Video/File` для выходных файлов
   - `/data` для persistent storage

4. **Инициализация моделей** → в методе `setup()`:
   - Загрузка один раз при старте контейнера
   - Использование keep_alive для сохранения в памяти

## 6. Детальная структура проекта

### 6.1 Файловая структура после миграции

```
VantageVeoConverter-fal/
├── fal_app.py              # Основной класс приложения fal.App
├── models.py               # Pydantic модели для input/output
├── endpoints.py            # Дополнительные endpoints
├── Dockerfile             # Контейнер с зависимостями
├── requirements-fal.txt   # Python зависимости для fal
├── .env                   # Переменные окружения
├── src/                   # Существующая бизнес-логика
│   ├── __init__.py
│   ├── ai_freeze_repair.py
│   ├── audio_sync.py
│   ├── binary_utils.py
│   ├── comfy_rife.py
│   ├── physical_retime.py
│   ├── timecode_freeze_predictor.py
│   ├── timing_analyzer.py
│   └── triple_diagnostic.py
├── weights/               # Веса моделей (опционально)
└── tests/                # Тесты
    └── test_endpoints.py
```

### 6.2 Pydantic модели (models.py)

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class RIFEMode(str, Enum):
    OFF = "off"
    ADAPTIVE = "adaptive"
    PRECISION = "precision"
    MAXIMUM = "maximum"

class VideoSyncInput(BaseModel):
    video_url: str = Field(
        description="URL или путь к видео файлу для синхронизации"
    )
    audio_url: str = Field(
        description="URL или путь к целевому аудио файлу"
    )
    rife_mode: RIFEMode = Field(
        default=RIFEMode.ADAPTIVE,
        description="Режим RIFE интерполяции"
    )
    diagnostic_mode: bool = Field(
        default=False,
        description="Включить режим диагностики с визуализацией"
    )
    
class VideoSyncOutput(BaseModel):
    synchronized_video_url: str = Field(
        description="URL синхронизированного видео"
    )
    diagnostic_video_url: Optional[str] = Field(
        default=None,
        description="URL диагностического видео (если включен режим диагностики)"
    )
    timecodes_url: Optional[str] = Field(
        default=None,
        description="URL файла с таймкодами"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Метаданные обработки"
    )
```

### 6.3 Endpoints (endpoints.py)

```python
import fal
from models import VideoSyncInput, VideoSyncOutput, RIFEMode
from fal.toolkit import Video, File, download_file

class VantageEndpoints:
    
    @fal.endpoint("/sync")
    def sync_video(self, input: VideoSyncInput) -> VideoSyncOutput:
        """Основной endpoint для синхронизации видео с аудио"""
        # Основная логика синхронизации
        pass
    
    @fal.endpoint("/diagnostic")
    def diagnostic_analysis(self, input: VideoSyncInput) -> VideoSyncOutput:
        """Диагностический анализ с визуализацией проблем"""
        # Логика диагностики
        pass
    
    @fal.endpoint("/batch")
    def batch_process(self, inputs: List[VideoSyncInput]) -> List[VideoSyncOutput]:
        """Пакетная обработка нескольких видео"""
        # Логика пакетной обработки
        pass
```

## 7. Оптимизации для fal.ai

### 7.1 Использование persistent storage

```python
from pathlib import Path
from fal.toolkit import FAL_PERSISTENT_DIR

class VantageVeoConverter(fal.App):
    def setup(self):
        # Путь к persistent хранилищу
        self.models_dir = FAL_PERSISTENT_DIR / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Проверка наличия моделей
        whisper_path = self.models_dir / "whisper_base.pt"
        if not whisper_path.exists():
            # Загрузка модели при первом запуске
            self.download_models()
```

### 7.2 Кеширование и оптимизация

```python
# Использование BuildKit cache в Dockerfile
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Использование fal optimize для моделей
from fal.toolkit import optimize

self.whisper_model = optimize(whisper.load_model("base"))
```

### 7.3 Streaming для больших файлов

```python
@fal.realtime("/stream")
def stream_processing(self, input: VideoSyncInput) -> Generator:
    """Streaming обработка для больших видео"""
    for progress in process_video_chunks(input):
        yield {"progress": progress, "status": "processing"}
```

## 8. Команды для работы

### 8.1 Разработка

```bash
# Установка зависимостей
pip install fal pydantic

# Локальный запуск
fal run fal_app.py::VantageVeoConverter

# Тестирование endpoint
curl $FAL_RUN_URL/sync \
  -H 'content-type: application/json' \
  -d '{"video_url": "path/to/video.mp4", "audio_url": "path/to/audio.wav"}'
```

### 8.2 Деплой

```bash
# Деплой с приватным доступом
fal deploy fal_app.py::VantageVeoConverter --auth=private

# Обновление существующего деплоя
fal deploy --strategy=rolling

# Масштабирование
fal apps scale vantage-veo-converter \
  --min-concurrency=0 \
  --max-concurrency=10 \
  --keep-alive=600
```

### 8.3 Мониторинг

```bash
# Список деплоев
fal apps list

# Просмотр логов
fal apps logs vantage-veo-converter

# Удаление деплоя
fal apps delete vantage-veo-converter
```

## 9. Интеграция с клиентами

### 9.1 Python клиент

```python
import fal_client

# Синхронный вызов
result = fal_client.run(
    "your-username/vantage-veo-converter",
    arguments={
        "video_url": "https://example.com/video.mp4",
        "audio_url": "https://example.com/audio.wav",
        "rife_mode": "adaptive"
    }
)

# Асинхронный с прогрессом
handler = fal_client.submit(
    "your-username/vantage-veo-converter",
    arguments={...}
)

for event in handler.iter_events(with_logs=True):
    if isinstance(event, fal_client.InProgress):
        print(f"Progress: {event.logs}")
```

### 9.2 JavaScript клиент

```javascript
import { fal } from "@fal-ai/client";

const result = await fal.subscribe("your-username/vantage-veo-converter", {
  input: {
    video_url: "https://example.com/video.mp4",
    audio_url: "https://example.com/audio.wav",
    rife_mode: "adaptive"
  }
});
```

## 10. Стоимость и производительность

### 10.1 Примерная стоимость

- **GPU-A100**: ~$1.89/час
- **GPU-H100**: ~$2.49/час
- **Хранилище**: $0.10/GB/месяц
- **Сеть**: $0.12/GB исходящий трафик

### 10.2 Оптимизация затрат

- Использовать `keep_alive` для частых запросов
- Настроить `min_concurrency=0` для scale-to-zero
- Использовать persistent storage для моделей
- Оптимизировать размер Docker образа

## 11. Следующие шаги

### Неделя 1: Подготовка
- [ ] Получить enterprise доступ к fal.ai
- [ ] Настроить локальное окружение
- [ ] Создать базовую структуру проекта

### Неделя 2: Реализация
- [ ] Портировать основную логику в fal.App
- [ ] Создать Dockerfile с зависимостями
- [ ] Написать Pydantic модели
- [ ] Реализовать endpoints

### Неделя 3: Тестирование и деплой
- [ ] Локальное тестирование
- [ ] Оптимизация производительности
- [ ] Деплой в production
- [ ] Настройка мониторинга

## 12. Риски и митигация

### Риски:
1. **Доступ к enterprise**: Может потребоваться время на одобрение
2. **Совместимость зависимостей**: Некоторые библиотеки могут требовать адаптации
3. **Размер контейнера**: Большой размер может влиять на скорость запуска

### Митигация:
1. Начать с локальной разработки и тестирования
2. Использовать multi-stage Docker builds
3. Оптимизировать зависимости и использовать кеширование

## 13. Контакты и ресурсы

- **fal.ai поддержка**: contact@fal.ai
- **Документация**: https://docs.fal.ai/
- **GitHub примеры**: https://github.com/fal-ai/fal
- **Discord сообщество**: https://discord.gg/fal-ai

---

**Статус**: План готов к реализации
**Дата создания**: 2025-08-12
**Автор**: VantageVeoConverter Team