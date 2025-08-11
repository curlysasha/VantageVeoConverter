# VantageVeoConverter - AI Video Synchronizer

🚀 **Автоматическая синхронизация видео с аудио + RIFE AI интерполяция**

## Быстрая установка

### На сервере (Docker/Linux)
```bash
# Одной командой - всё автоматически:
./init_server.sh
```

### Локальная установка
```bash
# 1. Создать виртуальное окружение (не обязательно)
python -m venv VantageVeoConverter
source VantageVeoConverter/bin/activate  # Linux/Mac
# VantageVeoConverter\Scripts\activate   # Windows

# 2. Автоматическая установка всех зависимостей
python install_dependencies.py

# 3. Запуск приложения
python app_rife_compact.py
```

## Что делает автоматическая установка

✅ **Системные зависимости**: espeak, libespeak-dev, build-essential  
✅ **Python пакеты**: numpy, scipy, torch, opencv, gradio, whisper  
✅ **Aeneas**: автоматически с 4 разными методами установки  
✅ **Бинарные файлы**: ffmpeg, ffprobe, mp4fpsmod  
✅ **RIFE модели**: автоматическая загрузка при первом запуске

## Возможности

🎯 **VFR синхронизация**: точное выравнивание видео по аудио  
🤖 **RIFE AI**: нейросетевая интерполяция для устранения фризов  
🔍 **Диагностика**: визуализация проблем с 3-панельным сравнением  
⚡ **GPU ускорение**: автоматическое использование CUDA если доступно  

## Интерфейс

После запуска открыть в браузере: `http://localhost:7860`

## Диагностика проблем

Если что-то не работает:
```bash
# Проверить системные зависимости
which ffmpeg ffprobe

# Переустановить aeneas вручную  
pip install setuptools==59.5.0
pip install aeneas

# Полная переустановка
rm -rf VantageVeoConverter/
python install_dependencies.py
```

---
**Система полностью автоматическая - просто запускай `./init_server.sh`!**