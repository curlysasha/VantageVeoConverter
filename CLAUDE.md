# VantageVeoConverter Project Memory

## 🎯 PERFECT FREEZE DETECTION ACHIEVED! (2025-08-09)

### BREAKTHROUGH: Simulation-Based Freeze Prediction
**ПРОБЛЕМА РЕШЕНА**: Теперь алгоритм предсказания фризов работает **ИДЕАЛЬНО**!

**ЧТО БЫЛО НЕПРАВИЛЬНО**:
- Старый алгоритм анализировал только отклонения интервалов между timecodes
- Показывал желтые рамки на ВСЁМ проблемном отрезке 
- Одинаковые проценты отклонения на нормальных и замороженных кадрах

**РЕШЕНИЕ - СИМУЛЯЦИЯ PHYSICAL RETIME**:
```python
# Симулируем точно тот же процесс что и physical_retime.py
frame_usage = {}  # output_frame_idx -> input_frame_idx

for output_time_ms in np.arange(timestamps_ms[0], timestamps_ms[-1], target_frame_duration_ms):
    # Находим какой input кадр будет использован для этого output времени
    for i in range(len(timestamps_ms) - 1):
        if timestamps_ms[i] <= output_time_ms < timestamps_ms[i + 1]:
            input_frame_idx = i
            break
    frame_usage[output_frame_idx] = input_frame_idx

# Детектируем РЕАЛЬНЫЕ дубликаты
if input_frame == prev_input_frame:
    # Это НАСТОЯЩИЙ фриз!
```

**РЕЗУЛЬТАТ**: 
- ✅ Желтые рамки только на **реально замороженных кадрах**
- ✅ Точное предсказание дубликатов до создания видео
- ✅ Никаких ложных срабатываний на нормальных кадрах
- ✅ Идеальное соответствие с actual freeze detection

**КЛЮЧЕВАЯ ИДЕЯ**: Вместо анализа отклонений интервалов - **симулировать physical frame mapping** и детектировать дубликаты по `frame_usage` таблице.

## 🚨 VFR vs CFR Video Problem - КРИТИЧЕСКАЯ ЗАМЕТКА! (2025-08-09)

### ПРОБЛЕМА С mp4fpsmod И OpenCV:

**ПРОБЛЕМА**: `mp4fpsmod` создаёт VFR (Variable Frame Rate) видео с оригинальными кадрами но новыми таймингами. Когда OpenCV читает такое видео, он **ИГНОРИРУЕТ VFR тайминги** и читает кадры с постоянной скоростью, показывая оригинальное видео без фризов!

**ЧТО ПРОИСХОДИЛО**:
1. `mp4fpsmod` применял timecodes → создавал VFR видео (правильно)
2. Плееры показывали видео с фризами (правильно)  
3. НО OpenCV читал те же кадры как CFR → получал оригинал без фризов (НЕПРАВИЛЬНО!)

**РЕШЕНИЕ**: Создать `physical_retime.py` который создаёт CFR видео с **физическими дубликатами кадров**, а не VFR с таймингами.

**КОД**:
```python
# НЕПРАВИЛЬНО - создаёт VFR файл:
retime_video(input_video_path, paths["timecodes"], paths["retimed_video"])

# ПРАВИЛЬНО - создаёт CFR с физическими дубликатами:
create_physical_retime(input_video_path, paths["timecodes"], paths["retimed_video"])
```

**ЗАПОМНИ**: 
- **VFR файлы** (mp4fpsmod) - для плееров и финального вывода
- **CFR файлы с физическими дубликатами** (physical_retime) - для анализа через OpenCV

### ДИАГНОСТИЧЕСКАЯ СИСТЕМА:

**Файлы**:
- `src/physical_retime.py` - создаёт CFR видео с реальными дубликатами кадров
- `src/comparison_diagnostic.py` - создаёт top-bottom сравнение с красными маркерами на фризах
- `src/timing_analyzer.py` - анализирует timecodes и визуально детектирует фризы

**Рабочий workflow диагностики**:
1. Создать timecodes через обычный pipeline
2. Применить `create_physical_retime()` для создания CFR видео с физическими дубликатами
3. Использовать `create_comparison_diagnostic()` для визуализации фризов

**Результат**: Top-bottom видео где низ показывает фризы с красными рамками.

## Project Architecture

### Основные модули:
- `app_rife_compact.py` - основной Gradio интерфейс (refactored from 1600+ lines)
- `src/rife_engine.py` - RIFE AI interpolation engine с GPU поддержкой
- `src/motion_preserving_interpolator.py` - motion-aware интерполяция без slideshow эффекта
- `src/timing_analyzer.py` - анализ timing проблем + визуальная детекция фризов
- `src/video_processor.py` - обработка видео и интерполяция
- `src/audio_sync.py` - аудио синхронизация и VFR timecodes
- `src/comparison.py` - создание comparison grid для разных режимов

### RIFE Режимы:
- **Off**: только VFR синхронизация без интерполяции
- **Adaptive**: умная интерполяция только в проблемных областях
- **Precision**: точечная интерполяция в точках VFR изменений
- **Maximum**: полная интерполяция всего видео

### Известные проблемы и решения:
- ✅ **Slideshow эффект**: решён через motion-preserving interpolator
- ✅ **Дублирование кадров**: исправлено через правильную логику интерполяции
- ✅ **VFR/CFR confusion**: решён через physical_retime для диагностики
- ✅ **GPU acceleration**: добавлена поддержка CUDA для RIFE

## Development History

### Session 2025-08-09:
- Обнаружена критическая проблема с VFR/CFR в диагностике
- Создан `physical_retime.py` для генерации CFR видео с физическими дубликатами
- Исправлена диагностическая система для корректного отображения фризов
- Рефакторинг диагностики с side-by-side на top-bottom layout