# Detector Dangerous

**Computer Vision / Audio ML система для мониторинга потенциально опасных событий по видео и аудио.**

Приложение получает потоки с камер и микрофонов, анализирует положение человека, опасные звуки (выстрелы) и заданные тревожные слова, после чего формирует события, сохраняет материалы инцидента и показывает их оператору.

> Основной фокус проекта: real-time CV/Audio ML pipeline - RTSP-видео, temporal pose analysis, анализ звука, настройка порогов, журнал событий и интеграция моделей в рабочее приложение.

## Что делает проект

Система объединяет несколько независимых каналов анализа:

1. Получает видео с локальных камер, файлов или RTSP-потоков.
2. Анализирует позу человека и изменение положения во времени.
3. Обрабатывает аудио с микрофонов и ищет потенциально опасные звуковые события.
4. Распознаёт заданные тревожные слова локально.
5. Объединяет результаты детекторов и формирует подтверждённое событие.
6. Сохраняет инцидент в SQLite, кадры/видеоклипы и при необходимости отправляет событие через REST/HTTP.
7. Показывает текущую работу системы и историю инцидентов в PyQt6-интерфейсе.

## Моя роль - Computer Vision / ML Engineer

Коммерческий проект для заказчика, команда из трёх человек. Я отвечал за **ML/CV-часть системы и интеграцию моделей в приложение реального времени**:

- подбирал и сравнивал модели Computer Vision и Audio ML для обнаружения падений и длительного горизонтального положения человека, опасных звуков и тревожных слов;
- настраивал пороги уверенности моделей и правила принятия решения с учётом реальных сценариев и акустической обстановки;
- интегрировал модели компьютерного зрения в систему обработки нескольких RTSP-видеопотоков на Python, PyTorch и OpenCV;
- разработал логику анализа положения человека по последовательности кадров на основе **YOLO-Pose и keypoints**, чтобы решение принималось по изменению позы во времени, а не по одному кадру;
- проверял аудиодетектор на наборе реальных сценариев и микрофонов, анализировал ложные и пропущенные срабатывания по журналам работы системы и корректировал пороги;
- подготовил запуск моделей через **ONNX / TensorRT** для ускорения inference;
- довёл ML-часть до запуска у заказчика: система одновременно обрабатывала данные с **2 камер и 1 микрофона**.

## Архитектура

```text
RTSP / Camera
      │
      ▼
 YOLO-Pose / CV
      │
      ▼
Temporal analysis ──────────────┐
                               │
Microphone                     │
   │                           │
   ├── Audio event detector ───┤
   └── Keyword detector ───────┤
                               ▼
                         Event processing
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
             SQLite        Frames/Clips    REST/HTTP
                │
                ▼
           PyQt6 UI / History
```


## Технологии

**Computer Vision / ML**
- Python
- PyTorch
- Ultralytics YOLO / YOLO-Pose
- OpenCV
- ONNX
- TensorRT
- NumPy

**Audio ML**
- YAMNet
- AST / PANNs-подобные аудиомодели
- Vosk
- обработка и калибровка микрофона
- temporal evidence / threshold tuning

**Engineering**
- RTSP
- FFmpeg
- multithreading
- SQLite
- REST / HTTP
- PyQt6
- logging
- unit / integration tests

## Структура репозитория

```text
.
├── video/
│   ├── fall_detector_temporal.py   # Temporal pose analysis
│   ├── fall_detector_legacy.py
│   ├── lying_classifier.py
│   └── model_optimizer.py
│
├── audio/
│   ├── gunshot_detector_runtime.py # Audio-event pipeline
│   ├── keyword_detector.py         # Keyword detection
│   ├── microphone.py               # Работа с микрофоном
│   └── calibration.py              # Калибровка аудио
│
├── core/
│   ├── event_bus.py
│   ├── incidents.py
│   ├── event_clips.py
│   ├── storage.py
│   └── alerts.py
│
├── ui/                             # PyQt6 interface
├── tools/                          # Диагностика и вспомогательные утилиты
├── tests/                          # Unit / integration tests
└── detector_config.example.json
```

## Запуск проекта

Проект ориентирован на Windows 10/11 и Python 3.12 x64.

Быстрый запуск:

```bat
install_and_run.bat
```

Или настройка по шагам:

```bat
setup.bat
run.bat
```

Пример конфигурации находится в `detector_config.example.json`. Рабочий `detector_config.json` создаётся локально.

Проверка установки и тесты:

```bat
verify_install.bat
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```
