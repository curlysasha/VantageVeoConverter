# Server Quick Setup

## One-Command Setup
```bash
./init_server.sh
```

## Manual Setup
```bash
# 1. Create virtual environment
python -m venv VantageVeoConverter
source VantageVeoConverter/bin/activate

# 2. Install dependencies automatically
python install_dependencies.py

# 3. Setup binaries
python setup_binaries.py

# 4. Start server
python app_rife_compact.py
```

## If Aeneas Fails
```bash
# Install system deps first
sudo apt-get install espeak libasound2-dev libespeak-dev

# Install numpy/scipy first
pip install numpy>=1.24,<2.3 scipy>=1.9.0

# Install aeneas with special flags
pip install --no-build-isolation aeneas>=1.7.3
```

## Restart Container Command
```bash
# Delete venv and start fresh
rm -rf VantageVeoConverter/
./init_server.sh
```

Система теперь автоматически:
- ✅ Создаёт venv если нужно
- ✅ Устанавливает системные зависимости  
- ✅ Правильно устанавливает aeneas
- ✅ Скачивает бинарные файлы
- ✅ Проверяет установку
- ✅ Запускает сервер