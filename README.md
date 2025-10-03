Interview IT Bot — симулятор собеседований с ИИ (DeepSeek + Telegram)

Коротко: это Telegram-бот, который проводит пользователя через серию вопросов по выбранной позиции и уровню (например, Java Developer — Middle), оценивает каждый ответ по шкале 0–100%, даёт фидбек, предлагает улучшенную формулировку и подсвечивает ключевые пункты. В конце — итоговый балл и прогресс в /stats.

🎯 Цели проекта

Показать скилл работы с ИИ: интеграция DeepSeek API, строгие JSON-контракты, устойчивый парсинг.

Реалистичная тренировка интервью: вопросы по роли/уровню/темам, поочерёдный диалог, разбор ответов.

Осознанная подготовка: прозрачные метрики (баллы, ключевые пункты, улучшенные ответы), история прогресса.

Простота запуска: один файл main.py, локальная SQLite, быстрый старт на Windows/macOS/Linux.

✨ Возможности

Выбор позиции (любой текст: “Java Developer”, “Python Backend”, “React Frontend”…).

Выбор уровня: Junior | Middle | Senior.

Выбор тематики: All | Collections | Streams | Concurrency | SQL | HTTP | Design Patterns.

Генерация пула вопросов под роль/уровень/тему.

Для каждого ответа:

оценка в %,

краткий вердикт,

по делу фидбек,

улучшенная версия ответа,

ключевые пункты, которые хотелось бы услышать.

Мягкая подсказка, если ответ слишком короткий (что добавить).

/stats — история последних сессий + общий средний балл.

Inline-кнопки: «⏭ Пропустить», «⏹ Завершить», «🔁 Пройти ещё раз».

🧱 Архитектура (в двух словах)

Telegram: aiogram 3 (FSM для диалога, inline-кнопки).

LLM: DeepSeek Chat (модель по умолчанию: deepseek-chat).

Хранилище: SQLite (interview.db) — сессии, средние баллы.

JSON-контракты: запросы к DeepSeek с системными подсказками, возврат строго в JSON; устойчивый парсинг (включая «раскоп» из код-блоков).

🚀 Быстрый старт
1) Клонируй и установи зависимости
git clone <your-repo-url> interview-bot
cd interview-bot
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt

2) Создай .env рядом с main.py
BOT_TOKEN=ваш_telegram_bot_token
DEEPSEEK_API_KEY=ваш_deepseek_api_key
DEEPSEEK_MODEL=deepseek-chat

3) Запусти
python main.py


Если PowerShell ругается на активацию venv:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

🧪 Как пользоваться

Напиши боту /start.

Введи позицию (например, Java Developer).

Выбери уровень и тему.

Отвечай на вопросы — после каждого ответа получишь оценку и рекомендации.

Посмотри прогресс: /stats.

🗂️ Структура проекта
.
├── main.py               # Весь код бота (aiogram + DeepSeek + SQLite)
├── requirements.txt      # Зависимости
├── .env                  # Токены (не коммитить)
└── interview.db          # Создаётся автоматически

🔑 Переменные окружения
Переменная	Обяз.	Описание
BOT_TOKEN	✅	Токен Telegram Bot API
DEEPSEEK_API_KEY	✅	Ключ DeepSeek API
DEEPSEEK_MODEL	❕	deepseek-chat (по умолчанию)
🧠 Логика генерации и оценки

Генерация вопросов: промпт с ролью, уровнем, темой; на выходе JSON:
{ "questions": ["...", "..."] }

Оценка ответа: промпт с вопросом и ответом; на выходе JSON:

{
  "score": 0-100,
  "verdict": "кратко",
  "feedback": "по делу",
  "corrected_answer": "улучшенная версия",
  "key_points": ["..."]
}


Надёжность: парсер извлекает JSON даже из текстов с код-блоками; лимитирует оценки в 0..100.

🛡️ Безопасность

Не коммить .env и interview.db.

Храни API-ключи в секретах CI/CD и переменных окружения.

Желательно ограничить права бота (Privacy Mode включён в BotFather).
