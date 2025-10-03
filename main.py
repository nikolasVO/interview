import asyncio
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    Message,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
)
from aiogram.exceptions import TelegramBadRequest
import aiohttp
from dotenv import load_dotenv

# ------------------------
# CONFIG & LOGGING
# ------------------------
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interview-bot")

if not BOT_TOKEN or not DEEPSEEK_API_KEY:
    raise RuntimeError("Укажи BOT_TOKEN и DEEPSEEK_API_KEY в .env")

bot = Bot(BOT_TOKEN)
dp = Dispatcher()

DB_PATH = "interview.db"

# ------------------------
# DATA MODELS
# ------------------------
@dataclass
class Evaluation:
    score: int
    verdict: str
    feedback: str
    corrected_answer: str
    key_points: List[str] = field(default_factory=list)

@dataclass
class InterviewSession:
    role: str
    level: str
    topics: List[str]
    questions: List[str]
    index: int = 0
    scores: List[int] = field(default_factory=list)

# ------------------------
# FSM
# ------------------------
class InterviewStates(StatesGroup):
    waiting_role = State()
    choosing_level = State()
    choosing_topics = State()
    in_progress = State()

# ------------------------
# DB INIT
# ------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        level TEXT NOT NULL,
        topics TEXT NOT NULL,
        questions_count INTEGER NOT NULL,
        avg_score REAL NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

def save_session_result(user_id: int, role: str, level: str, topics: List[str], questions_count: int, avg_score: float):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO sessions (user_id, role, level, topics, questions_count, avg_score, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, role, level, ",".join(topics) if topics else "All", questions_count, avg_score, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def get_user_stats(user_id: int, limit: int = 5):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT role, level, topics, questions_count, avg_score, created_at FROM sessions WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit))
    rows = cur.fetchall()
    cur.execute("SELECT AVG(avg_score) FROM sessions WHERE user_id=?", (user_id,))
    overall = cur.fetchone()[0]
    conn.close()
    return rows, overall

# ------------------------
# DEEPSEEK UTILITIES
# ------------------------
async def deepseek_chat_json(system_prompt: str, user_prompt: str, temperature: float = 0.3, max_retries: int = 2) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    async with aiohttp.ClientSession() as session:
        for attempt in range(max_retries + 1):
            async with session.post(DEEPSEEK_URL, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning("DeepSeek status=%s body=%s", resp.status, text)
                    if attempt == max_retries:
                        raise RuntimeError(f"DeepSeek error {resp.status}: {text}")
                    continue

                data = await resp.json()
                content = data["choices"][0]["message"]["content"]

                # Пробуем распарсить JSON
                for extractor in (
                    lambda s: json.loads(s),
                    lambda s: json.loads(re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.S).group(1)) if re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.S) else (_ for _ in ()).throw(ValueError()),
                    lambda s: json.loads(re.search(r"(\{.*\})", s, flags=re.S).group(1)) if re.search(r"(\{.*\})", s, flags=re.S) else (_ for _ in ()).throw(ValueError()),
                ):
                    try:
                        return extractor(content)
                    except Exception:
                        pass
                if attempt == max_retries:
                    raise RuntimeError("Не удалось распарсить JSON из ответа DeepSeek.")
    raise RuntimeError("DeepSeek unreachable")

async def generate_questions(role: str, level: str, topics: List[str], n: int = 8) -> List[str]:
    system_prompt = "Ты — опытный технический интервьюер. Отвечай строго в формате JSON."
    topics_str = ", ".join(topics) if topics and "All" not in topics else "All"
    user_prompt = f"""
Сгенерируй {n} конкретных и практичных вопросов для собеседования на позицию "{role}" уровня "{level}".
Тематики: {topics_str}. Если тематика "All" — перемешай ключевые темы для уровня.
Не добавляй ответы. Не нумеруй в тексте.
Верни строго JSON вида:
{{
  "questions": ["Вопрос 1", "Вопрос 2", "..."]
}}
"""
    data = await deepseek_chat_json(system_prompt, user_prompt)
    qs = data.get("questions") or []
    clean = []
    for q in qs:
        q = q.strip("-•— \n\t\r")
        if q:
            clean.append(q)
    return clean[:n] if len(clean) >= n else clean

async def evaluate_answer(role: str, level: str, topics: List[str], question: str, answer: str, short_hint: bool) -> Evaluation:
    system_prompt = "Ты — строгий и точный технический интервьюер. Возвращай только JSON."
    add_hint = "Да, добавь список недостающих аспектов, которые кандидат должен раскрыть." if short_hint else "Нет, не добавляй отдельный список подсказок."
    user_prompt = f"""
Позиция: "{role}", уровень: "{level}", тематики: {", ".join(topics) if topics else "All"}.

Вопрос:
{question}

Ответ кандидата:
{answer}

Требуется:
1) Оцени ответ по шкале 0..100 (целое).
2) Дай краткий вердикт (1–2 предложения).
3) Дай фидбек (2–5 предложений, по делу).
4) Предложи улучшенную версию ответа (5–10 предложений максимум).
5) Перечисли 3–6 ключевых пунктов, которые хотелось бы услышать.
6) {add_hint}

Верни строго JSON:
{{
  "score": 0-100,
  "verdict": "…",
  "feedback": "…",
  "corrected_answer": "…",
  "key_points": ["…", "…"]
}}
"""
    data = await deepseek_chat_json(system_prompt, user_prompt)
    try:
        score = int(data.get("score", 0))
        score = max(0, min(100, score))
    except Exception:
        score = 0
    verdict = str(data.get("verdict", "")).strip()
    feedback = str(data.get("feedback", "")).strip()
    corrected_answer = str(data.get("corrected_answer", "")).strip()
    key_points = data.get("key_points") or []
    if not isinstance(key_points, list):
        key_points = []

    return Evaluation(
        score=score,
        verdict=verdict,
        feedback=feedback,
        corrected_answer=corrected_answer,
        key_points=[str(k).strip() for k in key_points if str(k).strip()],
    )

# ------------------------
# UI HELPERS
# ------------------------
LEVELS = ["Junior", "Middle", "Senior"]
TOPICS = ["All", "Collections", "Streams", "Concurrency", "SQL", "HTTP", "Design Patterns"]

def kb_levels() -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton(text=l, callback_data=f"level:{l}") for l in LEVELS]
    return InlineKeyboardMarkup(inline_keyboard=[row])

def kb_topics() -> InlineKeyboardMarkup:
    # Одноразовый выбор одной темы для простоты (можно расширить до мультиселекта)
    rows = []
    row = []
    for t in TOPICS:
        row.append(InlineKeyboardButton(text=t, callback_data=f"topic:{t}"))
        if len(row) == 3:
            rows.append(row); row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton(text="✅ Подтвердить выбор", callback_data="topic:confirm")])
    rows.append([InlineKeyboardButton(text="↩️ Назад к уровню", callback_data="topic:back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)

def kb_in_progress() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⏭ Пропустить вопрос", callback_data="skip")],
        [InlineKeyboardButton(text="⏹ Завершить", callback_data="stop")],
    ])

def kb_restart() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔁 Пройти ещё раз", callback_data="restart")],
    ])

def format_eval(e: Evaluation) -> str:
    kp = "\n".join(f"• {k}" for k in e.key_points[:6]) if e.key_points else "—"
    return (
        f"**Оценка:** {e.score}%\n"
        f"**Вердикт:** {e.verdict}\n\n"
        f"**Фидбек:**\n{e.feedback}\n\n"
        f"**Как лучше ответить:**\n{e.corrected_answer}\n\n"
        f"**Ключевые пункты:**\n{kp}"
    )

# ------------------------
# HANDLERS
# ------------------------
@dp.message(Command("start"))
async def on_start(message: Message, state: FSMContext):
    await state.clear()
    txt = (
        "Привет! Я бот-симулятор собеседования в IT 💼🤖\n\n"
        "Напиши позицию (например, *Java Developer*, *Python Backend*, *Frontend React*).\n"
        "После этого ты выберешь уровень и тематику.\n\n"
        "Также доступны команды:\n"
        "• /stats — посмотреть прогресс\n"
    )
    await message.answer(txt, parse_mode="Markdown")
    await state.set_state(InterviewStates.waiting_role)

@dp.message(Command("stats"))
async def on_stats(message: Message):
    rows, overall = get_user_stats(message.from_user.id, limit=5)
    if not rows:
        await message.answer("Пока нет результатов. Пройди интервью и возвращайся!")
        return
    lines = []
    for role, level, topics, qn, avg, created in rows:
        dt = created.split("T")[0]
        lines.append(f"• {dt}: {role} ({level}, темы: {topics}) — {avg:.1f}% из {qn} вопросов")
    overall_txt = f"\nСредний по всем сессиям: {overall:.1f}%" if overall is not None else ""
    await message.answer("Последние результаты:\n" + "\n".join(lines) + overall_txt)

@dp.message(InterviewStates.waiting_role)
async def on_role(message: Message, state: FSMContext):
    role = message.text.strip()
    await state.update_data(role=role)
    await message.answer(f"Позиция: *{role}*.\nВыбери уровень:", reply_markup=kb_levels(), parse_mode="Markdown")
    await state.set_state(InterviewStates.choosing_level)

@dp.callback_query(F.data.startswith("level:"), InterviewStates.choosing_level)
async def on_choose_level(cb: CallbackQuery, state: FSMContext):
    level = cb.data.split(":", 1)[1]
    data = await state.get_data()
    await state.update_data(level=level, topics=["All"])  # по умолчанию All
    await cb.message.answer(f"Уровень: *{level}*.\nВыбери тематику (или нажми «Подтвердить» для всех):", reply_markup=kb_topics(), parse_mode="Markdown")
    await state.set_state(InterviewStates.choosing_topics)
    await cb.answer()

@dp.callback_query(F.data.startswith("topic:"), InterviewStates.choosing_topics)
async def on_choose_topic(cb: CallbackQuery, state: FSMContext):
    choice = cb.data.split(":", 1)[1]
    data = await state.get_data()
    topics = data.get("topics") or ["All"]

    if choice == "confirm":
        # Генерим вопросы
        role = data["role"]
        level = data["level"]
        await cb.message.answer("Генерирую вопросы… ⏳")
        try:
            questions = await generate_questions(role, level, topics, n=8)
        except Exception as e:
            logger.exception("Ошибка генерации вопросов: %s", e)
            await cb.message.answer("Не получилось сгенерировать вопросы. Попробуй ещё раз /start")
            await state.clear()
            await cb.answer()
            return

        if not questions:
            await cb.message.answer("ИИ не вернул вопросы. Попробуй, например: *Java Developer*.", parse_mode="Markdown")
            await cb.answer()
            return

        session = InterviewSession(role=role, level=level, topics=topics, questions=questions, index=0, scores=[])
        await state.update_data(session=session.__dict__)
        await cb.message.answer(
            f"Ок! Позиция: *{role}*, уровень: *{level}*, темы: *{', '.join(topics)}*.\n"
            f"Всего вопросов: *{len(questions)}*.\n\n"
            f"Вопрос 1:\n{questions[0]}",
            reply_markup=kb_in_progress(),
            parse_mode="Markdown"
        )
        await state.set_state(InterviewStates.in_progress)
        await cb.answer("Стартуем!")
        return

    if choice == "back":
        await state.set_state(InterviewStates.choosing_level)
        await cb.message.answer("Выбери уровень:", reply_markup=kb_levels())
        await cb.answer()
        return

    # Выбор одной темы (перезаписываем)
    await state.update_data(topics=[choice])
    await cb.message.answer(f"Выбрана тема: *{choice}*.\nНажми «✅ Подтвердить выбор».", parse_mode="Markdown")
    await cb.answer()

@dp.callback_query(F.data == "skip", InterviewStates.in_progress)
async def on_skip(cb: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    session = InterviewSession(**data["session"])
    session.index += 1
    if session.index >= len(session.questions):
        await finish_interview(cb.message, state, session)
        await cb.answer("Завершили")
        return
    await state.update_data(session=session.__dict__)
    await cb.message.answer(
        f"Вопрос {session.index + 1}:\n{session.questions[session.index]}",
        reply_markup=kb_in_progress()
    )
    await cb.answer("Пропущено")

@dp.callback_query(F.data == "stop", InterviewStates.in_progress)
async def on_stop(cb: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    session = InterviewSession(**data["session"])
    await finish_interview(cb.message, state, session)
    await cb.answer("Ок, прекращаем")

@dp.callback_query(F.data == "restart")
async def on_restart(cb: CallbackQuery, state: FSMContext):
    await state.clear()
    await cb.message.answer("Напиши позицию заново (например, *Java Developer*).", parse_mode="Markdown")
    await state.set_state(InterviewStates.waiting_role)
    await cb.answer("Перезапуск")

@dp.message(InterviewStates.in_progress)
async def on_answer(message: Message, state: FSMContext):
    data = await state.get_data()
    session = InterviewSession(**data["session"])
    role, level, topics = session.role, session.level, session.topics
    question = session.questions[session.index]
    answer = message.text or ""

    # Мягкая проверка длины (подскажем, но не блокируем)
    short_hint = len(answer.strip()) < 200

    await message.answer("Проверяю ответ… 🧠")
    try:
        ev = await evaluate_answer(role, level, topics, question, answer, short_hint=short_hint)
    except Exception as e:
        logger.exception("Ошибка оценки ответа: %s", e)
        await message.answer("Не смог оценить ответ (ошибка от ИИ). Попробуй ответить ещё раз или нажми Пропустить.")
        return

    session.scores.append(ev.score)
    await state.update_data(session=session.__dict__)

    # Если ответ короткий — отдельная дружелюбная подсказка
    if short_hint:
        await message.answer("Подсказка: попробуй добавить 1–2 конкретных примера, упоминание trade-offs и типичных подводных камней. Это повышает оценку.")

    # Отправляем фидбек
    try:
        await message.answer(format_eval(ev), parse_mode="Markdown")
    except TelegramBadRequest:
        await message.answer(format_eval(ev).replace("**", ""))

    # Следующий вопрос / финиш
    session.index += 1
    if session.index >= len(session.questions):
        await finish_interview(message, state, session)
    else:
        await state.update_data(session=session.__dict__)
        await message.answer(
            f"Вопрос {session.index + 1}:\n{session.questions[session.index]}",
            reply_markup=kb_in_progress()
        )

async def finish_interview(message: Message, state: FSMContext, session: InterviewSession):
    avg = round(sum(session.scores) / len(session.scores), 1) if session.scores else 0.0
    verdict = (
        "Отличная подготовка! 🔥" if avg >= 80 else
        "Хорошая база, подтяни пробелы. 👍" if avg >= 55 else
        "Нужно повторить основы и пройтись по ключевым темам. 💪"
    )
    # Сохраняем в БД
    save_session_result(
        user_id=message.from_user.id,
        role=session.role,
        level=session.level,
        topics=session.topics,
        questions_count=len(session.questions),
        avg_score=avg
    )
    await message.answer(
        f"✅ Собеседование завершено!\n"
        f"Позиция: *{session.role}* — уровень *{session.level}*, темы: *{', '.join(session.topics)}*\n"
        f"Вопросов: *{len(session.questions)}*\n"
        f"Средний балл: *{avg}%*\n\n"
        f"{verdict}\n\n"
        f"Команда: /stats — посмотреть прогресс.",
        reply_markup=kb_restart(),
        parse_mode="Markdown"
    )
    await state.clear()

# ------------------------
# ENTRY POINT
# ------------------------
async def main():
    init_db()
    logger.info("Starting bot...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
