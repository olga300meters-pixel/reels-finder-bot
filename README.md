# Reels Finder — Instagram DM бот (песня / фильм / одежда из Reels)

Пользователь пересылает Reels в Direct → нажимает кнопку → за 3–5 секунд получает трек (AudD), фильм/сериал или вещи (Gemini Vision). Кредиты, подписка, оплата через Lemon Squeezy, Comment-to-DM.

## Структура репозитория

```
reels-finder-bot/
├── main.py            # всё приложение (FastAPI)
├── requirements.txt
├── Dockerfile         # python:3.12-slim + ffmpeg, 1 воркер uvicorn
├── .env.example       # все переменные окружения с комментариями
├── .gitignore
├── .dockerignore
└── README.md
```

## Как устроено

| Эндпоинт | Что делает |
|---|---|
| `GET /webhook` | Верификация Meta (`hub.verify_token` → `hub.challenge`) |
| `POST /webhook` | Проверка `X-Hub-Signature-256`, дедуп по `mid` (TTLCache на 1000 записей), мгновенный `200`, вся логика в `BackgroundTasks` |
| `POST /payment-webhook` | Проверка `X-Signature` (HMAC SHA256) Lemon Squeezy, +50 кредитов или +30 дней подписки, DM-уведомление |
| `GET /health` | Проверка БД (для health check на Koyeb/Railway) |

**Конвейер поиска:** скачивание в `/tmp/{uuid}` (стрим с лимитом размера) → `ffprobe` →
- `FIND_MUSIC`: ровно 8 с аудио (с 0:03, или с 0:00 если ролик < 11 с) → моно MP3 22050 Гц 64 кбит/с → AudD;
- `FIND_MOVIE` / `FIND_ITEM`: 3 кадра на 20/50/80 % → `scale=640:-1 -q:v 4` → Gemini (JSON-схема ответа);
- `FIND_ALL`: аудио и кадры обрабатываются **параллельно**, Gemini делает один объединённый запрос.

Папка `/tmp/{uuid}` удаляется в `finally` при любом исходе.

**Отказоустойчивость Gemini:** `gemini-2.5-flash` (8 с) → при 429/503/5xx/404/таймауте/пустом ответе → `gemini-2.5-flash-lite`. Если упали обе модели или AudD, пользователь получает *«Temporary overload, your credit was not deducted…»*, и кредит не списывается.

**Кредиты списываются только при успешном ответе:**
- Если ничего не найдено, кредит не списывается, и пользователь получает об этом сообщение.
- `FIND_ALL` стоит 2 кредита. Если одна половина не смогла выполниться (фото без звука, сбой одного API), списывается 1 кредит.
- Двойное нажатие не спишет кредиты дважды: на каждого пользователя стоит `asyncio.Lock`, а списание делается атомарным `UPDATE ... WHERE credits >= cost`.

**Окно 24 часа:** если уведомление об оплате не доставлено (пользователь не писал боту более 24 ч), текст сохраняется в `users.pending_notice` и отправляется автоматически при следующем сообщении пользователя.

**Токен Instagram** (живёт 60 дней) бот сам обновляет каждые ~5 дней и хранит в таблице `app_kv`. Если вы поменяете `IG_ACCESS_TOKEN` в переменных окружения, бот начнёт использовать новый токен.

Таблицы создаются автоматически при старте (`users`, `payment_events` — защита от повторных платёжных вебхуков, `app_kv`).

---

## ⚠️ Прочитайте перед запуском (актуально на ноябрь 2026)

1. **Модели Gemini 2.5.** Google оставил `gemini-2.5-flash` и `gemini-2.5-flash-lite` без даты отключения, но **доступ к ним есть только у ключей, которые уже их использовали**. Для новых проектов Google рекомендует `gemini-3.8-flash` / `gemini-3.5-flash-lite`. Если на новом ключе в логах появляется `HTTP 404`, пропишите эти модели в `GEMINI_PRIMARY_MODEL` / `GEMINI_FALLBACK_MODEL`. Код при этом менять не нужно.
2. **Ссылки на Instagram/TikTok/YouTube.** Официальный API не даёт скачивать чужие публикации по ссылке, а скрейпинг нарушает правила Meta. Поэтому на такие ссылки бот отвечает инструкцией «нажмите ✈️ и перешлите сам Reels». Прямые ссылки на файлы (`.mp4`, `.jpg` и т.п.) принимаются.
3. **Тип вложения `share` убран.** Meta заменяет его на `ig_post`. Бот поддерживает `ig_reel`, `ig_post`, `share`, `video`, `image`, `audio`.
4. **Ровно 1 реплика и 1 воркер.** Дедупликация и блокировки хранятся в памяти процесса. Не включайте автоскейлинг.
5. **Lemon Squeezy** работает, но с января 2026 команда переносит разработку в Stripe Managed Payments. Подпись вебхуков и `custom_data` не менялись.
6. **Бесплатный хостинг и скорость:**
   - **Koyeb Free** — 0.1 vCPU и 512 МБ RAM, засыпает после 1 часа без трафика (холодный старт 1–5 с). Бот работает, но обработка займёт скорее 6–10 с, чем 3–5. Чтобы снизить нагрузку на CPU, кадры извлекаются по ключевым кадрам.
   - **Railway Free** даёт $1 кредита в месяц. Для работы 24/7 этого не хватит, нужен Hobby ($5/мес).
   - Для стабильных 3–5 с нужно ≥ 0.5 vCPU.

---

## Пошаговый запуск (~10–15 минут)

### Шаг 0. Подготовка
- Instagram-аккаунт бота переведите в **профессиональный** (Business или Creator).
- В приложении Instagram аккаунта бота: **Настройки → Сообщения и ответы на истории → Инструменты для сообщений → Разрешить доступ к сообщениям** — включить.
- Создайте приватный репозиторий на GitHub и залейте туда файлы проекта. Файл `.env` не коммитьте.

### Шаг 1. База данных Supabase (2 мин)
1. Откройте [supabase.com](https://supabase.com) → **New project**. Задайте пароль БД и выберите регион ближе к хостингу (например, Frankfurt для Koyeb FRA).
2. Нажмите **Connect** вверху проекта и выберите **Session pooler** (порт 5432). Этот вариант работает по IPv4, а прямое подключение на бесплатном тарифе работает только по IPv6 и с Railway/Koyeb может не соединиться.
3. Скопируйте URI, подставьте пароль и допишите `?sslmode=require`. Получится `DATABASE_URL`:
   ```
   postgresql://postgres.abcd1234:ПАРОЛЬ@aws-0-eu-central-1.pooler.supabase.com:5432/postgres?sslmode=require
   ```
   Transaction pooler (порт 6543) тоже подойдёт: `statement_cache_size=0` уже включён в коде.
4. Таблицы создавать не нужно, бот сделает это сам.

> Бесплатный проект Supabase ставится на паузу после 7 дней без активности. Работающий бот обращается к БД постоянно, так что пауза не наступит.
> **Альтернатива — Neon:** скопируйте connection string как есть. Параметр `channel_binding` бот удалит сам.

### Шаг 2. Ключи API (3 мин)
- **Gemini:** [aistudio.google.com/apikey](https://aistudio.google.com/apikey) → Create API key → `GEMINI_API_KEY`.
- **AudD:** [dashboard.audd.io](https://dashboard.audd.io) → токен → `AUDD_API_TOKEN`.
- **Lemon Squeezy:**
  1. Создайте два продукта:
     - «50 credits» — разовый платёж;
     - «Unlimited» — подписка, ежемесячно.
  2. В каждом продукте нажмите **Share** и скопируйте ссылку чекаута: ссылку пакета кредитов → `LEMON_CHECKOUT_URL`, ссылку подписки → `LEMON_SUBSCRIPTION_CHECKOUT_URL`.
  3. Узнайте ID вариантов (видны в URL варианта или через API) и пропишите их в `LEMON_CREDIT_PACK_VARIANT_IDS` и `LEMON_SUBSCRIPTION_VARIANT_IDS`.
  4. Придумайте случайную строку → `LEMON_SIGNING_SECRET`. Она понадобится в шаге 6.

### Шаг 3. Приложение в Meta for Developers (3 мин)
1. Откройте [developers.facebook.com/apps](https://developers.facebook.com/apps) → **Create app** → сценарий **«Manage messaging & content on Instagram»** (Instagram API).
2. Перейдите в **Instagram → API setup with Instagram login**:
   - **Generate access tokens** → **Add account** → войдите аккаунтом бота → **Generate token** → `IG_ACCESS_TOKEN`;
   - на той же странице скопируйте **Instagram app secret** → `IG_APP_SECRET`.
3. Проверьте, что включены разрешения `instagram_business_basic`, `instagram_business_manage_messages` и `instagram_business_manage_comments`.
4. Придумайте `IG_VERIFY_TOKEN` (любая длинная строка).

### Шаг 4. Деплой

**Вариант A — Railway:**
1. [railway.com](https://railway.com) → **New Project → Deploy from GitHub repo** → выберите репозиторий. Railway найдёт `Dockerfile` сам.
2. Откройте **Variables → Raw Editor**, вставьте содержимое `.env.example` со своими значениями и нажмите **Deploy**.
3. В **Settings → Networking → Generate Domain** получите адрес вида `https://xxx.up.railway.app`.
4. Количество реплик оставьте равным **1**.

**Вариант B — Koyeb (бесплатно):**
1. [app.koyeb.com](https://app.koyeb.com) → **Create Web Service → GitHub** → выберите репозиторий.
2. **Builder:** Dockerfile. **Instance:** Free. **Region:** Frankfurt или Washington.
3. **Environment variables:** добавьте все переменные из `.env.example`.
4. **Exposed port:** `8000`. **Health check:** HTTP, путь `/health`.
5. **Deploy**. Адрес будет вида `https://имя-организация.koyeb.app`.

Проверка: `https://ВАШ_ДОМЕН/health` должен вернуть `{"status":"ok","db":true}`.

### Шаг 5. Вебхук в Meta (2 мин)
1. В приложении Meta откройте **Instagram → API setup with Instagram login → Configure webhooks**:
   - **Callback URL:** `https://ВАШ_ДОМЕН/webhook`;
   - **Verify token:** значение `IG_VERIFY_TOKEN`;
   - нажмите **Verify and save**.
2. Подпишитесь на поля **messages**, **messaging_postbacks** и **comments**.
3. В блоке аккаунта включите **Webhook subscription** для аккаунта бота.

### Шаг 6. Вебхук Lemon Squeezy (1 мин)
**Settings → Webhooks → +**:
- **URL:** `https://ВАШ_ДОМЕН/payment-webhook`;
- **Signing secret:** значение `LEMON_SIGNING_SECRET`;
- **События:** `order_created`, `subscription_payment_success`.

Сначала проверьте всё в **Test mode**.

### Шаг 7. Тест
Пока приложение в режиме разработки, бот отвечает только пользователям с ролью в приложении:
1. Добавьте тестировщика: **App roles → Roles → Add People → Instagram Tester**.
2. Тестировщик принимает приглашение в Instagram: **Настройки → Приложения и сайты → Приглашения тестировщиков**.

Затем с тестового аккаунта:
1. Перешлите боту любой Reels — должны появиться 4 кнопки.
2. Нажмите **🎵 Find Song** — придёт ответ с треком и останется 2 кредита.
3. Напишите `FIND` под постом бота — придёт Private Reply в Direct.
4. Потратьте кредиты — придёт карточка оплаты. Оплатите тестовой картой Lemon — придёт «✅ Payment received» и +50 кредитов.

### Шаг 8. Запуск для всех пользователей
- Подайте **App Review** на Advanced Access для `instagram_business_manage_messages` и `instagram_business_manage_comments`. Потребуются скринкаст работы бота и Privacy Policy URL.
- Пройдите **Business Verification**.
- Переключите приложение в режим **Live**.

---

## Команды для пользователей (текстом)
Quick Replies не показываются в Instagram на десктопе, поэтому бот понимает и текстовые команды:

| Команда | Действие |
|---|---|
| `song`, `music` | поиск песни |
| `movie`, `film`, `show` | поиск фильма/сериала |
| `outfit`, `items`, `clothes` | поиск вещей |
| `all`, `everything` | поиск всего сразу |
| `balance`, `credits` | показать баланс |
| `buy`, `upgrade` | показать карточку оплаты |

## Локальный запуск
```bash
pip install -r requirements.txt          # ffmpeg должен быть установлен в системе
cp .env.example .env                     # заполните значения
set -a && source .env && set +a
uvicorn main:app --port 8000 --workers 1
# для вебхуков Meta нужен публичный HTTPS: ngrok http 8000 / cloudflared tunnel
```

## Частые проблемы

| Симптом в логах | Причина / решение |
|---|---|
| `Rejected webhook with invalid signature` | В `IG_APP_SECRET` указан Meta App Secret вместо Instagram App Secret. Можно перечислить оба через запятую. |
| `Gemini ... HTTP 404` | Ключу недоступны модели 2.5 — переключитесь на `gemini-3.8-flash` / `gemini-3.5-flash-lite`. |
| `Graph API error ... subcode=2534037` | Переписку ведёт Meta Business AI. Отключите Business AI для Direct в настройках аккаунта. |
| `code=10 / subcode=2018278` | Прошло больше 24 ч с последнего сообщения пользователя. Уведомление поставлено в очередь и придёт при следующем сообщении. |
| `Could not connect to database` | Используйте Session pooler URI (IPv4), а не Direct connection (IPv6). |
| Бот отвечает только вам | Приложение в Development mode — см. шаг 8. |

## Что не входит в MVP
Не реализованы:
- возвраты (`order_refunded`) — кредиты не отзываются автоматически;
- админ-панель;
- горизонтальное масштабирование. Для него дедупликацию и блокировки нужно вынести в Redis или Postgres.
