"""
Reels Finder — Instagram DM bot (official Instagram Messaging API).

A user forwards a Reel / video / photo to the business account in Direct,
taps a Quick Reply, and within ~3-5 s the bot identifies:
  * the song            (AudD API, 8-second mono MP3 clip)
  * the movie / series  (Gemini Vision, 3 frames)
  * clothes / items     (Gemini Vision, 3 frames)

Stack: FastAPI + uvicorn, httpx (pooled AsyncClient), asyncpg, google-genai,
ffmpeg via asyncio.create_subprocess_exec.

IMPORTANT: run with exactly ONE uvicorn worker / ONE replica. Webhook
de-duplication and per-user locks live in process memory.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections import deque
import hashlib
import html
import ipaddress
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import string
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import quote_plus, urlencode, urljoin, urlparse

import asyncpg
import httpx
from cachetools import TTLCache
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("reels-finder")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)


# =============================================================================
# Configuration
# =============================================================================


def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.getenv(name, default)
    if value is not None:
        value = value.strip()
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value or ""


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw and raw.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw and raw.strip() else default


def _env_list(name: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, "").split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    # Instagram / Meta
    ig_access_token: str
    ig_verify_token: str
    ig_app_secrets: list[str]
    ig_account_id: str
    graph_api_base: str
    bot_name: str
    contact_email: str
    instagram_handle: str
    operator_name: str
    # Database
    database_url: str
    # Gemini
    gemini_api_key: str
    gemini_primary_model: str
    gemini_fallback_model: str
    gemini_timeout_sec: float
    gemini_extra_models: str
    backup_ai_url: str
    backup_ai_key: str
    backup_ai_model: str
    gemini_thinking_level: str
    # AudD
    audd_api_token: str
    audd_timeout_sec: float
    # Lemon Squeezy
    lemon_checkout_url: str
    lemon_subscription_checkout_url: str
    lemon_signing_secret: str
    lemon_subscription_variant_ids: list[str]
    lemon_credit_pack_variant_ids: list[str]
    paywall_image_url: str
    # Business rules
    free_credits: int
    credits_per_pack: int
    subscription_days: int
    # Runtime limits
    max_download_mb: int
    max_parallel_jobs: int
    inbox_poll_sec: int
    tmp_root: str
    allow_private_urls: bool
    # Reel downloader (yt-dlp)
    admin_code: str
    ask_language: bool
    referral_credits: int
    referral_daily_cap: int
    mentions_enabled: bool
    # Stripe
    stripe_secret_key: str
    stripe_webhook_secret: str
    stripe_prices: dict[str, str]   # plan code → Stripe Price ID
    pack_credits: dict[str, int]    # plan code → credits (one-time packs)
    stripe_automatic_tax: bool
    public_url: str
    sub_daily_limit: int
    mentions_per_hour: int
    ig_cookies: str
    download_proxy: str
    # Shazam (second recogniser; better for phonk / TikTok-style remixes)
    shazam_enabled: bool
    ai_song_enabled: bool

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            ig_access_token=_env("IG_ACCESS_TOKEN", required=True),
            ig_verify_token=_env("IG_VERIFY_TOKEN", required=True),
            ig_app_secrets=_env_list("IG_APP_SECRET") or [_env("IG_APP_SECRET", required=True)],
            ig_account_id=_env("IG_ACCOUNT_ID", "me"),
            # Unified login tokens (EAA...) work on graph.facebook.com; Instagram-login tokens (IG...)
            # on graph.instagram.com. Picked automatically unless GRAPH_API_BASE is set.
            graph_api_base=(_env("GRAPH_API_BASE") or (
                "https://graph.facebook.com/v26.0" if _env("IG_ACCESS_TOKEN").startswith("EAA")
                else "https://graph.instagram.com/v24.0")).rstrip("/"),
            bot_name=_env("BOT_NAME", "Reels Finder"),
            contact_email=_env("CONTACT_EMAIL", ""),
            instagram_handle=_env("INSTAGRAM_HANDLE", "").lstrip("@"),
            operator_name=_env("OPERATOR_NAME", ""),
            database_url=_env("DATABASE_URL", required=True),
            gemini_api_key=_env("GEMINI_API_KEY", required=True),
            gemini_primary_model=_env("GEMINI_PRIMARY_MODEL", "gemini-2.5-flash"),
            gemini_fallback_model=_env("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite"),
            # Per-attempt timeout; vision calls always get at least 12 s (item lists are long answers).
            gemini_timeout_sec=_env_float("GEMINI_TIMEOUT_SEC", 12.0),
            # Extra Gemini models tried after primary/fallback (comma list). Unknown names are skipped.
            gemini_extra_models=_env("GEMINI_EXTRA_MODELS", ""),
            # Optional second AI provider for photo/video tasks when ALL Gemini models fail.
            # Any OpenAI-compatible API, e.g. OpenRouter: https://openrouter.ai/api/v1 + a vision model.
            backup_ai_url=_env("BACKUP_AI_URL", "https://openrouter.ai/api/v1").rstrip("/"),
            backup_ai_key=_env("BACKUP_AI_KEY", ""),
            backup_ai_model=_env("BACKUP_AI_MODEL", ""),
            gemini_thinking_level=_env("GEMINI_THINKING_LEVEL", "minimal").upper(),
            audd_api_token=_env("AUDD_API_TOKEN", required=True),
            audd_timeout_sec=_env_float("AUDD_TIMEOUT_SEC", 8.0),
            lemon_checkout_url=_env("LEMON_CHECKOUT_URL", ""),  # empty = payments off
            lemon_subscription_checkout_url=_env("LEMON_SUBSCRIPTION_CHECKOUT_URL", ""),
            lemon_signing_secret=_env("LEMON_SIGNING_SECRET", ""),
            lemon_subscription_variant_ids=_env_list("LEMON_SUBSCRIPTION_VARIANT_IDS"),
            lemon_credit_pack_variant_ids=_env_list("LEMON_CREDIT_PACK_VARIANT_IDS"),
            paywall_image_url=_env("PAYWALL_IMAGE_URL", ""),
            free_credits=_env_int("FREE_CREDITS", 3),
            credits_per_pack=_env_int("CREDITS_PER_PACK", 50),
            subscription_days=_env_int("SUBSCRIPTION_DAYS", 30),
            max_download_mb=_env_int("MAX_DOWNLOAD_MB", 80),
            max_parallel_jobs=_env_int("MAX_PARALLEL_JOBS", 4),
            # Picks up DMs stuck in "Message requests" (Meta may not send webhooks for them). 0 = off.
            # Safety net for DMs stuck in "Message requests". Each poll is one Graph API call and Meta
            # limits calls per Instagram account (error 80002), so keep this slow. 0 = off.
            inbox_poll_sec=_env_int("INBOX_POLL_SEC", 60),
            tmp_root=_env("TMP_ROOT", "/tmp"),
            # Only for local tests: lets the bot fetch media from localhost / private networks.
            allow_private_urls=_env("ALLOW_PRIVATE_URLS", "").lower() in ("1", "true", "yes"),
            # Optional: cookies.txt of a spare Instagram account (raw or base64) and a proxy
            # (http://user:pass@host:port) — only needed if Instagram starts blocking the server.
            # Secret for the DM command "admin <code> <credits>" (adds test credits to yourself).
            # Empty = admin command disabled. Use 16+ random characters.
            admin_code=_env("ADMIN_CODE", ""),
            # New users pick a language with flag buttons first (0 = everyone gets English).
            ask_language=_env("ASK_LANGUAGE", "1").lower() not in ("0", "false", "no"),
            # Invite a friend → both get this many credits; max rewarded invites per person per day.
            referral_credits=_env_int("REFERRAL_CREDITS", 3),
            referral_daily_cap=_env_int("REFERRAL_DAILY_CAP", 10),
            # Answer @mentions in comments under other people's Reels (free, rate-limited).
            mentions_enabled=_env("MENTIONS_ENABLED", "1").lower() not in ("0", "false", "no"),
            mentions_per_hour=_env_int("MENTIONS_PER_HOUR", 20),
            # Stripe: secret (or restricted) key + webhook signing secret + Price IDs from the dashboard.
            stripe_secret_key=_env("STRIPE_SECRET_KEY", ""),
            stripe_webhook_secret=_env("STRIPE_WEBHOOK_SECRET", ""),
            stripe_prices={code: price for code, price in (
                ("s", _env("STRIPE_PRICE_PACK_S", "")), ("l", _env("STRIPE_PRICE_PACK_L", "")),
                ("m", _env("STRIPE_PRICE_MONTHLY", "")), ("y", _env("STRIPE_PRICE_YEARLY", ""))) if price},
            pack_credits={"s": _env_int("PACK_S_CREDITS", 50), "l": _env_int("PACK_L_CREDITS", 150)},
            stripe_automatic_tax=_env("STRIPE_AUTOMATIC_TAX", "0").lower() in ("1", "true", "yes"),
            # Public https address of this server (Railway fills RAILWAY_PUBLIC_DOMAIN by itself).
            public_url=(_env("PUBLIC_URL", "") or (f"https://{_env('RAILWAY_PUBLIC_DOMAIN', '')}"
                                                    if _env("RAILWAY_PUBLIC_DOMAIN", "") else "")).rstrip("/"),
            # Fair use for Unlimited: max searches per day (protects against bots / resellers).
            sub_daily_limit=_env_int("SUB_DAILY_LIMIT", 100),
            ig_cookies=_env("IG_COOKIES", ""),
            download_proxy=_env("DOWNLOAD_PROXY", ""),
            shazam_enabled=_env("SHAZAM_ENABLED", "1").lower() not in ("0", "false", "no"),
            # Gemini listens to the audio when fingerprinting fails (remixes, covers, edits).
            ai_song_enabled=_env("AI_SONG_ENABLED", "1").lower() not in ("0", "false", "no"),
        )


# =============================================================================
# Constants & user-facing copy (English UI)
# =============================================================================

FIND_MUSIC = "FIND_MUSIC"
FIND_MOVIE = "FIND_MOVIE"
FIND_ITEM = "FIND_ITEM"
FIND_ALL = "FIND_ALL"
FIND_PLACE = "FIND_PLACE"
ACTIONS = {FIND_MUSIC, FIND_MOVIE, FIND_ITEM, FIND_PLACE, FIND_ALL}
ACTION_COST = {FIND_MUSIC: 1, FIND_MOVIE: 1, FIND_ITEM: 1, FIND_PLACE: 1, FIND_ALL: 2}
MY_LISTS = "MY_LISTS"
INVITE = "INVITE"
LIST_KINDS = ("song", "movie", "item", "place")


def default_replies() -> list[dict]:
    """The two buttons that always make sense: your saved finds and free credits via invites."""
    return [{"content_type": "text", "title": "📚 My lists", "payload": MY_LISTS},
            {"content_type": "text", "title": "🎁 Invite friends", "payload": INVITE}]


def followup_replies(done: str) -> list[dict]:
    """Buttons after a search: the other single searches + My lists; after 'Find everything' only
    the two default buttons."""
    if done == FIND_ALL:
        return default_replies()
    return [q for q in QUICK_REPLIES if q["payload"] not in (done, FIND_ALL)]

# =============================================================================
# Languages: the user picks one with flag buttons on first contact (or sends "language").
# Bot texts are translated here; Gemini writes its descriptions in the same language.
# =============================================================================
LANGUAGES = {  # code: (button title, English name for the AI prompt)
    "en": ("🇬🇧 English", "English"),
    "es": ("🇪🇸 Español", "Spanish"),
    "pt": ("🇧🇷 Português", "Brazilian Portuguese"),
    "de": ("🇩🇪 Deutsch", "German"),
    "fr": ("🇫🇷 Français", "French"),
    "it": ("🇮🇹 Italiano", "Italian"),
    "uk": ("🇺🇦 Українська", "Ukrainian"),
    "ru": ("🇷🇺 Русский", "Russian"),
}
_TR_ORDER = ("es", "pt", "de", "fr", "it", "uk", "ru")
TR: dict[str, dict[str, str]] = {}
CURRENT_LANG: contextvars.ContextVar[str] = contextvars.ContextVar("lang", default="en")


def _L(en: str, *others: str) -> str:
    assert len(others) == len(_TR_ORDER), en
    TR[en] = dict(zip(_TR_ORDER, others, strict=True))
    return en


def _t(text: str, **kw: Any) -> str:
    """Translates an English bot text into the current user's language (unknown text → as is)."""
    lang = CURRENT_LANG.get()
    out = TR.get(text, {}).get(lang, text) if lang != "en" else text
    return out.format(**kw) if kw else out


def lang_picker() -> list[dict]:
    return [{"content_type": "text", "title": title, "payload": f"LANG_{code}"}
            for code, (title, _) in LANGUAGES.items()]


MSG_CHOOSE_LANG = "🌍 Choose your language 👇"
LANGUAGE_COMMANDS = {"language", "lang", "languages", "idioma", "língua", "lingua", "sprache", "langue",
                     "мова", "язык", "change language", "/language", "🌍"}
MSG_LANG_SET = _L(
    "✅ English it is! (send “language” to change)",
    "✅ ¡Español activado! (escribe “idioma” para cambiar)",
    "✅ Português ativado! (envie “idioma” para trocar)",
    "✅ Deutsch aktiviert! (schreib „Sprache“ zum Ändern)",
    "✅ Français activé ! (envoie « langue » pour changer)",
    "✅ Italiano attivato! (scrivi “lingua” per cambiare)",
    "✅ Українську увімкнено! (напиши «мова», щоб змінити)",
    "✅ Русский включён! (напиши «язык», чтобы сменить)",
)
# Quick-reply titles (Instagram limit: 20 characters)
_L("🎵 Find Song", "🎵 Buscar canción", "🎵 Achar música", "🎵 Song finden", "🎵 La chanson",
   "🎵 Trova canzone", "🎵 Знайти пісню", "🎵 Найти песню")
_L("🎬 Find Movie", "🎬 Buscar película", "🎬 Achar filme", "🎬 Film finden", "🎬 Le film",
   "🎬 Trova film", "🎬 Знайти фільм", "🎬 Найти фильм")
_L("👕 Find Outfit/Item", "👕 Ropa/objetos", "👕 Roupa/itens", "👕 Outfit/Sachen", "👕 Tenue/objets",
   "👕 Outfit/oggetti", "👕 Одяг/речі", "👕 Одежда/вещи")
_L("🔥 Find Everything", "🔥 Buscar todo", "🔥 Achar tudo", "🔥 Alles finden", "🔥 Tout trouver",
   "🔥 Trova tutto", "🔥 Знайти все", "🔥 Найти всё")
# Link-button cards
_L("📺 Where to watch", "📺 Dónde ver", "📺 Onde assistir", "📺 Wo streamen", "📺 Où regarder",
   "📺 Dove guardarlo", "📺 Де дивитися", "📺 Где смотреть")
_L("🎞 Trailer", "🎞 Tráiler", "🎞 Trailer", "🎞 Trailer", "🎞 Bande-annonce", "🎞 Trailer",
   "🎞 Трейлер", "🎞 Трейлер")
_L("Where to watch & details", "Dónde verla y detalles", "Onde assistir e detalhes", "Wo streamen & Details",
   "Où regarder & détails", "Dove guardarlo e dettagli", "Де дивитися та деталі", "Где смотреть и подробности")
# Result texts
_L("🎵 Song found!", "🎵 ¡Canción encontrada!", "🎵 Música encontrada!", "🎵 Song gefunden!",
   "🎵 Chanson trouvée !", "🎵 Canzone trovata!", "🎵 Пісню знайдено!", "🎵 Песня найдена!")
_L("⚡ This Reel uses a sped-up version of:", "⚡ Este Reel usa una versión acelerada de:",
   "⚡ Este Reel usa uma versão acelerada de:", "⚡ Dieses Reel nutzt eine schnellere Version von:",
   "⚡ Ce Reel utilise une version accélérée de :", "⚡ Questo Reel usa una versione accelerata di:",
   "⚡ У цьому Reels прискорена версія:", "⚡ В этом Reels ускоренная версия:")
_L("🐢 This Reel uses a slowed-down version of:", "🐢 Este Reel usa una versión ralentizada de:",
   "🐢 Este Reel usa uma versão desacelerada de:", "🐢 Dieses Reel nutzt eine langsamere Version von:",
   "🐢 Ce Reel utilise une version ralentie de :", "🐢 Questo Reel usa una versione rallentata di:",
   "🐢 У цьому Reels уповільнена версія:", "🐢 В этом Reels замедленная версия:")
_L("🎛 This Reel uses a {v} of:", "🎛 Este Reel usa una versión ({v}) de:", "🎛 Este Reel usa uma versão ({v}) de:",
   "🎛 Dieses Reel nutzt eine Version ({v}) von:", "🎛 Ce Reel utilise une version ({v}) de :",
   "🎛 Questo Reel usa una versione ({v}) di:", "🎛 У цьому Reels версія ({v}):", "🎛 В этом Reels версия ({v}):")
_L("🤔 Most likely:", "🤔 Lo más probable:", "🤔 Provavelmente:", "🤔 Wahrscheinlich:", "🤔 Très probablement :",
   "🤔 Molto probabilmente:", "🤔 Найімовірніше:", "🤔 Скорее всего:")
_L("🎬 It's from:", "🎬 Es de:", "🎬 É de:", "🎬 Das ist aus:", "🎬 Ça vient de :", "🎬 Viene da:",
   "🎬 Це з:", "🎬 Это из:")
_L("Movie", "Película", "Filme", "Film", "Film", "Film", "Фільм", "Фильм")
_L("TV series", "Serie", "Série", "Serie", "Série", "Serie TV", "Серіал", "Сериал")
_L("Anime", "Anime", "Anime", "Anime", "Anime", "Anime", "Аніме", "Аниме")
_L("Documentary", "Documental", "Documentário", "Doku", "Documentaire", "Documentario", "Документальний",
   "Документальный")
_L("Music video", "Videoclip", "Videoclipe", "Musikvideo", "Clip", "Video musicale", "Кліп", "Клип")
_L("🎭 Cast: {x}", "🎭 Reparto: {x}", "🎭 Elenco: {x}", "🎭 Besetzung: {x}", "🎭 Casting : {x}", "🎭 Cast: {x}",
   "🎭 У ролях: {x}", "🎭 В ролях: {x}")
_L("📍 Scene: {x}", "📍 Escena: {x}", "📍 Cena: {x}", "📍 Szene: {x}", "📍 Scène : {x}", "📍 Scena: {x}",
   "📍 Сцена: {x}", "📍 Сцена: {x}")
_L("⏱ Moment: {x}", "⏱ Momento: {x}", "⏱ Momento: {x}", "⏱ Moment: {x}", "⏱ Moment : {x}", "⏱ Momento: {x}",
   "⏱ Момент: {x}", "⏱ Момент: {x}")
_L("🎯 Confidence: high", "🎯 Confianza: alta", "🎯 Confiança: alta", "🎯 Sicherheit: hoch",
   "🎯 Confiance : élevée", "🎯 Affidabilità: alta", "🎯 Впевненість: висока", "🎯 Уверенность: высокая")
_L("🎯 Confidence: medium", "🎯 Confianza: media", "🎯 Confiança: média", "🎯 Sicherheit: mittel",
   "🎯 Confiance : moyenne", "🎯 Affidabilità: media", "🎯 Впевненість: середня", "🎯 Уверенность: средняя")
_L("👕 Items spotted:", "👕 Objetos encontrados:", "👕 Itens encontrados:", "👕 Gefundene Sachen:",
   "👕 Objets repérés :", "👕 Oggetti trovati:", "👕 Знайдені речі:", "👕 Найденные вещи:")
_L("🎵 Song: search temporarily unavailable", "🎵 Canción: búsqueda no disponible por ahora",
   "🎵 Música: busca indisponível no momento", "🎵 Song: Suche gerade nicht verfügbar",
   "🎵 Chanson : recherche indisponible pour l'instant", "🎵 Canzone: ricerca momentaneamente non disponibile",
   "🎵 Пісня: пошук тимчасово недоступний", "🎵 Песня: поиск временно недоступен")
_L("🎵 Song: not recognised", "🎵 Canción: no reconocida", "🎵 Música: não reconhecida", "🎵 Song: nicht erkannt",
   "🎵 Chanson : non reconnue", "🎵 Canzone: non riconosciuta", "🎵 Пісня: не розпізнано",
   "🎵 Песня: не распознана")
_L("🎬 Movie & 👕 items: temporarily unavailable", "🎬 Película y 👕 objetos: no disponible por ahora",
   "🎬 Filme e 👕 itens: indisponível no momento", "🎬 Film & 👕 Sachen: gerade nicht verfügbar",
   "🎬 Film & 👕 objets : indisponible pour l'instant", "🎬 Film e 👕 oggetti: momentaneamente non disponibile",
   "🎬 Фільм і 👕 речі: тимчасово недоступно", "🎬 Фильм и 👕 вещи: временно недоступно")
_L("🎬 Movie & 👕 items: nothing identifiable", "🎬 Película y 👕 objetos: nada identificable",
   "🎬 Filme e 👕 itens: nada identificável", "🎬 Film & 👕 Sachen: nichts Erkennbares",
   "🎬 Film & 👕 objets : rien d'identifiable", "🎬 Film e 👕 oggetti: niente di riconoscibile",
   "🎬 Фільм і 👕 речі: нічого не знайдено", "🎬 Фильм и 👕 вещи: ничего не найдено")
_L("🎵 Song: no sound in this media", "🎵 Canción: este archivo no tiene sonido", "🎵 Música: sem som nesta mídia",
   "🎵 Song: kein Ton vorhanden", "🎵 Chanson : pas de son", "🎵 Canzone: nessun audio", "🎵 Пісня: немає звуку",
   "🎵 Песня: нет звука")
_L("💳 Credits left: {n}", "💳 Créditos restantes: {n}", "💳 Créditos restantes: {n}", "💳 Credits übrig: {n}",
   "💳 Crédits restants : {n}", "💳 Crediti rimasti: {n}", "💳 Залишилось кредитів: {n}",
   "💳 Осталось кредитов: {n}")
_L("⭐ Unlimited until {date}", "⭐ Ilimitado hasta {date}", "⭐ Ilimitado até {date}", "⭐ Unlimited bis {date}",
   "⭐ Illimité jusqu'au {date}", "⭐ Illimitato fino al {date}", "⭐ Безліміт до {date}", "⭐ Безлимит до {date}")
_L("Want something else from this Reel? 👇", "¿Quieres buscar algo más en este Reel? 👇",
   "Quer achar mais alguma coisa neste Reel? 👇", "Noch was anderes aus diesem Reel? 👇",
   "Autre chose à trouver dans ce Reel ? 👇", "Vuoi trovare altro in questo Reel? 👇",
   "Знайти ще щось у цьому Reels? 👇", "Найти ещё что-то в этом Reels? 👇")
_L("Oops, I couldn't save that one 😕 Please send it again in a moment.",
   "Uy, no pude guardarlo 😕 Mándalo de nuevo en un momento.", "Ops, não consegui salvar 😕 Mande de novo daqui a pouco.",
   "Ups, das konnte ich nicht speichern 😕 Schick es gleich nochmal.",
   "Oups, je n'ai pas pu l'enregistrer 😕 Renvoie-le dans un instant.", "Ops, non sono riuscito a salvarlo 😕 Rimandalo tra poco.",
   "Ой, не вдалося зберегти 😕 Надішли ще раз за мить.", "Ой, не получилось сохранить 😕 Пришли ещё раз через минутку.")
_L("🎁 Code applied: +{credits} credits!\n💳 Credits left: {balance}\n\nSend me a Reel to use them 👇",
   "🎁 ¡Código aplicado: +{credits} créditos!\n💳 Créditos restantes: {balance}\n\nMándame un Reel para usarlos 👇",
   "🎁 Código aplicado: +{credits} créditos!\n💳 Créditos restantes: {balance}\n\nMande um Reel para usar 👇",
   "🎁 Code eingelöst: +{credits} Credits!\n💳 Credits übrig: {balance}\n\nSchick mir ein Reel, um sie zu nutzen 👇",
   "🎁 Code appliqué : +{credits} crédits !\n💳 Crédits restants : {balance}\n\nEnvoie-moi un Reel pour les utiliser 👇",
   "🎁 Codice applicato: +{credits} crediti!\n💳 Crediti rimasti: {balance}\n\nMandami un Reel per usarli 👇",
   "🎁 Код активовано: +{credits} кредитів!\n💳 Залишилось кредитів: {balance}\n\nНадішли Reels, щоб їх використати 👇",
   "🎁 Код активирован: +{credits} кредитов!\n💳 Осталось кредитов: {balance}\n\nПришли Reels, чтобы их потратить 👇")
_L("You've already used this code 🙂", "Ya usaste este código 🙂", "Você já usou este código 🙂",
   "Diesen Code hast du schon benutzt 🙂", "Tu as déjà utilisé ce code 🙂", "Hai già usato questo codice 🙂",
   "Ти вже використав цей код 🙂", "Ты уже использовал этот код 🙂")
_L("This code has expired or run out 😕", "Este código caducó o se agotó 😕", "Este código expirou ou esgotou 😕",
   "Dieser Code ist abgelaufen oder aufgebraucht 😕", "Ce code a expiré ou est épuisé 😕",
   "Questo codice è scaduto o esaurito 😕", "Цей код закінчився або вичерпаний 😕", "Этот код истёк или закончился 😕")
_L("Hey there! 👋 I'm {bot_name}.\n\nSend me any Reel (tap ✈️ under the Reel → send it to me), a video or a photo, "
   "and in a few seconds I'll find:\n🎵 the song\n🎬 the movie or TV show\n👕 the outfit & items\n📍 the place\n\n{balance}",
   "¡Hola! 👋 Soy {bot_name}.\n\nMándame cualquier Reel (toca ✈️ debajo del Reel → envíamelo), un video o una foto, "
   "y en segundos encontraré:\n🎵 la canción\n🎬 la película o serie\n👕 la ropa y los objetos\n📍 el lugar\n\n{balance}",
   "Oi! 👋 Eu sou o {bot_name}.\n\nMande qualquer Reel (toque em ✈️ embaixo do Reel → envie pra mim), um vídeo ou "
   "uma foto, e em segundos eu acho:\n🎵 a música\n🎬 o filme ou série\n👕 a roupa e os itens\n📍 o lugar\n\n{balance}",
   "Hey! 👋 Ich bin {bot_name}.\n\nSchick mir ein Reel (✈️ unter dem Reel → an mich senden), ein Video oder ein Foto, "
   "und in Sekunden finde ich:\n🎵 den Song\n🎬 den Film oder die Serie\n👕 das Outfit & die Sachen\n📍 den Ort\n\n{balance}",
   "Salut ! 👋 Je suis {bot_name}.\n\nEnvoie-moi un Reel (✈️ sous le Reel → envoie-le-moi), une vidéo ou une photo, "
   "et en quelques secondes je trouve :\n🎵 la chanson\n🎬 le film ou la série\n👕 la tenue et les objets\n📍 le lieu\n\n{balance}",
   "Ciao! 👋 Sono {bot_name}.\n\nMandami un Reel (tocca ✈️ sotto il Reel → inviamelo), un video o una foto, "
   "e in pochi secondi trovo:\n🎵 la canzone\n🎬 il film o la serie\n👕 l'outfit e gli oggetti\n📍 il luogo\n\n{balance}",
   "Привіт! 👋 Я {bot_name}.\n\nНадішли мені будь-який Reels (✈️ під Reels → мені), відео або фото, "
   "і за кілька секунд я знайду:\n🎵 пісню\n🎬 фільм або серіал\n👕 одяг і речі\n📍 місце\n\n{balance}",
   "Привет! 👋 Я {bot_name}.\n\nПришли мне любой Reels (✈️ под Reels → мне), видео или фото, "
   "и за пару секунд я найду:\n🎵 песню\n🎬 фильм или сериал\n👕 одежду и вещи\n📍 место\n\n{balance}")

_L("📍 Find Place", "📍 Buscar lugar", "📍 Achar lugar", "📍 Ort finden", "📍 Le lieu", "📍 Trova luogo",
   "📍 Знайти місце", "📍 Найти место")
_L("📚 My lists", "📚 Mis listas", "📚 Minhas listas", "📚 Meine Listen", "📚 Mes listes", "📚 Le mie liste",
   "📚 Мої списки", "📚 Мои списки")
_L("🎁 Invite friends", "🎁 Invitar amigos", "🎁 Convidar amigos", "🎁 Freunde einladen", "🎁 Inviter des amis",
   "🎁 Invita amici", "🎁 Запросити друзів", "🎁 Пригласить друзей")
_L("📍 Place found!", "📍 ¡Lugar encontrado!", "📍 Lugar encontrado!", "📍 Ort gefunden!", "📍 Lieu trouvé !",
   "📍 Luogo trovato!", "📍 Місце знайдено!", "📍 Место найдено!")
_L("🔎 Photos & info", "🔎 Fotos e info", "🔎 Fotos e info", "🔎 Fotos & Infos", "🔎 Photos & infos",
   "🔎 Foto e info", "🔎 Фото та інфо", "🔎 Фото и инфо")
_L("🎬 Movie, 👕 items & 📍 place: temporarily unavailable", "🎬 Película, 👕 objetos y 📍 lugar: no disponible por ahora",
   "🎬 Filme, 👕 itens e 📍 lugar: indisponível no momento", "🎬 Film, 👕 Sachen & 📍 Ort: gerade nicht verfügbar",
   "🎬 Film, 👕 objets & 📍 lieu : indisponible pour l'instant", "🎬 Film, 👕 oggetti e 📍 luogo: momentaneamente non disponibile",
   "🎬 Фільм, 👕 речі та 📍 місце: тимчасово недоступно", "🎬 Фильм, 👕 вещи и 📍 место: временно недоступно")
_L("🎬 Movie, 👕 items & 📍 place: nothing identifiable", "🎬 Película, 👕 objetos y 📍 lugar: nada identificable",
   "🎬 Filme, 👕 itens e 📍 lugar: nada identificável", "🎬 Film, 👕 Sachen & 📍 Ort: nichts Erkennbares",
   "🎬 Film, 👕 objets & 📍 lieu : rien d'identifiable", "🎬 Film, 👕 oggetti e 📍 luogo: niente di riconoscibile",
   "🎬 Фільм, 👕 речі та 📍 місце: нічого не знайдено", "🎬 Фильм, 👕 вещи и 📍 место: ничего не найдено")
_L("🎁 Invite friends and get +{n} free searches for each friend who joins — your friend gets +{n} too!\nJust forward them the next message 👇",
   "🎁 Invita a tus amigos y gana +{n} búsquedas gratis por cada amigo que se una. ¡Tu amigo también recibe +{n}!\nSolo reenvíale el siguiente mensaje 👇",
   "🎁 Convide amigos e ganhe +{n} buscas grátis por cada amigo que entrar — seu amigo também ganha +{n}!\nÉ só encaminhar a próxima mensagem 👇",
   "🎁 Lade Freunde ein und bekomm +{n} Gratis-Suchen für jeden Freund, der mitmacht – dein Freund bekommt auch +{n}!\nLeite ihm einfach die nächste Nachricht weiter 👇",
   "🎁 Invite tes amis et gagne +{n} recherches gratuites pour chaque ami qui rejoint — ton ami reçoit aussi +{n} !\nTransfère-lui simplement le message suivant 👇",
   "🎁 Invita gli amici e ottieni +{n} ricerche gratis per ogni amico che si unisce: anche il tuo amico riceve +{n}!\nInoltragli il prossimo messaggio 👇",
   "🎁 Запрошуй друзів і отримуй +{n} безкоштовних пошуків за кожного, хто приєднається — друг теж отримає +{n}!\nПросто перешли йому наступне повідомлення 👇",
   "🎁 Приглашай друзей и получай +{n} бесплатных поисков за каждого, кто присоединится — друг тоже получит +{n}!\nПросто перешли ему следующее сообщение 👇")
_L("🔥 This bot finds the song, movie, outfit & place from any Instagram Reel — right in DM!\n\n1️⃣ Open {link}\n2️⃣ Send it this code: {code}\n\nWe both get +{n} free searches 🎁",
   "🔥 Este bot encuentra la canción, la película, la ropa y el lugar de cualquier Reel de Instagram, ¡directo en DM!\n\n1️⃣ Abre {link}\n2️⃣ Envíale este código: {code}\n\nLos dos recibimos +{n} búsquedas gratis 🎁",
   "🔥 Esse bot acha a música, o filme, a roupa e o lugar de qualquer Reel do Instagram — direto na DM!\n\n1️⃣ Abra {link}\n2️⃣ Mande este código: {code}\n\nNós dois ganhamos +{n} buscas grátis 🎁",
   "🔥 Dieser Bot findet Song, Film, Outfit & Ort aus jedem Instagram-Reel – direkt per DM!\n\n1️⃣ Öffne {link}\n2️⃣ Schick ihm diesen Code: {code}\n\nWir beide bekommen +{n} Gratis-Suchen 🎁",
   "🔥 Ce bot trouve la chanson, le film, la tenue et le lieu de n'importe quel Reel Instagram, directement en DM !\n\n1️⃣ Ouvre {link}\n2️⃣ Envoie-lui ce code : {code}\n\nOn reçoit tous les deux +{n} recherches gratuites 🎁",
   "🔥 Questo bot trova la canzone, il film, l'outfit e il luogo di qualsiasi Reel di Instagram, direttamente in DM!\n\n1️⃣ Apri {link}\n2️⃣ Mandagli questo codice: {code}\n\nRiceviamo entrambi +{n} ricerche gratis 🎁",
   "🔥 Цей бот знаходить пісню, фільм, одяг і місце з будь-якого Reels в Instagram — прямо в директі!\n\n1️⃣ Відкрий {link}\n2️⃣ Надішли йому код: {code}\n\nМи обидва отримаємо +{n} безкоштовних пошуків 🎁",
   "🔥 Этот бот находит песню, фильм, одежду и место из любого Reels в Instagram — прямо в директе!\n\n1️⃣ Открой {link}\n2️⃣ Отправь ему код: {code}\n\nМы оба получим +{n} бесплатных поисков 🎁")
_L("🔋 That was your last free search.", "🔋 Esa fue tu última búsqueda gratis.", "🔋 Essa foi sua última busca grátis.",
   "🔋 Das war deine letzte Gratis-Suche.", "🔋 C'était ta dernière recherche gratuite.", "🔋 Era la tua ultima ricerca gratuita.",
   "🔋 Це був твій останній безкоштовний пошук.", "🔋 Это был твой последний бесплатный поиск.")
_L("🎁 Invite code accepted: +{credits} credits!\n💳 Credits left: {balance}\n\nSend me a Reel 👇",
   "🎁 ¡Código de invitación aceptado: +{credits} créditos!\n💳 Créditos restantes: {balance}\n\nMándame un Reel 👇",
   "🎁 Código de convite aceito: +{credits} créditos!\n💳 Créditos restantes: {balance}\n\nMande um Reel 👇",
   "🎁 Einladungscode angenommen: +{credits} Credits!\n💳 Credits übrig: {balance}\n\nSchick mir ein Reel 👇",
   "🎁 Code d'invitation accepté : +{credits} crédits !\n💳 Crédits restants : {balance}\n\nEnvoie-moi un Reel 👇",
   "🎁 Codice invito accettato: +{credits} crediti!\n💳 Crediti rimasti: {balance}\n\nMandami un Reel 👇",
   "🎁 Код запрошення прийнято: +{credits} кредитів!\n💳 Залишилось кредитів: {balance}\n\nНадішли Reels 👇",
   "🎁 Код приглашения принят: +{credits} кредитов!\n💳 Осталось кредитов: {balance}\n\nПришли Reels 👇")
_L("🎉 A friend joined with your invite code! +{credits} credits\n💳 Credits left: {balance}",
   "🎉 ¡Un amigo se unió con tu código! +{credits} créditos\n💳 Créditos restantes: {balance}",
   "🎉 Um amigo entrou com seu código! +{credits} créditos\n💳 Créditos restantes: {balance}",
   "🎉 Ein Freund ist mit deinem Code dabei! +{credits} Credits\n💳 Credits übrig: {balance}",
   "🎉 Un ami a rejoint avec ton code ! +{credits} crédits\n💳 Crédits restants : {balance}",
   "🎉 Un amico si è unito con il tuo codice! +{credits} crediti\n💳 Crediti rimasti: {balance}",
   "🎉 Друг приєднався за твоїм кодом! +{credits} кредитів\n💳 Залишилось кредитів: {balance}",
   "🎉 Друг присоединился по твоему коду! +{credits} кредитов\n💳 Осталось кредитов: {balance}")
_L("That's your own invite code 😄 Share it with friends!", "Ese es tu propio código 😄 ¡Compártelo con tus amigos!",
   "Esse é o seu próprio código 😄 Compartilhe com os amigos!", "Das ist dein eigener Code 😄 Teil ihn mit Freunden!",
   "C'est ton propre code 😄 Partage-le avec tes amis !", "Questo è il tuo codice 😄 Condividilo con gli amici!",
   "Це твій власний код 😄 Поділися ним із друзями!", "Это твой собственный код 😄 Поделись им с друзьями!")
_L("You've already used an invite code 🙂", "Ya usaste un código de invitación 🙂", "Você já usou um código de convite 🙂",
   "Du hast schon einen Einladungscode benutzt 🙂", "Tu as déjà utilisé un code d'invitation 🙂", "Hai già usato un codice invito 🙂",
   "Ти вже використав код запрошення 🙂", "Ты уже использовал код приглашения 🙂")
_L("Invite codes are only for new users 🙂", "Los códigos de invitación son solo para usuarios nuevos 🙂",
   "Códigos de convite são só para novos usuários 🙂", "Einladungscodes gibt's nur für neue Nutzer 🙂",
   "Les codes d'invitation sont réservés aux nouveaux utilisateurs 🙂", "I codici invito sono solo per i nuovi utenti 🙂",
   "Коди запрошення лише для нових користувачів 🙂", "Коды приглашения только для новых пользователей 🙂")
_L("This invite code has reached today's limit — try again tomorrow 🙂", "Este código alcanzó el límite de hoy, inténtalo mañana 🙂",
   "Este código atingiu o limite de hoje — tente amanhã 🙂", "Dieser Code hat das Tageslimit erreicht – versuch es morgen nochmal 🙂",
   "Ce code a atteint la limite du jour — réessaie demain 🙂", "Questo codice ha raggiunto il limite di oggi: riprova domani 🙂",
   "Цей код вичерпав ліміт на сьогодні — спробуй завтра 🙂", "Этот код исчерпал лимит на сегодня — попробуй завтра 🙂")
_L("📚 Your lists — what should I show?", "📚 Tus listas: ¿qué te muestro?", "📚 Suas listas — o que eu mostro?",
   "📚 Deine Listen – was soll ich zeigen?", "📚 Tes listes — qu'est-ce que je t'affiche ?", "📚 Le tue liste: cosa ti mostro?",
   "📚 Твої списки — що показати?", "📚 Твои списки — что показать?")
_L("📚 Your lists are empty for now — every song, movie, item and place I find for you is saved here automatically.",
   "📚 Tus listas están vacías por ahora: cada canción, película, objeto y lugar que encuentre se guardará aquí automáticamente.",
   "📚 Suas listas estão vazias por enquanto — cada música, filme, item e lugar que eu achar fica salvo aqui automaticamente.",
   "📚 Deine Listen sind noch leer – jeder Song, Film, jede Sache und jeder Ort, den ich finde, landet automatisch hier.",
   "📚 Tes listes sont vides pour l'instant — chaque chanson, film, objet et lieu que je trouve y est enregistré automatiquement.",
   "📚 Le tue liste sono vuote per ora: ogni canzone, film, oggetto e luogo che trovo viene salvato qui automaticamente.",
   "📚 Твої списки поки порожні — кожна пісня, фільм, річ і місце, які я знайду, зберігаються тут автоматично.",
   "📚 Твои списки пока пусты — каждая песня, фильм, вещь и место, которые я найду, сохраняются здесь автоматически.")
_L("🎵 Songs ({n})", "🎵 Canciones ({n})", "🎵 Músicas ({n})", "🎵 Songs ({n})", "🎵 Chansons ({n})", "🎵 Canzoni ({n})",
   "🎵 Пісні ({n})", "🎵 Песни ({n})")
_L("🎬 Movies ({n})", "🎬 Películas ({n})", "🎬 Filmes ({n})", "🎬 Filme ({n})", "🎬 Films ({n})", "🎬 Film ({n})",
   "🎬 Фільми ({n})", "🎬 Фильмы ({n})")
_L("👕 Items ({n})", "👕 Objetos ({n})", "👕 Itens ({n})", "👕 Sachen ({n})", "👕 Objets ({n})", "👕 Oggetti ({n})",
   "👕 Речі ({n})", "👕 Вещи ({n})")
_L("📍 Places ({n})", "📍 Lugares ({n})", "📍 Lugares ({n})", "📍 Orte ({n})", "📍 Lieux ({n})", "📍 Luoghi ({n})",
   "📍 Місця ({n})", "📍 Места ({n})")
_L("🎵 Your songs ({n}):", "🎵 Tus canciones ({n}):", "🎵 Suas músicas ({n}):", "🎵 Deine Songs ({n}):",
   "🎵 Tes chansons ({n}) :", "🎵 Le tue canzoni ({n}):", "🎵 Твої пісні ({n}):", "🎵 Твои песни ({n}):")
_L("🎬 Your movies & shows ({n}):", "🎬 Tus películas y series ({n}):", "🎬 Seus filmes e séries ({n}):",
   "🎬 Deine Filme & Serien ({n}):", "🎬 Tes films & séries ({n}) :", "🎬 I tuoi film e serie ({n}):",
   "🎬 Твої фільми та серіали ({n}):", "🎬 Твои фильмы и сериалы ({n}):")
_L("👕 Your items ({n}):", "👕 Tus objetos ({n}):", "👕 Seus itens ({n}):", "👕 Deine Sachen ({n}):",
   "👕 Tes objets ({n}) :", "👕 I tuoi oggetti ({n}):", "👕 Твої речі ({n}):", "👕 Твои вещи ({n}):")
_L("📍 Your places ({n}):", "📍 Tus lugares ({n}):", "📍 Seus lugares ({n}):", "📍 Deine Orte ({n}):",
   "📍 Tes lieux ({n}) :", "📍 I tuoi luoghi ({n}):", "📍 Твої місця ({n}):", "📍 Твои места ({n}):")
_L("🔗 Open", "🔗 Abrir", "🔗 Abrir", "🔗 Öffnen", "🔗 Ouvrir", "🔗 Apri", "🔗 Відкрити", "🔗 Открыть")
_L("👋 DM me any Reel — I find songs, movies, outfits & places!",
   "👋 Mándame cualquier Reel por DM: encuentro canciones, películas, ropa y lugares.",
   "👋 Me mande qualquer Reel na DM — eu acho músicas, filmes, roupas e lugares!",
   "👋 Schick mir ein Reel per DM – ich finde Songs, Filme, Outfits & Orte!",
   "👋 Envoie-moi n'importe quel Reel en DM — je trouve chansons, films, tenues et lieux !",
   "👋 Mandami un Reel in DM: trovo canzoni, film, outfit e luoghi!",
   "👋 Надішли мені будь-який Reels у директ — знаходжу пісні, фільми, одяг і місця!",
   "👋 Пришли мне любой Reels в директ — нахожу песни, фильмы, одежду и места!")

_L("🎬 Movie/show: not identified", "🎬 Película/serie: no identificada", "🎬 Filme/série: não identificado",
   "🎬 Film/Serie: nicht erkannt", "🎬 Film/série : non identifié", "🎬 Film/serie: non identificato",
   "🎬 Фільм/серіал: не визначено", "🎬 Фильм/сериал: не определён")
_L("👕 Items: nothing identifiable", "👕 Objetos: nada identificable", "👕 Itens: nada identificável",
   "👕 Sachen: nichts Erkennbares", "👕 Objets : rien d'identifiable", "👕 Oggetti: niente di riconoscibile",
   "👕 Речі: нічого не знайдено", "👕 Вещи: ничего не найдено")
_L("📍 Place: not recognised", "📍 Lugar: no reconocido", "📍 Lugar: não reconhecido", "📍 Ort: nicht erkannt",
   "📍 Lieu : non reconnu", "📍 Luogo: non riconosciuto", "📍 Місце: не впізнано", "📍 Место: не распознано")

_L("💳 Choose a plan 👇 You'll pay on a secure Stripe page — the purchase is linked to your Instagram account automatically.",
   "💳 Elige un plan 👇 Pagarás en una página segura de Stripe y la compra se vincula a tu cuenta de Instagram automáticamente.",
   "💳 Escolha um plano 👇 Você paga numa página segura do Stripe — a compra é vinculada à sua conta do Instagram automaticamente.",
   "💳 Wähle einen Plan 👇 Du zahlst auf einer sicheren Stripe-Seite – der Kauf wird automatisch mit deinem Instagram-Konto verknüpft.",
   "💳 Choisis une offre 👇 Tu paies sur une page Stripe sécurisée — l'achat est lié automatiquement à ton compte Instagram.",
   "💳 Scegli un piano 👇 Paghi su una pagina sicura di Stripe e l'acquisto viene collegato automaticamente al tuo account Instagram.",
   "💳 Обери тариф 👇 Оплата на захищеній сторінці Stripe — покупка автоматично прив'язується до твого акаунта Instagram.",
   "💳 Выбери тариф 👇 Оплата на защищённой странице Stripe — покупка автоматически привяжется к твоему аккаунту Instagram.")
_L("🔋 You're out of credits.", "🔋 Te quedaste sin créditos.", "🔋 Seus créditos acabaram.", "🔋 Deine Credits sind aufgebraucht.",
   "🔋 Tu n'as plus de crédits.", "🔋 Hai finito i crediti.", "🔋 Кредити закінчилися.", "🔋 Кредиты закончились.")
_L("⚡ {n} searches", "⚡ {n} búsquedas", "⚡ {n} buscas", "⚡ {n} Suchen", "⚡ {n} recherches", "⚡ {n} ricerche",
   "⚡ {n} пошуків", "⚡ {n} поисков")
_L("One-time purchase • credits never expire", "Pago único • los créditos no caducan", "Pagamento único • créditos não expiram",
   "Einmalzahlung • Credits verfallen nicht", "Achat unique • les crédits n'expirent pas", "Acquisto singolo • i crediti non scadono",
   "Разова покупка • кредити не згорають", "Разовая покупка • кредиты не сгорают")
_L("♾ Unlimited · 1 month", "♾ Ilimitado · 1 mes", "♾ Ilimitado · 1 mês", "♾ Unlimited · 1 Monat", "♾ Illimité · 1 mois",
   "♾ Illimitato · 1 mese", "♾ Безліміт · 1 місяць", "♾ Безлимит · 1 месяц")
_L("♾ Unlimited · 1 year", "♾ Ilimitado · 1 año", "♾ Ilimitado · 1 ano", "♾ Unlimited · 1 Jahr", "♾ Illimité · 1 an",
   "♾ Illimitato · 1 anno", "♾ Безліміт · 1 рік", "♾ Безлимит · 1 год")
_L("Search as much as you like • cancel anytime", "Busca todo lo que quieras • cancela cuando quieras",
   "Busque à vontade • cancele quando quiser", "So viel suchen, wie du willst • jederzeit kündbar",
   "Cherche autant que tu veux • résiliable à tout moment", "Cerca quanto vuoi • disdici quando vuoi",
   "Шукай скільки завгодно • скасування будь-коли", "Ищи сколько угодно • отмена в любой момент")
_L("Best value • cancel anytime", "La mejor opción • cancela cuando quieras", "Melhor custo-benefício • cancele quando quiser",
   "Am günstigsten • jederzeit kündbar", "Le meilleur prix • résiliable à tout moment", "Il più conveniente • disdici quando vuoi",
   "Найвигідніше • скасування будь-коли", "Выгоднее всего • отмена в любой момент")
_L("Buy · {price}", "Comprar · {price}", "Comprar · {price}", "Kaufen · {price}", "Acheter · {price}", "Acquista · {price}",
   "Купити · {price}", "Купить · {price}")
_L("Subscribe · {price}", "Suscribirse · {price}", "Assinar · {price}", "Abonnieren · {price}", "S'abonner · {price}",
   "Abbonati · {price}", "Підписатися · {price}", "Подписаться · {price}")
_L("⭐ Your Unlimited plan", "⭐ Tu plan Ilimitado", "⭐ Seu plano Ilimitado", "⭐ Dein Unlimited-Abo", "⭐ Ton offre Illimitée",
   "⭐ Il tuo piano Illimitato", "⭐ Твій Безліміт", "⭐ Твой Безлимит")
_L("Active until {date} • change or cancel", "Activo hasta {date} • cambiar o cancelar", "Ativo até {date} • alterar ou cancelar",
   "Aktiv bis {date} • ändern oder kündigen", "Actif jusqu'au {date} • modifier ou résilier", "Attivo fino al {date} • modifica o disdici",
   "Активний до {date} • змінити або скасувати", "Активен до {date} • изменить или отменить")
_L("⚙️ Manage", "⚙️ Gestionar", "⚙️ Gerenciar", "⚙️ Verwalten", "⚙️ Gérer", "⚙️ Gestisci", "⚙️ Керувати", "⚙️ Управлять")
_L("You've reached today's fair-use limit for Unlimited ({n} searches) — it resets tomorrow 🙂",
   "Llegaste al límite de uso justo de hoy del plan Ilimitado ({n} búsquedas): se reinicia mañana 🙂",
   "Você atingiu o limite de uso justo de hoje do Ilimitado ({n} buscas) — ele reinicia amanhã 🙂",
   "Du hast das heutige Fair-Use-Limit von Unlimited erreicht ({n} Suchen) – morgen geht's weiter 🙂",
   "Tu as atteint la limite d'utilisation équitable du jour pour l'offre Illimitée ({n} recherches) — elle se réinitialise demain 🙂",
   "Hai raggiunto il limite giornaliero di uso corretto dell'Illimitato ({n} ricerche): si azzera domani 🙂",
   "Ти досяг денного ліміту чесного використання Безліміту ({n} пошуків) — завтра він оновиться 🙂",
   "Ты достиг дневного лимита честного использования Безлимита ({n} поисков) — завтра он обновится 🙂")
_L("✅ Payment received!", "✅ ¡Pago recibido!", "✅ Pagamento recebido!", "✅ Zahlung erhalten!", "✅ Paiement reçu !",
   "✅ Pagamento ricevuto!", "✅ Оплату отримано!", "✅ Оплата получена!")
_L("Go back to Instagram — your purchase is already active.", "Vuelve a Instagram: tu compra ya está activa.",
   "Volte para o Instagram — sua compra já está ativa.", "Geh zurück zu Instagram – dein Kauf ist schon aktiv.",
   "Retourne sur Instagram — ton achat est déjà actif.", "Torna su Instagram: il tuo acquisto è già attivo.",
   "Повертайся в Instagram — покупка вже активна.", "Возвращайся в Instagram — покупка уже активна.")
_L("Payment cancelled — nothing was charged.", "Pago cancelado: no se cobró nada.", "Pagamento cancelado — nada foi cobrado.",
   "Zahlung abgebrochen – es wurde nichts berechnet.", "Paiement annulé — rien n'a été débité.", "Pagamento annullato: nessun addebito.",
   "Оплату скасовано — нічого не списано.", "Оплата отменена — ничего не списано.")
_L("Back to Instagram", "Volver a Instagram", "Voltar ao Instagram", "Zurück zu Instagram", "Retour sur Instagram",
   "Torna su Instagram", "Назад в Instagram", "Назад в Instagram")
_L("This payment link is invalid. Open a fresh one from the bot.", "Este enlace de pago no es válido. Abre uno nuevo desde el bot.",
   "Este link de pagamento é inválido. Abra um novo pelo bot.", "Dieser Zahlungslink ist ungültig. Öffne einen neuen im Bot.",
   "Ce lien de paiement n'est pas valide. Ouvre-en un nouveau depuis le bot.", "Questo link di pagamento non è valido. Aprine uno nuovo dal bot.",
   "Це посилання для оплати недійсне. Відкрий нове в боті.", "Эта ссылка на оплату недействительна. Открой новую в боте.")
_L("Payments are temporarily unavailable. Please try again in a few minutes.",
   "Los pagos no están disponibles por ahora. Inténtalo de nuevo en unos minutos.",
   "Pagamentos indisponíveis no momento. Tente de novo em alguns minutos.",
   "Zahlungen sind gerade nicht verfügbar. Bitte versuch es in ein paar Minuten nochmal.",
   "Les paiements sont momentanément indisponibles. Réessaie dans quelques minutes.",
   "I pagamenti non sono disponibili al momento. Riprova tra qualche minuto.",
   "Оплата тимчасово недоступна. Спробуй за кілька хвилин.", "Оплата временно недоступна. Попробуй через несколько минут.")

_L("📚 Saved to My lists", "📚 Guardado en Mis listas", "📚 Salvo em Minhas listas", "📚 In Meine Listen gespeichert",
   "📚 Enregistré dans Mes listes", "📚 Salvato nelle mie liste", "📚 Збережено в «Мої списки»",
   "📚 Сохранено в «Мои списки»")
_L("🔍 Search by photo", "🔍 Buscar por foto", "🔍 Buscar por foto", "🔍 Fotosuche", "🔍 Recherche photo",
   "🔍 Cerca per foto", "🔍 Пошук за фото", "🔍 Поиск по фото")

# Typed commands in every language (Quick Replies aren't shown on Instagram desktop)
LOCAL_TEXT_COMMANDS = {
    "canción": "FIND_MUSIC", "cancion": "FIND_MUSIC", "música": "FIND_MUSIC", "musica": "FIND_MUSIC",
    "canzone": "FIND_MUSIC", "chanson": "FIND_MUSIC", "musique": "FIND_MUSIC", "lied": "FIND_MUSIC",
    "пісня": "FIND_MUSIC", "песня": "FIND_MUSIC", "музыка": "FIND_MUSIC", "музика": "FIND_MUSIC",
    "película": "FIND_MOVIE", "pelicula": "FIND_MOVIE", "filme": "FIND_MOVIE", "série": "FIND_MOVIE",
    "serie": "FIND_MOVIE", "фільм": "FIND_MOVIE", "фильм": "FIND_MOVIE", "серіал": "FIND_MOVIE",
    "сериал": "FIND_MOVIE",
    "ropa": "FIND_ITEM", "roupa": "FIND_ITEM", "tenue": "FIND_ITEM", "vêtements": "FIND_ITEM",
    "kleidung": "FIND_ITEM", "vestiti": "FIND_ITEM", "одяг": "FIND_ITEM", "одежда": "FIND_ITEM",
    "речі": "FIND_ITEM", "вещи": "FIND_ITEM",
    "todo": "FIND_ALL", "tudo": "FIND_ALL", "alles": "FIND_ALL", "tout": "FIND_ALL", "tutto": "FIND_ALL",
    "все": "FIND_ALL", "всё": "FIND_ALL",
}
LOCAL_BALANCE_COMMANDS = {"saldo", "créditos", "creditos", "guthaben", "solde", "crédits", "crediti",
                          "баланс", "кредити", "кредиты"}


QUICK_REPLIES = [
    {"content_type": "text", "title": "🎵 Find Song", "payload": FIND_MUSIC},
    {"content_type": "text", "title": "🎬 Find Movie", "payload": FIND_MOVIE},
    {"content_type": "text", "title": "👕 Find Outfit/Item", "payload": FIND_ITEM},
    {"content_type": "text", "title": "📍 Find Place", "payload": FIND_PLACE},
    {"content_type": "text", "title": "🔥 Find Everything", "payload": FIND_ALL},
    {"content_type": "text", "title": "📚 My lists", "payload": MY_LISTS},
]

# Typed fallbacks (Quick Replies are not rendered on Instagram desktop).
TEXT_COMMANDS = {
    "song": FIND_MUSIC, "music": FIND_MUSIC, "find song": FIND_MUSIC, "track": FIND_MUSIC,
    "movie": FIND_MOVIE, "film": FIND_MOVIE, "show": FIND_MOVIE, "series": FIND_MOVIE,
    "find movie": FIND_MOVIE,
    "outfit": FIND_ITEM, "item": FIND_ITEM, "items": FIND_ITEM, "clothes": FIND_ITEM,
    "find outfit": FIND_ITEM, "find item": FIND_ITEM,
    "all": FIND_ALL, "everything": FIND_ALL, "find everything": FIND_ALL,
    "place": FIND_PLACE, "location": FIND_PLACE, "where": FIND_PLACE, "find place": FIND_PLACE,
    "lugar": FIND_PLACE, "local": FIND_PLACE, "ort": FIND_PLACE, "lieu": FIND_PLACE, "luogo": FIND_PLACE,
    "місце": FIND_PLACE, "место": FIND_PLACE, "где": FIND_PLACE, "де": FIND_PLACE,
}
# @mention in a comment → what to look for, and in which language to answer
MENTION_WORDS: dict[str, tuple[str, str]] = {}
for _lang, _action, _words in (
    ("en", FIND_MUSIC, "song music track sound audio beat 🎵 🎶"),
    ("en", FIND_MOVIE, "movie film show series 🎬"),
    ("en", FIND_ITEM, "outfit fit wearing wear shoes sneakers jacket hoodie dress bag glasses 👕 👟"),
    ("en", FIND_PLACE, "where place location spot 📍"),
    ("es", FIND_MUSIC, "canción cancion música musica"), ("es", FIND_MOVIE, "película pelicula serie"),
    ("es", FIND_ITEM, "ropa zapatillas chaqueta"), ("es", FIND_PLACE, "dónde donde lugar"),
    ("pt", FIND_MUSIC, "música som"), ("pt", FIND_MOVIE, "filme"), ("pt", FIND_ITEM, "roupa tênis"),
    ("pt", FIND_PLACE, "onde"),
    ("de", FIND_MUSIC, "lied welcher"), ("de", FIND_ITEM, "kleidung jacke schuhe"), ("de", FIND_PLACE, "wo ort"),
    ("fr", FIND_MUSIC, "chanson musique"), ("fr", FIND_ITEM, "tenue vêtements"), ("fr", FIND_PLACE, "où lieu"),
    ("it", FIND_MUSIC, "canzone"), ("it", FIND_ITEM, "vestiti scarpe"), ("it", FIND_PLACE, "dove luogo"),
    ("uk", FIND_MUSIC, "пісня пісню музика"), ("uk", FIND_MOVIE, "фільм серіал"), ("uk", FIND_ITEM, "одяг кросівки"),
    ("uk", FIND_PLACE, "де місце"),
    ("ru", FIND_MUSIC, "песня песню музыка трек"), ("ru", FIND_MOVIE, "фильм сериал"),
    ("ru", FIND_ITEM, "одежда кроссовки куртка"), ("ru", FIND_PLACE, "где место"),
):
    for _w in _words.split():
        MENTION_WORDS.setdefault(_w, (_action, _lang))


def mention_intent(text: str) -> tuple[str, str]:
    """('FIND_*', lang) from a comment like '@bot what song is this?'. Default: song, English."""
    words = re.findall(r"[\w']+|[\U0001F300-\U0001FAFF]", text.lower())
    for w in words:
        if w in MENTION_WORDS:
            return MENTION_WORDS[w]
    return FIND_MUSIC, "en"


LIST_COMMANDS = {"list", "lists", "my list", "my lists", "collection", "my finds", "history", "listas",
                 "mis listas", "minhas listas", "listen", "meine listen", "liste", "listes", "mes listes",
                 "le mie liste", "список", "списки", "мои списки", "мої списки", "колекція", "коллекция"}
INVITE_COMMANDS = {"invite", "invite friends", "referral", "friends", "refer", "invitar", "convidar",
                   "einladen", "inviter", "invita", "invitare", "пригласить", "запросити", "друзья", "друзі"}
BALANCE_COMMANDS = {"balance", "credits", "status", "my credits"}
BUY_COMMANDS = {"buy", "upgrade", "pricing", "premium", "subscribe", "pay", "plans", "comprar", "precios",
                "kaufen", "acheter", "acquista", "купить", "купити", "тарифы", "тарифи", "оплата"}
MANAGE_COMMANDS = {"manage", "subscription", "my subscription", "cancel", "cancel subscription", "unsubscribe",
                   "подписка", "отменить подписку", "підписка", "suscripción", "assinatura", "abo", "abonnement"}
TEXT_COMMANDS.update(LOCAL_TEXT_COMMANDS)
BALANCE_COMMANDS |= LOCAL_BALANCE_COMMANDS
PROMO_RE = re.compile(r"[A-Z0-9][A-Z0-9_-]{2,23}")
DELETE_COMMANDS = {"delete my data", "delete data", "delete my account", "remove my data"}
MSG_DATA_DELETED = (
    "Done ✅ Your data has been deleted: your ID, balance, subscription and last media link.\n"
    "If you message me again, a new empty profile will be created (free searches are not renewed)."
)

MSG_OVERLOAD = "Temporary overload, your credit was not deducted. Please try again in a moment."
MSG_MENU = "What should I find? 👇"
MSG_NO_MEDIA = "Send me a Reel, video or photo first 🙂\nTap ✈️ under any Reel → send it to me."
MSG_LINK_UNSUPPORTED = (
    "I can't open TikTok / YouTube links yet 🙈\n"
    "Send me an Instagram Reel (tap ✈️ under it → send to me) or the video file itself."
)
MSG_REEL_BLOCKED = (
    "Instagram didn't let me open this Reel right now 😕 (it may be private or restricted).\n"
    "Quick fix: screen-record it with sound (or save the video) and send me the file. "
    "For movie/outfit search a screenshot works too.\nNo credit was used."
)
MSG_MEDIA_RESTRICTED = (
    "I couldn't access this Reel 😕 It may be from a private account or restricted by the author.\n"
    "Try another Reel, or send the video file / a screenshot."
)
MSG_EXPIRED = (
    "This media is no longer available (Instagram links expire after a while). "
    "Please share the Reel again 🙏 No credit was used."
)
MSG_DOWNLOAD_FAILED = "I couldn't download this media 😕 Please share it again. No credit was used."
MSG_TOO_BIG = "This file is too large for me 😅 Please send a shorter clip. No credit was used."
MSG_NOT_MEDIA = "That link doesn't point to a video or image I can read 🤔 Please share the Reel itself."
MSG_UNSUPPORTED = "I couldn't read this file format 😕 Please send a Reel, an MP4 video or a JPG/PNG photo."
MSG_NO_AUDIO = "This one has no sound, so there's no song to find 🔇 No credit was used."
MSG_PHOTO_NO_AUDIO = "That's a photo — there's no sound to recognise 🙂 Send a Reel or video for song search. No credit was used."
MSG_NO_VIDEO = "That's an audio clip — I can only find songs in it 🎵 Tap “Find Song”."
MSG_NOT_FOUND = {
    FIND_MUSIC: "😕 I couldn't recognise the song in this clip — it may be an original sound, a voice-over or a heavy remix.\nNo credit was used.",
    FIND_MOVIE: "😕 I couldn't confidently identify a movie or TV show here — it may be original content.\nNo credit was used.",
    FIND_ITEM: "😕 I couldn't spot any identifiable clothes or items here.\nNo credit was used.",
    FIND_ALL: "😕 I couldn't find a song, a movie/show, identifiable items or a place in this one.\nNo credits were used.",
    FIND_PLACE: "😕 I couldn't recognise where this was filmed — it may be an ordinary street or an indoor place.\nNo credit was used.",
}
MSG_PAYMENTS_SOON = (
    "You've used all your free searches 🔋\n"
    "Paid plans are coming very soon — stay tuned, I'll be ready for your next Reel!"
)
MSG_PAYMENT_CREDITS = "✅ Payment received — thank you! +{n} credits added.\nYour balance: {balance} credits.\nSend me any Reel to continue 🎬"
MSG_PAYMENT_SUB = "✅ Unlimited is active until {date}! 🔥\nSend me any Reel — every search is on us now."


# ---- translations of the messages above (order: es, pt, de, fr, it, uk, ru)
_L(MSG_MENU, "¿Qué busco? 👇", "O que eu procuro? 👇", "Was soll ich finden? 👇", "Qu'est-ce que je cherche ? 👇",
   "Cosa cerco? 👇", "Що шукаємо? 👇", "Что ищем? 👇")
_L("Find something else? 👇", "¿Busco algo más? 👇", "Procuro mais alguma coisa? 👇", "Noch etwas finden? 👇",
   "Je cherche autre chose ? 👇", "Cerco altro? 👇", "Шукаємо ще щось? 👇", "Ищем что-то ещё? 👇")
_L(MSG_NO_MEDIA,
   "Primero mándame un Reel, video o foto 🙂\nToca ✈️ debajo de cualquier Reel → envíamelo.",
   "Primeiro me mande um Reel, vídeo ou foto 🙂\nToque em ✈️ embaixo de qualquer Reel → envie pra mim.",
   "Schick mir zuerst ein Reel, Video oder Foto 🙂\nTippe unter einem Reel auf ✈️ → schick es mir.",
   "Envoie-moi d'abord un Reel, une vidéo ou une photo 🙂\nAppuie sur ✈️ sous un Reel → envoie-le-moi.",
   "Prima mandami un Reel, un video o una foto 🙂\nTocca ✈️ sotto un Reel → inviamelo.",
   "Спершу надішли мені Reels, відео або фото 🙂\nНатисни ✈️ під будь-яким Reels → надішли мені.",
   "Сначала пришли мне Reels, видео или фото 🙂\nНажми ✈️ под любым Reels → отправь мне.")
_L(MSG_LINK_UNSUPPORTED,
   "Todavía no puedo abrir enlaces de TikTok / YouTube 🙈\nMándame un Reel de Instagram (toca ✈️ debajo → envíamelo) o el archivo de video.",
   "Ainda não consigo abrir links do TikTok / YouTube 🙈\nMande um Reel do Instagram (toque em ✈️ embaixo → envie pra mim) ou o próprio arquivo de vídeo.",
   "TikTok- / YouTube-Links kann ich noch nicht öffnen 🙈\nSchick mir ein Instagram-Reel (✈️ darunter → an mich senden) oder die Videodatei.",
   "Je ne peux pas encore ouvrir les liens TikTok / YouTube 🙈\nEnvoie-moi un Reel Instagram (✈️ dessous → envoie-le-moi) ou le fichier vidéo.",
   "Non posso ancora aprire link di TikTok / YouTube 🙈\nMandami un Reel di Instagram (tocca ✈️ sotto → inviamelo) o il file video.",
   "Посилання TikTok / YouTube я поки не відкриваю 🙈\nНадішли Reels з Instagram (✈️ під ним → мені) або сам відеофайл.",
   "Ссылки TikTok / YouTube я пока не открываю 🙈\nПришли Reels из Instagram (✈️ под ним → мне) или сам видеофайл.")
_L(MSG_REEL_BLOCKED,
   "Instagram no me dejó abrir este Reel ahora 😕 (puede ser privado o restringido).\nSolución rápida: graba la pantalla con sonido (o guarda el video) y mándame el archivo. Para película/ropa también sirve una captura.\nNo se gastó ningún crédito.",
   "O Instagram não me deixou abrir este Reel agora 😕 (pode ser privado ou restrito).\nDica: grave a tela com som (ou salve o vídeo) e me mande o arquivo. Para filme/roupa, um print também serve.\nNenhum crédito foi usado.",
   "Instagram lässt mich dieses Reel gerade nicht öffnen 😕 (evtl. privat oder eingeschränkt).\nSchnelle Lösung: Bildschirm mit Ton aufnehmen (oder Video speichern) und mir die Datei schicken. Für Film/Outfit reicht auch ein Screenshot.\nEs wurde kein Credit verbraucht.",
   "Instagram ne m'a pas laissé ouvrir ce Reel pour l'instant 😕 (il est peut-être privé ou restreint).\nAstuce : fais un enregistrement d'écran avec le son (ou enregistre la vidéo) et envoie-moi le fichier. Pour film/tenue, une capture d'écran suffit.\nAucun crédit utilisé.",
   "Instagram non mi ha fatto aprire questo Reel ora 😕 (potrebbe essere privato o limitato).\nSoluzione veloce: registra lo schermo con l'audio (o salva il video) e mandami il file. Per film/outfit va bene anche uno screenshot.\nNessun credito usato.",
   "Instagram зараз не дав відкрити цей Reels 😕 (можливо, він приватний або обмежений).\nШвидке рішення: запиши екран зі звуком (або збережи відео) і надішли мені файл. Для пошуку фільму/одягу підійде і скриншот.\nКредит не списано.",
   "Instagram сейчас не дал открыть этот Reels 😕 (возможно, он закрытый или с ограничениями).\nБыстрое решение: запиши экран со звуком (или сохрани видео) и пришли мне файл. Для поиска фильма/одежды подойдёт и скриншот.\nКредит не списан.")
_L(MSG_MEDIA_RESTRICTED,
   "No pude acceder a este Reel 😕 Puede ser de una cuenta privada o estar restringido por el autor.\nPrueba otro Reel o mándame el archivo de video / una captura.",
   "Não consegui acessar este Reel 😕 Pode ser de uma conta privada ou restrito pelo autor.\nTente outro Reel ou mande o arquivo de vídeo / um print.",
   "Ich komme an dieses Reel nicht ran 😕 Es ist evtl. von einem privaten Konto oder vom Autor eingeschränkt.\nProbier ein anderes Reel oder schick die Videodatei / einen Screenshot.",
   "Je n'ai pas pu accéder à ce Reel 😕 Il vient peut-être d'un compte privé ou est restreint par l'auteur.\nEssaie un autre Reel, ou envoie le fichier vidéo / une capture d'écran.",
   "Non riesco ad accedere a questo Reel 😕 Potrebbe essere di un account privato o limitato dall'autore.\nProva un altro Reel o mandami il file video / uno screenshot.",
   "Не вдалося відкрити цей Reels 😕 Можливо, він із приватного акаунта або автор обмежив доступ.\nСпробуй інший Reels або надішли відеофайл / скриншот.",
   "Не получилось открыть этот Reels 😕 Возможно, он из закрытого аккаунта или автор ограничил доступ.\nПопробуй другой Reels или пришли видеофайл / скриншот.")
_L(MSG_EXPIRED,
   "Este archivo ya no está disponible (los enlaces de Instagram caducan). Comparte el Reel de nuevo 🙏 No se gastó ningún crédito.",
   "Esta mídia não está mais disponível (os links do Instagram expiram). Compartilhe o Reel de novo 🙏 Nenhum crédito foi usado.",
   "Dieses Medium ist nicht mehr verfügbar (Instagram-Links laufen ab). Teile das Reel bitte nochmal 🙏 Es wurde kein Credit verbraucht.",
   "Ce média n'est plus disponible (les liens Instagram expirent). Partage le Reel à nouveau 🙏 Aucun crédit utilisé.",
   "Questo contenuto non è più disponibile (i link di Instagram scadono). Condividi di nuovo il Reel 🙏 Nessun credito usato.",
   "Цей файл уже недоступний (посилання Instagram з часом зникають). Надішли Reels ще раз 🙏 Кредит не списано.",
   "Этот файл уже недоступен (ссылки Instagram со временем истекают). Пришли Reels ещё раз 🙏 Кредит не списан.")
_L(MSG_DOWNLOAD_FAILED,
   "No pude descargar este archivo 😕 Compártelo de nuevo. No se gastó ningún crédito.",
   "Não consegui baixar esta mídia 😕 Compartilhe de novo. Nenhum crédito foi usado.",
   "Ich konnte das nicht herunterladen 😕 Teile es bitte nochmal. Es wurde kein Credit verbraucht.",
   "Je n'ai pas pu télécharger ce média 😕 Partage-le à nouveau. Aucun crédit utilisé.",
   "Non sono riuscito a scaricarlo 😕 Condividilo di nuovo. Nessun credito usato.",
   "Не вдалося завантажити 😕 Надішли ще раз. Кредит не списано.",
   "Не получилось скачать 😕 Пришли ещё раз. Кредит не списан.")
_L(MSG_TOO_BIG,
   "Este archivo es demasiado grande 😅 Mándame un clip más corto. No se gastó ningún crédito.",
   "Este arquivo é grande demais 😅 Mande um clipe mais curto. Nenhum crédito foi usado.",
   "Die Datei ist zu groß für mich 😅 Schick einen kürzeren Clip. Es wurde kein Credit verbraucht.",
   "Ce fichier est trop lourd pour moi 😅 Envoie un clip plus court. Aucun crédit utilisé.",
   "Questo file è troppo grande 😅 Mandami una clip più corta. Nessun credito usato.",
   "Файл завеликий 😅 Надішли коротший кліп. Кредит не списано.",
   "Файл слишком большой 😅 Пришли клип покороче. Кредит не списан.")
_L(MSG_NOT_MEDIA,
   "Ese enlace no lleva a un video o imagen que pueda leer 🤔 Comparte el Reel directamente.",
   "Esse link não leva a um vídeo ou imagem que eu consiga ler 🤔 Compartilhe o próprio Reel.",
   "Der Link führt nicht zu einem Video oder Bild, das ich lesen kann 🤔 Teile bitte das Reel selbst.",
   "Ce lien ne mène pas à une vidéo ou une image que je peux lire 🤔 Partage directement le Reel.",
   "Quel link non porta a un video o a un'immagine che posso leggere 🤔 Condividi direttamente il Reel.",
   "Це посилання не веде до відео чи фото, яке я можу прочитати 🤔 Надішли сам Reels.",
   "Эта ссылка не ведёт на видео или фото, которое я могу прочитать 🤔 Пришли сам Reels.")
_L(MSG_UNSUPPORTED,
   "No pude leer este formato 😕 Mándame un Reel, un video MP4 o una foto JPG/PNG.",
   "Não consegui ler este formato 😕 Mande um Reel, um vídeo MP4 ou uma foto JPG/PNG.",
   "Dieses Format kann ich nicht lesen 😕 Schick ein Reel, ein MP4-Video oder ein JPG/PNG-Foto.",
   "Je ne peux pas lire ce format 😕 Envoie un Reel, une vidéo MP4 ou une photo JPG/PNG.",
   "Non riesco a leggere questo formato 😕 Mandami un Reel, un video MP4 o una foto JPG/PNG.",
   "Не можу прочитати цей формат 😕 Надішли Reels, відео MP4 або фото JPG/PNG.",
   "Не могу прочитать этот формат 😕 Пришли Reels, видео MP4 или фото JPG/PNG.")
_L(MSG_NO_AUDIO,
   "Este no tiene sonido, así que no hay canción que buscar 🔇 No se gastó ningún crédito.",
   "Este não tem som, então não há música para achar 🔇 Nenhum crédito foi usado.",
   "Das hier hat keinen Ton – also kein Song zum Finden 🔇 Es wurde kein Credit verbraucht.",
   "Celui-ci n'a pas de son, donc pas de chanson à trouver 🔇 Aucun crédit utilisé.",
   "Questo non ha audio, quindi nessuna canzone da trovare 🔇 Nessun credito usato.",
   "Тут немає звуку, тож пісню не знайти 🔇 Кредит не списано.",
   "Тут нет звука, так что песню не найти 🔇 Кредит не списан.")
_L(MSG_PHOTO_NO_AUDIO,
   "Es una foto: no hay sonido que reconocer 🙂 Mándame un Reel o video para buscar canciones. No se gastó ningún crédito.",
   "É uma foto — não tem som para reconhecer 🙂 Mande um Reel ou vídeo para buscar músicas. Nenhum crédito foi usado.",
   "Das ist ein Foto – da gibt's keinen Ton 🙂 Für die Songsuche schick ein Reel oder Video. Es wurde kein Credit verbraucht.",
   "C'est une photo, il n'y a pas de son à reconnaître 🙂 Envoie un Reel ou une vidéo pour chercher une chanson. Aucun crédit utilisé.",
   "È una foto: non c'è audio da riconoscere 🙂 Mandami un Reel o un video per cercare canzoni. Nessun credito usato.",
   "Це фото — тут немає звуку 🙂 Для пошуку пісні надішли Reels або відео. Кредит не списано.",
   "Это фото — тут нет звука 🙂 Для поиска песни пришли Reels или видео. Кредит не списан.")
_L(MSG_NO_VIDEO,
   "Es un audio: solo puedo buscar canciones 🎵 Toca «Buscar canción».",
   "É um áudio — só consigo achar músicas nele 🎵 Toque em “Achar música”.",
   "Das ist ein Audio – darin kann ich nur Songs finden 🎵 Tippe auf „Song finden“.",
   "C'est un audio : je ne peux y trouver que des chansons 🎵 Appuie sur « La chanson ».",
   "È un audio: posso cercare solo canzoni 🎵 Tocca «Trova canzone».",
   "Це аудіо — тут я можу знайти лише пісню 🎵 Натисни «Знайти пісню».",
   "Это аудио — тут я могу найти только песню 🎵 Нажми «Найти песню».")
_L(MSG_NOT_FOUND[FIND_MUSIC],
   "😕 No pude reconocer la canción de este clip: puede ser un sonido original, una voz en off o un remix muy fuerte.\nNo se gastó ningún crédito.",
   "😕 Não reconheci a música deste clipe — pode ser um som original, uma narração ou um remix pesado.\nNenhum crédito foi usado.",
   "😕 Den Song in diesem Clip konnte ich nicht erkennen – evtl. Original-Audio, Voice-over oder ein heftiger Remix.\nEs wurde kein Credit verbraucht.",
   "😕 Je n'ai pas reconnu la chanson de ce clip : c'est peut-être un son original, une voix off ou un gros remix.\nAucun crédit utilisé.",
   "😕 Non ho riconosciuto la canzone in questa clip: potrebbe essere un audio originale, una voce fuori campo o un remix pesante.\nNessun credito usato.",
   "😕 Не вдалося розпізнати пісню — можливо, це оригінальний звук, озвучка або сильний ремікс.\nКредит не списано.",
   "😕 Не получилось распознать песню — возможно, это оригинальный звук, озвучка или сильный ремикс.\nКредит не списан.")
_L(MSG_NOT_FOUND[FIND_MOVIE],
   "😕 No pude identificar con seguridad una película o serie: puede ser contenido original.\nNo se gastó ningún crédito.",
   "😕 Não consegui identificar com certeza um filme ou série — pode ser conteúdo original.\nNenhum crédito foi usado.",
   "😕 Einen Film oder eine Serie konnte ich hier nicht sicher erkennen – evtl. eigener Content.\nEs wurde kein Credit verbraucht.",
   "😕 Je n'ai pas pu identifier avec certitude un film ou une série : c'est peut-être du contenu original.\nAucun crédit utilisé.",
   "😕 Non sono riuscito a identificare con certezza un film o una serie: potrebbe essere un contenuto originale.\nNessun credito usato.",
   "😕 Не вдалося впевнено визначити фільм чи серіал — можливо, це авторський контент.\nКредит не списано.",
   "😕 Не получилось уверенно определить фильм или сериал — возможно, это авторский контент.\nКредит не списан.")
_L(MSG_NOT_FOUND[FIND_ITEM],
   "😕 No encontré ropa ni objetos identificables aquí.\nNo se gastó ningún crédito.",
   "😕 Não encontrei roupas ou itens identificáveis aqui.\nNenhum crédito foi usado.",
   "😕 Hier konnte ich keine erkennbaren Klamotten oder Sachen finden.\nEs wurde kein Credit verbraucht.",
   "😕 Je n'ai repéré aucun vêtement ou objet identifiable ici.\nAucun crédit utilisé.",
   "😕 Non ho trovato vestiti o oggetti riconoscibili qui.\nNessun credito usato.",
   "😕 Тут не знайшлося одягу чи речей, які можна впізнати.\nКредит не списано.",
   "😕 Тут не нашлось одежды или вещей, которые можно опознать.\nКредит не списан.")
_L(MSG_NOT_FOUND[FIND_ALL],
   "😕 No encontré canción, película/serie, objetos ni lugar identificables en este.\nNo se gastaron créditos.",
   "😕 Não achei música, filme/série, itens nem lugar identificáveis neste.\nNenhum crédito foi usado.",
   "😕 Hier konnte ich weder Song, Film/Serie, erkennbare Sachen noch einen Ort finden.\nEs wurden keine Credits verbraucht.",
   "😕 Je n'ai trouvé ni chanson, ni film/série, ni objet, ni lieu identifiable ici.\nAucun crédit utilisé.",
   "😕 Non ho trovato né canzone, né film/serie, né oggetti, né luoghi riconoscibili.\nNessun credito usato.",
   "😕 Тут не знайшлося ні пісні, ні фільму/серіалу, ні речей, ні місця.\nКредити не списано.",
   "😕 Тут не нашлось ни песни, ни фильма/сериала, ни вещей, ни места.\nКредиты не списаны.")
_L(MSG_NOT_FOUND[FIND_PLACE],
   "😕 No pude reconocer dónde se grabó: puede ser una calle común o un interior.\nNo se gastó ningún crédito.",
   "😕 Não consegui reconhecer onde foi gravado — pode ser uma rua comum ou um lugar fechado.\nNenhum crédito foi usado.",
   "😕 Ich konnte nicht erkennen, wo das gedreht wurde – evtl. eine normale Straße oder ein Innenraum.\nEs wurde kein Credit verbraucht.",
   "😕 Je n'ai pas reconnu où ça a été filmé : peut-être une rue ordinaire ou un intérieur.\nAucun crédit utilisé.",
   "😕 Non ho riconosciuto dove è stato girato: potrebbe essere una strada qualunque o un interno.\nNessun credito usato.",
   "😕 Не вдалося впізнати, де це знято — можливо, звичайна вулиця або приміщення.\nКредит не списано.",
   "😕 Не получилось узнать, где это снято — возможно, обычная улица или помещение.\nКредит не списан.")
_L(MSG_OVERLOAD,
   "Sobrecarga temporal, no se descontó tu crédito. Inténtalo de nuevo en un momento.",
   "Sobrecarga temporária, seu crédito não foi descontado. Tente de novo daqui a pouco.",
   "Kurz überlastet, dein Credit wurde nicht abgezogen. Versuch es gleich nochmal.",
   "Surcharge temporaire, ton crédit n'a pas été débité. Réessaie dans un instant.",
   "Sovraccarico temporaneo, il credito non è stato scalato. Riprova tra poco.",
   "Тимчасове перевантаження, кредит не списано. Спробуй ще раз за мить.",
   "Временная перегрузка, кредит не списан. Попробуй ещё раз через минутку.")
_L(MSG_DATA_DELETED,
   "Listo ✅ Se eliminaron tus datos: tu ID, saldo, suscripción y el último enlace.\nSi me vuelves a escribir, se creará un perfil nuevo vacío (las búsquedas gratis no se renuevan).",
   "Pronto ✅ Seus dados foram apagados: seu ID, saldo, assinatura e o último link.\nSe você me escrever de novo, um perfil novo e vazio será criado (as buscas grátis não são renovadas).",
   "Erledigt ✅ Deine Daten wurden gelöscht: ID, Guthaben, Abo und letzter Medienlink.\nWenn du mir wieder schreibst, wird ein neues leeres Profil angelegt (Gratis-Suchen gibt's nicht erneut).",
   "C'est fait ✅ Tes données ont été supprimées : ton ID, ton solde, ton abonnement et le dernier lien.\nSi tu m'écris à nouveau, un nouveau profil vide sera créé (les recherches gratuites ne sont pas renouvelées).",
   "Fatto ✅ I tuoi dati sono stati eliminati: ID, saldo, abbonamento e ultimo link.\nSe mi scrivi di nuovo, verrà creato un nuovo profilo vuoto (le ricerche gratis non si rinnovano).",
   "Готово ✅ Твої дані видалено: ID, баланс, підписку та останнє посилання.\nЯкщо напишеш знову, створиться новий порожній профіль (безкоштовні пошуки не відновлюються).",
   "Готово ✅ Твои данные удалены: ID, баланс, подписка и последняя ссылка.\nЕсли напишешь снова, создастся новый пустой профиль (бесплатные поиски не возобновляются).")
_L(MSG_PAYMENTS_SOON,
   "Ya usaste todas tus búsquedas gratis 🔋\nLos planes de pago llegan muy pronto. ¡Atento, estaré listo para tu próximo Reel!",
   "Você usou todas as suas buscas grátis 🔋\nOs planos pagos chegam em breve — fique ligado, estarei pronto para o seu próximo Reel!",
   "Du hast alle Gratis-Suchen verbraucht 🔋\nBezahlpläne kommen sehr bald – bleib dran, ich bin bereit für dein nächstes Reel!",
   "Tu as utilisé toutes tes recherches gratuites 🔋\nLes offres payantes arrivent très bientôt — reste à l'écoute, je serai prêt pour ton prochain Reel !",
   "Hai usato tutte le ricerche gratuite 🔋\nI piani a pagamento arrivano prestissimo: resta sintonizzato, sarò pronto per il tuo prossimo Reel!",
   "Безкоштовні пошуки закінчилися 🔋\nПлатні тарифи вже зовсім скоро — стеж за новинами!",
   "Бесплатные поиски закончились 🔋\nПлатные тарифы уже совсем скоро — следи за новостями!")
_L(MSG_PAYMENT_CREDITS,
   "✅ Pago recibido, ¡gracias! +{n} créditos añadidos.\nTu saldo: {balance} créditos.\nMándame cualquier Reel para seguir 🎬",
   "✅ Pagamento recebido — obrigado! +{n} créditos adicionados.\nSeu saldo: {balance} créditos.\nMande qualquer Reel para continuar 🎬",
   "✅ Zahlung erhalten – danke! +{n} Credits hinzugefügt.\nDein Guthaben: {balance} Credits.\nSchick mir ein Reel, um weiterzumachen 🎬",
   "✅ Paiement reçu, merci ! +{n} crédits ajoutés.\nTon solde : {balance} crédits.\nEnvoie-moi un Reel pour continuer 🎬",
   "✅ Pagamento ricevuto, grazie! +{n} crediti aggiunti.\nSaldo: {balance} crediti.\nMandami un Reel per continuare 🎬",
   "✅ Оплату отримано — дякую! +{n} кредитів.\nБаланс: {balance} кредитів.\nНадсилай Reels 🎬",
   "✅ Оплата получена — спасибо! +{n} кредитов.\nБаланс: {balance} кредитов.\nПрисылай Reels 🎬")
_L(MSG_PAYMENT_SUB,
   "✅ ¡Ilimitado activo hasta el {date}! 🔥\nMándame cualquier Reel: todas las búsquedas van por mi cuenta.",
   "✅ Ilimitado ativo até {date}! 🔥\nMande qualquer Reel — agora todas as buscas são por minha conta.",
   "✅ Unlimited ist aktiv bis {date}! 🔥\nSchick mir beliebige Reels – jede Suche geht jetzt auf mich.",
   "✅ Illimité actif jusqu'au {date} ! 🔥\nEnvoie-moi n'importe quel Reel, toutes les recherches sont offertes.",
   "✅ Illimitato attivo fino al {date}! 🔥\nMandami qualsiasi Reel: ogni ricerca è inclusa.",
   "✅ Безліміт активний до {date}! 🔥\nНадсилай будь-які Reels — усі пошуки включено.",
   "✅ Безлимит активен до {date}! 🔥\nПрисылай любые Reels — все поиски включены.")


def welcome_text(bot_name: str, balance_line: str) -> str:
    return _t(
        "Hey there! 👋 I'm {bot_name}.\n\nSend me any Reel (tap ✈️ under the Reel → send it to me), a video or a photo, "
        "and in a few seconds I'll find:\n🎵 the song\n🎬 the movie or TV show\n👕 the outfit & items\n📍 the place\n\n{balance}",
        bot_name=bot_name, balance=balance_line)


def private_reply_text(free_credits: int) -> str:
    return (
        "Hey! 👋 Thanks for the comment!\n"
        "I find the song 🎵, the movie 🎬 or the outfit 👕 from any Reel in seconds.\n\n"
        "👉 Forward me any Reel right here in DM (tap ✈️ under it → send to me) and pick what to find. "
        f"Your first {free_credits} searches are free!"
    )


# Instagram post / Reel page links (Meta sends these instead of a video file for shared Reels).
IG_POST_RE = re.compile(
    r"^https?://(?:www\.|m\.)?instagram\.com/(?:[A-Za-z0-9_.]+/)?(reels?|p|tv)/([A-Za-z0-9_-]{5,})",
    re.IGNORECASE,
)
# Domains whose *page* links we cannot fetch.
UNSUPPORTED_LINK_DOMAINS = (
    "instagram.com", "instagr.am", "tiktok.com", "youtube.com", "youtu.be",
    "facebook.com", "fb.watch", "x.com", "twitter.com", "pinterest.",
)
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
FIND_KEYWORD_RE = re.compile(r"\bfind\b", re.IGNORECASE)

BUILD = "2026-11-28 photo-search"  # shown in the startup log to confirm which code is deployed
IG_TEXT_LIMIT = 950  # Instagram text messages must be < 1000 characters


# =============================================================================
# Errors
# =============================================================================


class ExternalServiceError(Exception):
    """An upstream AI/recognition API is overloaded or down. Never charge the user."""


class MediaUnavailable(Exception):
    """The media cannot be processed; carries a user-facing explanation."""

    def __init__(self, user_message: str, log_reason: str = ""):
        super().__init__(log_reason or user_message)
        self.user_message = user_message


class GraphAPIError(Exception):
    def __init__(self, status: int, code: int | None, subcode: int | None, message: str):
        super().__init__(f"Graph API error status={status} code={code} subcode={subcode}: {message}")
        self.status = status
        self.code = code
        self.subcode = subcode
        self.message = message

    @property
    def outside_messaging_window(self) -> bool:
        msg = (self.message or "").lower()
        return (
            self.subcode in (2018278, 2534022)
            or self.code == 10
            or "outside of allowed window" in msg
            or "outside the allowed window" in msg
        )


# =============================================================================
# Idempotency & per-user locking
# =============================================================================


class Deduplicator:
    """Remembers the last N processed webhook ids (message mid / comment id)."""

    def __init__(self, maxsize: int = 1000, ttl_sec: int = 6 * 3600):
        self._cache: TTLCache[str, bool] = TTLCache(maxsize=maxsize, ttl=ttl_sec)

    def seen(self, key: str) -> bool:
        """Returns True if the key was already processed, otherwise records it."""
        if key in self._cache:
            return True
        self._cache[key] = True
        return False


class UserLocks:
    """Serialises jobs of one user so double taps can't double-spend credits."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._waiters: dict[str, int] = {}

    @asynccontextmanager
    async def hold(self, key: str):
        lock = self._locks.setdefault(key, asyncio.Lock())
        self._waiters[key] = self._waiters.get(key, 0) + 1
        try:
            async with lock:
                yield
        finally:
            self._waiters[key] -= 1
            if self._waiters[key] <= 0:
                self._waiters.pop(key, None)
                self._locks.pop(key, None)


# =============================================================================
# Database (asyncpg)
# =============================================================================

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    igsid           TEXT PRIMARY KEY,
    credits         INT NOT NULL DEFAULT 3,
    sub_until       TIMESTAMPTZ NULL,
    last_media_url  TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_media_type TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS pending_notice  TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_media_mid  TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS lang            TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS ref_code        TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS users_ref_code_idx ON users (ref_code);
ALTER TABLE users ADD COLUMN IF NOT EXISTS pay_token       TEXT;      -- secret part of the user's payment links
CREATE UNIQUE INDEX IF NOT EXISTS users_pay_token_idx ON users (pay_token);
ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_customer TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS quota_day       DATE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS quota_used      INT NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS finds (            -- "My lists": what the bot found for each user
    id          BIGSERIAL PRIMARY KEY,
    igsid       TEXT NOT NULL,
    kind        TEXT NOT NULL,                -- song / movie / item / place
    title       TEXT NOT NULL,
    subtitle    TEXT NOT NULL DEFAULT '',
    url         TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS finds_user_kind_title_idx ON finds (igsid, kind, title);

CREATE TABLE IF NOT EXISTS referrals (
    referee_hash TEXT PRIMARY KEY,            -- hashed invited user: one invite bonus per person, ever
    referrer     TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE users ADD COLUMN IF NOT EXISTS updated_at      TIMESTAMPTZ NOT NULL DEFAULT now();

CREATE TABLE IF NOT EXISTS payment_events (
    event_key   TEXT PRIMARY KEY,
    igsid       TEXT,
    event_name  TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS deleted_users (
    id_hash     TEXT PRIMARY KEY,
    deleted_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS app_kv (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS promo_codes (
    code        TEXT PRIMARY KEY,
    credits     INT  NOT NULL,
    max_uses    INT  NOT NULL,
    used        INT  NOT NULL DEFAULT 0,
    expires_at  TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS promo_redemptions (
    code        TEXT NOT NULL,
    id_hash     TEXT NOT NULL,          -- hashed user id: survives "delete my data", no plain id kept
    redeemed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (code, id_hash)
);

-- Supabase exposes the public schema through its REST "Data API". Row Level Security with no
-- policies closes that door; the bot connects as the table owner and is not affected.
ALTER TABLE users          ENABLE ROW LEVEL SECURITY;
ALTER TABLE payment_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE deleted_users  ENABLE ROW LEVEL SECURITY;
ALTER TABLE app_kv         ENABLE ROW LEVEL SECURITY;
ALTER TABLE promo_codes    ENABLE ROW LEVEL SECURITY;
ALTER TABLE promo_redemptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE finds          ENABLE ROW LEVEL SECURITY;
ALTER TABLE referrals      ENABLE ROW LEVEL SECURITY;
"""


class Database:
    def __init__(self, dsn: str, free_credits: int):
        self._dsn = self._normalize_dsn(dsn)
        self._free_credits = free_credits
        self.pool: asyncpg.Pool | None = None

    @staticmethod
    def _normalize_dsn(dsn: str) -> str:
        """asyncpg doesn't understand libpq-only params that Neon/Supabase add to URLs."""
        parts = urlparse(dsn)
        if not parts.query:
            return dsn
        unsupported = {"channel_binding", "pgbouncer", "connection_limit", "pool_timeout"}
        query = "&".join(
            kv for kv in parts.query.split("&") if kv and kv.split("=", 1)[0].lower() not in unsupported
        )
        return parts._replace(query=query).geturl()

    async def connect(self) -> None:
        last_exc: Exception | None = None
        for attempt in range(1, 6):
            try:
                self.pool = await asyncpg.create_pool(
                    dsn=self._dsn,
                    min_size=1,
                    max_size=5,
                    command_timeout=10,
                    max_inactive_connection_lifetime=120,
                    # Required for Supabase/Neon PgBouncer poolers (transaction mode).
                    statement_cache_size=0,
                )
                async with self.pool.acquire() as conn:
                    await conn.execute(SCHEMA_SQL)
                log.info("Database connected, schema ensured")
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("DB connect attempt %s failed: %s", attempt, exc)
                await asyncio.sleep(min(2 * attempt, 8))
        raise RuntimeError(f"Could not connect to database: {last_exc}")

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    async def ping(self) -> bool:
        assert self.pool
        try:
            async with self.pool.acquire() as conn:
                return await conn.fetchval("SELECT 1") == 1
        except Exception:  # noqa: BLE001
            return False

    # --- users ---------------------------------------------------------------

    @staticmethod
    def _id_hash(igsid: str) -> str:
        return hashlib.sha256(f"reels-finder:{igsid}".encode()).hexdigest()

    async def _starting_credits(self, igsid: str) -> int:
        """Users who deleted their data don't get the free searches again."""
        assert self.pool
        deleted = await self.pool.fetchval(
            "SELECT 1 FROM deleted_users WHERE id_hash = $1", self._id_hash(igsid))
        return 0 if deleted else self._free_credits

    async def get_or_create_user(self, igsid: str) -> asyncpg.Record:
        assert self.pool
        row = await self.pool.fetchrow("SELECT * FROM users WHERE igsid = $1", igsid)
        if row:
            return row
        return await self.pool.fetchrow(
            """
            INSERT INTO users (igsid, credits) VALUES ($1, $2)
            ON CONFLICT (igsid) DO UPDATE SET igsid = EXCLUDED.igsid
            RETURNING *
            """,
            igsid,
            await self._starting_credits(igsid),
        )

    async def delete_user(self, igsid: str) -> None:
        """Deletes the user's profile; keeps only a one-way hash to block free-credit farming."""
        assert self.pool
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute("DELETE FROM users WHERE igsid = $1", igsid)
            await conn.execute("DELETE FROM finds WHERE igsid = $1", igsid)
            await conn.execute("DELETE FROM referrals WHERE referrer = $1", igsid)
            await conn.execute(
                "INSERT INTO deleted_users (id_hash) VALUES ($1) ON CONFLICT (id_hash) DO NOTHING",
                self._id_hash(igsid))

    async def set_last_media(self, igsid: str, url: str, media_type: str, mid: str = "") -> None:
        assert self.pool
        await self.pool.execute(
            """
            INSERT INTO users (igsid, credits, last_media_url, last_media_type, last_media_mid)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (igsid) DO UPDATE
               SET last_media_url = EXCLUDED.last_media_url,
                   last_media_type = EXCLUDED.last_media_type,
                   last_media_mid = EXCLUDED.last_media_mid,
                   updated_at = now()
            """,
            igsid,
            await self._starting_credits(igsid),
            url,
            media_type,
            mid or None,
        )

    async def create_promo(self, code: str, credits: int, max_uses: int, days: int | None) -> bool:
        assert self.pool
        res = await self.pool.execute(
            """INSERT INTO promo_codes (code, credits, max_uses, expires_at)
               VALUES ($1, $2::int, $3::int, CASE WHEN $4::int IS NULL THEN NULL
                                             ELSE now() + make_interval(days => $4::int) END)
               ON CONFLICT (code) DO NOTHING""", code, credits, max_uses, days)
        return res.endswith("1")

    async def list_promos(self) -> list[asyncpg.Record]:
        assert self.pool
        return await self.pool.fetch(
            """SELECT * FROM promo_codes WHERE used < max_uses AND (expires_at IS NULL OR expires_at > now())
               ORDER BY created_at DESC LIMIT 20""")

    async def disable_promo(self, code: str) -> bool:
        assert self.pool
        res = await self.pool.execute("UPDATE promo_codes SET max_uses = used WHERE code = $1", code)
        return res.endswith("1")

    async def redeem_promo(self, igsid: str, code: str) -> tuple[str, int, int]:
        """→ (status, credits_added, new_balance). status: ok | unknown | used | expired."""
        assert self.pool
        async with self.pool.acquire() as conn, conn.transaction():
            promo = await conn.fetchrow("SELECT * FROM promo_codes WHERE code = $1 FOR UPDATE", code)
            if not promo:
                return "unknown", 0, 0
            if promo["used"] >= promo["max_uses"] or (
                    promo["expires_at"] and promo["expires_at"] <= datetime.now(timezone.utc)):
                return "expired", 0, 0
            inserted = await conn.execute(
                "INSERT INTO promo_redemptions (code, id_hash) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                code, self._id_hash(igsid))
            if not inserted.endswith("1"):
                return "used", 0, 0
            await conn.execute("UPDATE promo_codes SET used = used + 1 WHERE code = $1", code)
            balance = await conn.fetchval(
                "UPDATE users SET credits = credits + $2::int, updated_at = now() WHERE igsid = $1 RETURNING credits",
                igsid, promo["credits"])
            return "ok", promo["credits"], balance or 0

    # ---- My lists
    async def save_finds(self, igsid: str, finds: list[tuple[str, str, str, str]], keep: int = 100) -> None:
        if not finds:
            return
        assert self.pool
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                """INSERT INTO finds (igsid, kind, title, subtitle, url) VALUES ($1, $2, $3, $4, $5)
                   ON CONFLICT (igsid, kind, title) DO UPDATE
                   SET subtitle = EXCLUDED.subtitle, url = EXCLUDED.url, created_at = now()""",
                [(igsid, k, t[:200], st[:200], u[:1000]) for k, t, st, u in finds])
            await conn.execute(  # keep only the newest `keep` per kind
                """DELETE FROM finds f USING (
                       SELECT id, row_number() OVER (PARTITION BY kind ORDER BY created_at DESC) AS rn
                       FROM finds WHERE igsid = $1) old
                   WHERE f.id = old.id AND old.rn > $2::int""", igsid, keep)

    async def count_finds(self, igsid: str) -> dict[str, int]:
        assert self.pool
        rows = await self.pool.fetch("SELECT kind, count(*) AS n FROM finds WHERE igsid = $1 GROUP BY kind", igsid)
        return {r["kind"]: r["n"] for r in rows}

    async def list_finds(self, igsid: str, kind: str, limit: int = 30) -> list[asyncpg.Record]:
        assert self.pool
        return await self.pool.fetch(
            "SELECT * FROM finds WHERE igsid = $1 AND kind = $2 ORDER BY created_at DESC LIMIT $3::int",
            igsid, kind, limit)

    # ---- Referrals
    async def ref_code(self, igsid: str) -> str:
        assert self.pool
        await self.get_or_create_user(igsid)
        code = await self.pool.fetchval("SELECT ref_code FROM users WHERE igsid = $1", igsid)
        while not code:
            candidate = "REF-" + secrets.token_hex(3).upper()
            try:
                code = await self.pool.fetchval(
                    "UPDATE users SET ref_code = $2 WHERE igsid = $1 AND ref_code IS NULL RETURNING ref_code",
                    igsid, candidate) or await self.pool.fetchval(
                    "SELECT ref_code FROM users WHERE igsid = $1", igsid)
            except asyncpg.UniqueViolationError:
                continue
        return code

    async def redeem_referral(self, igsid: str, code: str, credits: int, daily_cap: int,
                              new_user_days: int = 7) -> tuple[str, str, int, int]:
        """→ (status, referrer, referee_balance, referrer_balance).
        status: ok | unknown | self | used | old | cap."""
        assert self.pool
        async with self.pool.acquire() as conn, conn.transaction():
            referrer = await conn.fetchval("SELECT igsid FROM users WHERE ref_code = $1", code)
            if not referrer:
                return "unknown", "", 0, 0
            if referrer == igsid:
                return "self", referrer, 0, 0
            me = await conn.fetchrow("SELECT created_at FROM users WHERE igsid = $1", igsid)
            if me and me["created_at"] < datetime.now(timezone.utc) - timedelta(days=new_user_days):
                return "old", referrer, 0, 0
            today = await conn.fetchval(
                "SELECT count(*) FROM referrals WHERE referrer = $1 AND created_at > now() - interval '1 day'",
                referrer)
            if today >= daily_cap:
                return "cap", referrer, 0, 0
            inserted = await conn.execute(
                "INSERT INTO referrals (referee_hash, referrer) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                self._id_hash(igsid), referrer)
            if not inserted.endswith("1"):
                return "used", referrer, 0, 0
            add = "UPDATE users SET credits = credits + $2::int, updated_at = now() WHERE igsid = $1 RETURNING credits"
            mine = await conn.fetchval(add, igsid, credits)
            theirs = await conn.fetchval(add, referrer, credits)
            return "ok", referrer, mine or 0, theirs or 0

    async def get_lang(self, igsid: str) -> str | None:
        assert self.pool
        return await self.pool.fetchval("SELECT lang FROM users WHERE igsid = $1", igsid)

    async def set_lang(self, igsid: str, lang: str) -> None:
        assert self.pool
        await self.get_or_create_user(igsid)
        await self.pool.execute("UPDATE users SET lang = $2, updated_at = now() WHERE igsid = $1", igsid, lang)

    async def add_credits(self, igsid: str, amount: int) -> int:
        assert self.pool
        return await self.pool.fetchval(
            "UPDATE users SET credits = credits + $2::int, updated_at = now() WHERE igsid = $1 RETURNING credits",
            igsid, amount)

    async def charge(self, igsid: str, cost: int) -> int | None:
        """Atomically deducts credits. Returns the new balance or None if insufficient."""
        assert self.pool
        return await self.pool.fetchval(
            """
            UPDATE users SET credits = credits - $2, updated_at = now()
             WHERE igsid = $1 AND credits >= $2
         RETURNING credits
            """,
            igsid,
            cost,
        )

    async def set_pending_notice(self, igsid: str, text: str) -> None:
        assert self.pool
        await self.pool.execute(
            """
            UPDATE users
               SET pending_notice = COALESCE(pending_notice || E'\\n\\n', '') || $2,
                   updated_at = now()
             WHERE igsid = $1
            """,
            igsid,
            text,
        )

    async def pop_pending_notice(self, igsid: str) -> str | None:
        assert self.pool
        return await self.pool.fetchval(
            """
            UPDATE users u SET pending_notice = NULL
              FROM (SELECT igsid, pending_notice FROM users WHERE igsid = $1 FOR UPDATE) old
             WHERE u.igsid = old.igsid AND old.pending_notice IS NOT NULL
         RETURNING old.pending_notice
            """,
            igsid,
        )

    # --- payments --------------------------------------------------------------

    async def apply_credit_pack(self, event_key: str, igsid: str, event_name: str, credits: int) -> int | None:
        """Returns new balance, or None if this payment event was already processed."""
        assert self.pool
        async with self.pool.acquire() as conn, conn.transaction():
            claimed = await conn.fetchval(
                "INSERT INTO payment_events (event_key, igsid, event_name) VALUES ($1, $2, $3) "
                "ON CONFLICT (event_key) DO NOTHING RETURNING event_key",
                event_key, igsid, event_name,
            )
            if not claimed:
                return None
            return await conn.fetchval(
                """
                INSERT INTO users (igsid, credits) VALUES ($1, $2::int + $3::int)
                ON CONFLICT (igsid) DO UPDATE
                   SET credits = users.credits + $3::int, updated_at = now()
                RETURNING credits
                """,
                igsid, self._free_credits, credits,
            )

    # ---- Stripe
    async def pay_token(self, igsid: str) -> str:
        assert self.pool
        await self.get_or_create_user(igsid)
        token = await self.pool.fetchval("SELECT pay_token FROM users WHERE igsid = $1", igsid)
        while not token:
            try:
                token = await self.pool.fetchval(
                    "UPDATE users SET pay_token = $2 WHERE igsid = $1 AND pay_token IS NULL RETURNING pay_token",
                    igsid, secrets.token_urlsafe(18)) or await self.pool.fetchval(
                    "SELECT pay_token FROM users WHERE igsid = $1", igsid)
            except asyncpg.UniqueViolationError:
                continue
        return token

    async def user_by_pay_token(self, token: str) -> asyncpg.Record | None:
        assert self.pool
        if not token or len(token) > 64:
            return None
        return await self.pool.fetchrow("SELECT * FROM users WHERE pay_token = $1", token)

    async def set_stripe_customer(self, igsid: str, customer: str) -> None:
        assert self.pool
        await self.pool.execute("UPDATE users SET stripe_customer = $2 WHERE igsid = $1 AND $2 <> ''",
                                igsid, customer)

    async def set_subscription_until(self, event_key: str, igsid: str, event_name: str,
                                     until: datetime) -> datetime | None:
        """sub_until = max(current, until). None if this event was already processed."""
        assert self.pool
        async with self.pool.acquire() as conn, conn.transaction():
            claimed = await conn.fetchval(
                "INSERT INTO payment_events (event_key, igsid, event_name) VALUES ($1, $2, $3) "
                "ON CONFLICT (event_key) DO NOTHING RETURNING event_key", event_key, igsid, event_name)
            if not claimed:
                return None
            return await conn.fetchval(
                """INSERT INTO users (igsid, credits, sub_until) VALUES ($1, $2::int, $3)
                   ON CONFLICT (igsid) DO UPDATE
                   SET sub_until = GREATEST(COALESCE(users.sub_until, $3), $3), updated_at = now()
                   RETURNING sub_until""", igsid, self._free_credits, until)

    async def end_subscription(self, igsid: str, ended_at: datetime) -> None:
        assert self.pool
        await self.pool.execute("UPDATE users SET sub_until = LEAST(sub_until, $2), updated_at = now() "
                                "WHERE igsid = $1 AND sub_until IS NOT NULL", igsid, ended_at)

    async def remove_credits(self, event_key: str, igsid: str, event_name: str, credits: int) -> int | None:
        assert self.pool
        async with self.pool.acquire() as conn, conn.transaction():
            claimed = await conn.fetchval(
                "INSERT INTO payment_events (event_key, igsid, event_name) VALUES ($1, $2, $3) "
                "ON CONFLICT (event_key) DO NOTHING RETURNING event_key", event_key, igsid, event_name)
            if not claimed:
                return None
            return await conn.fetchval(
                "UPDATE users SET credits = GREATEST(credits - $2::int, 0), updated_at = now() "
                "WHERE igsid = $1 RETURNING credits", igsid, credits)

    async def sub_quota_used(self, igsid: str) -> int:
        assert self.pool
        return await self.pool.fetchval(
            "SELECT CASE WHEN quota_day = CURRENT_DATE THEN quota_used ELSE 0 END FROM users WHERE igsid = $1",
            igsid) or 0

    async def add_sub_quota(self, igsid: str, n: int) -> None:
        assert self.pool
        await self.pool.execute(
            """UPDATE users SET quota_used = CASE WHEN quota_day = CURRENT_DATE THEN quota_used + $2::int
                                                  ELSE $2::int END,
                                quota_day = CURRENT_DATE WHERE igsid = $1""", igsid, n)

    async def apply_subscription(self, event_key: str, igsid: str, event_name: str, days: int) -> datetime | None:
        """Extends sub_until by N days from max(now, current sub_until). None if duplicate."""
        assert self.pool
        async with self.pool.acquire() as conn, conn.transaction():
            claimed = await conn.fetchval(
                "INSERT INTO payment_events (event_key, igsid, event_name) VALUES ($1, $2, $3) "
                "ON CONFLICT (event_key) DO NOTHING RETURNING event_key",
                event_key, igsid, event_name,
            )
            if not claimed:
                return None
            return await conn.fetchval(
                """
                INSERT INTO users (igsid, credits, sub_until)
                VALUES ($1, $2::int, now() + make_interval(days => $3::int))
                ON CONFLICT (igsid) DO UPDATE
                   SET sub_until = GREATEST(COALESCE(users.sub_until, now()), now())
                                   + make_interval(days => $3::int),
                       updated_at = now()
                RETURNING sub_until
                """,
                igsid, self._free_credits, days,
            )

    # --- key/value (access-token storage) -------------------------------------

    async def kv_get(self, key: str) -> asyncpg.Record | None:
        assert self.pool
        return await self.pool.fetchrow("SELECT value, updated_at FROM app_kv WHERE key = $1", key)

    async def kv_set(self, key: str, value: str) -> None:
        assert self.pool
        await self.pool.execute(
            "INSERT INTO app_kv (key, value) VALUES ($1, $2) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
            key, value,
        )


# =============================================================================
# Instagram Messaging API client
# =============================================================================


def split_text(text: str, limit: int = IG_TEXT_LIMIT) -> list[str]:
    """Splits long text on paragraph/line boundaries to respect the IG 1000-char limit."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for para in text.split("\n"):
        candidate = f"{current}\n{para}" if current else para
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        while len(para) > limit:
            chunks.append(para[:limit])
            para = para[limit:]
        current = para
    if current:
        chunks.append(current)
    return chunks


def extract_graph_media(message: dict[str, Any]) -> list[str]:
    """Media URLs from a Graph API message object (attachments + shares)."""
    urls: list[str] = []
    for a in (message.get("attachments") or {}).get("data") or []:
        for key in ("video_data", "image_data"):
            u = _s((a.get(key) or {}).get("url"))
            if u:
                urls.append(u)
        if _s(a.get("file_url")):
            urls.append(_s(a.get("file_url")))
    for sh in (message.get("shares") or {}).get("data") or []:
        for key in ("url", "link"):
            if _s(sh.get(key)):
                urls.append(_s(sh.get(key)))
    seen: set[str] = set()
    return [u for u in urls if u.startswith("http") and not (u in seen or seen.add(u))]


def graph_message_to_event(message: dict[str, Any], own_id: str) -> dict[str, Any] | None:
    """Turns a Graph API message into the same shape as a messaging webhook event."""
    sender = _s((message.get("from") or {}).get("id"))
    mid = _s(message.get("id"))
    if not sender or not mid:
        return None
    attachments: list[dict[str, Any]] = []
    for a in (message.get("attachments") or {}).get("data") or []:
        if _s((a.get("video_data") or {}).get("url")):
            attachments.append({"type": "video", "payload": {"url": a["video_data"]["url"]}})
        elif _s((a.get("image_data") or {}).get("url")):
            attachments.append({"type": "image", "payload": {"url": a["image_data"]["url"]}})
        elif _s(a.get("file_url")):
            attachments.append({"type": "share", "payload": {"url": a["file_url"]}})
    for sh in (message.get("shares") or {}).get("data") or []:
        url = _s(sh.get("url")) or _s(sh.get("link"))
        kind = _s(sh.get("type")).lower()
        if url:
            attachments.append({"type": kind if kind in ("ig_reel", "reel", "ig_post") else "share",
                                "payload": {"url": url}})
    msg: dict[str, Any] = {"mid": mid}
    if _s(message.get("message")):
        msg["text"] = _s(message.get("message"))
    if attachments:
        msg["attachments"] = attachments
    if "text" not in msg and "attachments" not in msg:
        return None
    return {"sender": {"id": sender}, "recipient": {"id": own_id}, "message": msg}


def parse_graph_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z")
    except (TypeError, ValueError):
        return None


class InstagramClient:
    def __init__(self, http: httpx.AsyncClient, base: str, account_id: str, token: str):
        self._http = http
        self._base = base
        self._account_id = account_id
        self.token = token

    @property
    def _messages_url(self) -> str:
        return f"{self._base}/{self._account_id}/messages"

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in (1, 2):
            try:
                resp = await self._http.post(
                    self._messages_url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=httpx.Timeout(10.0, connect=3.0),
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt == 1:
                    await asyncio.sleep(0.3)
                    continue
                raise GraphAPIError(0, None, None, f"transport error: {exc}") from exc

            try:
                data = resp.json()
            except ValueError:
                data = {}
            if resp.status_code < 400 and "error" not in data:
                return data

            err = data.get("error") or {}
            gerr = GraphAPIError(
                resp.status_code,
                err.get("code"),
                err.get("error_subcode"),
                err.get("message") or resp.text[:300],
            )
            # Retry only transient server-side failures.
            if resp.status_code >= 500 and attempt == 1:
                last_exc = gerr
                await asyncio.sleep(0.3)
                continue
            raise gerr
        raise GraphAPIError(0, None, None, f"unreachable: {last_exc}")

    async def send_text(self, igsid: str, text: str, quick_replies: list[dict] | None = None) -> None:
        chunks = split_text(text)
        for i, chunk in enumerate(chunks):
            message: dict[str, Any] = {"text": chunk}
            if quick_replies and i == len(chunks) - 1:
                message["quick_replies"] = quick_replies
            await self._post({"recipient": {"id": igsid}, "messaging_type": "RESPONSE", "message": message})

    async def send_action(self, igsid: str, action: str) -> None:
        await self._post({"recipient": {"id": igsid}, "sender_action": action})

    async def send_generic_template(self, igsid: str, elements: list[dict]) -> None:
        await self._post(
            {
                "recipient": {"id": igsid},
                "messaging_type": "RESPONSE",
                "message": {
                    "attachment": {
                        "type": "template",
                        "payload": {"template_type": "generic", "elements": elements},
                    }
                },
            }
        )

    async def send_private_reply(self, comment_id: str, text: str) -> None:
        await self._post({"recipient": {"comment_id": comment_id}, "message": {"text": text}})

    async def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        resp = await self._http.get(
            f"{self._base}/{path}",
            params=params,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=httpx.Timeout(10.0, connect=3.0),
        )
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if resp.status_code >= 400 or "error" in data:
            err = data.get("error") or {}
            raise GraphAPIError(resp.status_code, err.get("code"), err.get("error_subcode"),
                                err.get("message") or resp.text[:300])
        return data

    async def get_message_media_urls(self, mid: str) -> list[str]:
        """Alternative media URLs for a message, read through the Graph API."""
        for fields in ("attachments{video_data,image_data,file_url},shares{link,url,type,name}",
                       "attachments,shares"):
            try:
                data = await self._get(mid, {"fields": fields})
                break
            except GraphAPIError as exc:
                log.warning("Message lookup with fields %s failed: %s", fields, exc)
        else:
            return []
        log.info("Message lookup keys: %s", sorted(data.keys()))
        return extract_graph_media(data)

    async def list_recent_messages(self) -> list[dict[str, Any]]:
        """Recent messages across conversations, including the Requests folder."""
        field_sets = (
            "updated_time,messages.limit(5){id,created_time,from,message,"
            "attachments{video_data,image_data,file_url},shares{link,url,type,name}}",
            "updated_time,messages.limit(5){id,created_time,from,message,attachments,shares}",
        )
        last_exc: GraphAPIError | None = None
        for fields in field_sets:
            try:
                data = await self._get(f"{self._account_id}/conversations",
                                       {"platform": "instagram", "fields": fields, "limit": "10"})
                break
            except GraphAPIError as exc:
                last_exc = exc
                if exc.code != 100:  # 100 = invalid parameter/field: try the simpler field set
                    raise
        else:
            raise last_exc or GraphAPIError(0, None, None, "conversations unavailable")
        out: list[dict[str, Any]] = []
        for conv in data.get("data") or []:
            out.extend((conv.get("messages") or {}).get("data") or [])
        return out

    async def get_mentioned_comment(self, comment_id: str) -> dict[str, Any]:
        fields = f"mentioned_comment.comment_id({comment_id}){{text,username,media{{id,permalink,media_type}}}}"
        return (await self._get(self._account_id, {"fields": fields})).get("mentioned_comment") or {}

    async def get_mentioned_media(self, media_id: str) -> dict[str, Any]:
        fields = f"mentioned_media.media_id({media_id}){{caption,username,permalink,media_type}}"
        return (await self._get(self._account_id, {"fields": fields})).get("mentioned_media") or {}

    async def reply_to_mention(self, media_id: str, comment_id: str, message: str) -> None:
        """Public reply under the comment (or the caption) where our account was @mentioned."""
        params = {"media_id": media_id, "message": message}
        if comment_id:
            params["comment_id"] = comment_id
        resp = await self._http.post(f"{self._base}/{self._account_id}/mentions", params=params,
                                     headers={"Authorization": f"Bearer {self.token}"},
                                     timeout=httpx.Timeout(10.0, connect=3.0))
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if resp.status_code >= 400 or "error" in data:
            err = data.get("error") or {}
            raise GraphAPIError(resp.status_code, err.get("code"), err.get("error_subcode"),
                                err.get("message") or resp.text[:300])

    async def subscribe_app(self, fields: str) -> dict[str, Any]:
        """Subscribes the Instagram account to this app's webhooks (needed for Unified login)."""
        resp = await self._http.post(
            f"{self._base}/{self._account_id}/subscribed_apps",
            params={"subscribed_fields": fields},
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=httpx.Timeout(10.0, connect=3.0),
        )
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if resp.status_code >= 400 or "error" in data:
            err = data.get("error") or {}
            raise GraphAPIError(resp.status_code, err.get("code"), err.get("error_subcode"),
                                err.get("message") or resp.text[:300])
        return data

    async def refresh_long_lived_token(self) -> str:
        """Instagram-Login long-lived tokens live 60 days and can be refreshed."""
        resp = await self._http.get(
            "https://graph.instagram.com/refresh_access_token",
            params={"grant_type": "ig_refresh_token", "access_token": self.token},
            timeout=httpx.Timeout(10.0, connect=3.0),
        )
        data = resp.json()
        if resp.status_code >= 400 or "access_token" not in data:
            raise GraphAPIError(resp.status_code, (data.get("error") or {}).get("code"), None, str(data)[:300])
        return data["access_token"]


# =============================================================================
# Media processing: download + ffmpeg/ffprobe
# =============================================================================


@dataclass
class MediaInfo:
    path: Path
    kind: str  # "video" | "image" | "audio"
    duration: float
    has_audio: bool
    has_video: bool


async def run_cmd(args: list[str], timeout: float) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, out, err
    except asyncio.TimeoutError as exc:
        raise MediaUnavailable(MSG_UNSUPPORTED, f"{args[0]} timed out") from exc
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()


IMAGE_FORMATS = {"image2", "png_pipe", "jpeg_pipe", "webp_pipe", "bmp_pipe", "tiff_pipe", "heif"}
IMAGE_CODECS = {"mjpeg", "png", "webp", "bmp", "tiff", "hevc_image"}


async def assert_public_url(url: str) -> None:
    """Blocks SSRF: the bot must never fetch localhost, private networks or cloud metadata IPs."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise MediaUnavailable(MSG_NOT_MEDIA, f"bad url scheme/host: {parts.scheme}")
    if parts.port not in (None, 80, 443):
        raise MediaUnavailable(MSG_NOT_MEDIA, f"non-standard port {parts.port}")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(parts.hostname, None)
    except OSError as exc:
        raise MediaUnavailable(MSG_DOWNLOAD_FAILED, f"dns error: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise MediaUnavailable(MSG_NOT_MEDIA, f"blocked non-public address {ip}")


REDIRECT_CODES = {301, 302, 303, 307, 308}

# Tried in order when a media URL answers with an HTML page instead of a file.
DOWNLOAD_USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36",
    "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)",
)


def instagram_post_url(url: str) -> str | None:
    """Canonical https://www.instagram.com/<reel|p|tv>/<code>/ for an Instagram post link, else None."""
    m = IG_POST_RE.match(url.strip())
    if not m:
        return None
    kind = "reel" if m.group(1).lower().startswith("reel") else m.group(1).lower()
    return f"https://www.instagram.com/{kind}/{m.group(2)}/"


@dataclass
class ResolvedPost:
    video_url: str = ""   # progressive mp4 (video + audio)
    dash_video: str = ""  # used only if no progressive format exists
    dash_audio: str = ""
    thumbnail: str = ""
    caption: str = ""
    music_title: str = ""    # Instagram's own audio label
    music_artist: str = ""
    music_original: bool = False  # True = "original audio" (user-uploaded), False = licensed track
    location: str = ""           # Instagram location tag, e.g. "Santorini, Greece"
    location_lat: float | None = None
    location_lng: float | None = None


def pick_post_formats(info: dict) -> ResolvedPost:
    """Chooses the lightest usable formats from yt-dlp's info dict (no network)."""
    if info.get("_type") == "playlist":  # carousel: first entry that has a video, else the first one
        entries = [e for e in (info.get("entries") or []) if e]
        with_video = [e for e in entries if e.get("formats") or e.get("url")]
        parent = info
        info = dict((with_video or entries or [{}])[0])
        for key in ("description", "ig_music", "ig_location"):
            info.setdefault(key, parent.get(key))
    music = info.get("ig_music") or {}
    out = ResolvedPost(thumbnail=_s(info.get("thumbnail")), caption=_s(info.get("description"))[:600],
                       music_title=_s(music.get("title")), music_artist=_s(music.get("artist")),
                       music_original=bool(music.get("original")))
    loc = info.get("ig_location") or {}
    out.location = _s(loc.get("name"))[:120]
    if isinstance(loc.get("lat"), (int, float)) and isinstance(loc.get("lng"), (int, float)):
        out.location_lat, out.location_lng = float(loc["lat"]), float(loc["lng"])
    formats = [f for f in (info.get("formats") or []) if _s(f.get("url")).startswith("http")]
    if not formats and _s(info.get("url")).startswith("http"):
        formats = [info]

    def has(f: dict, kind: str) -> bool:
        return _s(f.get(kind)) not in ("none",)  # missing codec info = unknown = assume present

    def height(f: dict) -> int:
        return int(f.get("height") or 0)

    def by_size(f: dict) -> tuple:  # prefer ~480-720p: enough for vision, small for speed
        h = height(f)
        return (0 if 400 <= h <= 1080 else 1, abs(h - 720) if h else 9999)

    progressive = [f for f in formats if has(f, "vcodec") and has(f, "acodec")
                   and "dash" not in _s(f.get("format_id")).lower()]
    if progressive:
        out.video_url = sorted(progressive, key=by_size)[0]["url"]
        return out
    videos = [f for f in formats if has(f, "vcodec") and _s(f.get("acodec")) == "none"]
    audios = [f for f in formats if has(f, "acodec") and _s(f.get("vcodec")) == "none"]
    if videos:
        out.dash_video = sorted(videos, key=by_size)[0]["url"]
    if audios:
        out.dash_audio = sorted(audios, key=lambda f: float(f.get("abr") or f.get("tbr") or 0))[-1]["url"]
    if not videos and not audios and formats:
        out.video_url = formats[-1]["url"]
    return out


def ig_music_from_product(product: dict) -> dict:
    """Instagram's audio label from the raw media JSON: licensed track or 'original audio'."""
    clips = (product or {}).get("clips_metadata") or {}
    asset = (((clips.get("music_info") or {}).get("music_asset_info"))
             or (((product or {}).get("music_metadata") or {}).get("music_info") or {}).get("music_asset_info")
             or {})
    if _s(asset.get("title")):
        return {"title": _s(asset.get("title")), "artist": _s(asset.get("display_artist")), "original": False}
    sound = clips.get("original_sound_info") or {}
    if _s(sound.get("original_audio_title")):
        return {"title": _s(sound.get("original_audio_title")),
                "artist": _s((sound.get("ig_artist") or {}).get("username")), "original": True}
    return {}


_YTDL_HOOKED = False


def _hook_ytdl_instagram() -> None:
    """yt-dlp drops Instagram's music label; this keeps it as info['ig_music']."""
    global _YTDL_HOOKED
    if _YTDL_HOOKED:
        return
    _YTDL_HOOKED = True
    try:
        from yt_dlp.extractor.instagram import InstagramBaseIE

        original = InstagramBaseIE._extract_product

        def _extract_product(self, product_info, *args, **kwargs):  # noqa: ANN001
            info = original(self, product_info, *args, **kwargs)
            try:
                raw = product_info[0] if isinstance(product_info, list) else product_info
                if isinstance(info, dict) and isinstance(raw, dict):
                    info["ig_music"] = ig_music_from_product(raw)
                    loc = raw.get("location") or {}
                    if isinstance(loc, dict) and _s(loc.get("name")):
                        name, city = _s(loc.get("name")), _s(loc.get("city"))
                        if city and city.lower() not in name.lower():
                            name = f"{name}, {city}"
                        info["ig_location"] = {"name": name, "lat": loc.get("lat"), "lng": loc.get("lng")}
            except Exception:  # noqa: BLE001
                pass
            return info

        InstagramBaseIE._extract_product = _extract_product
    except Exception as exc:  # noqa: BLE001 — never break downloads over metadata
        log.warning("Could not hook yt-dlp Instagram music metadata: %s", exc)


class _QuietYtdlLogger:
    def debug(self, msg: str) -> None: ...
    def info(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...
    def error(self, msg: str) -> None:
        log.debug("yt-dlp: %s", msg)


class ReelExtractor:
    """Resolves an Instagram Reel/post page into direct CDN media URLs with yt-dlp.
    Results are cached per post for 10 minutes, so a second button tap is instant."""

    def __init__(self, cookies: str = "", proxy: str = "", tmp_root: str = "/tmp"):
        self.proxy = proxy
        self.cookiefile = self._write_cookies(cookies, tmp_root) if cookies else ""
        self.cache: TTLCache = TTLCache(maxsize=500, ttl=600)
        self.inflight: dict[str, asyncio.Task] = {}

    @staticmethod
    def _write_cookies(raw: str, tmp_root: str) -> str:
        text = raw.strip()
        if "\t" not in text:
            try:
                import base64
                text = base64.b64decode(text).decode("utf-8")
            except Exception:  # noqa: BLE001
                log.warning("IG_COOKIES is neither a cookies.txt nor base64 of one — ignored")
                return ""
        text = text.lstrip("\ufeff").replace("\r\n", "\n")  # files saved on Windows
        if not text.startswith("# Netscape"):
            text = "# Netscape HTTP Cookie File\n" + text
        path = Path(tmp_root) / f"igcookies-{uuid.uuid4().hex}.txt"
        path.write_text(text + "\n")
        path.chmod(0o600)
        log.info("Instagram cookies loaded for the Reel downloader")
        return str(path)

    def _extract_sync(self, url: str) -> dict:
        import yt_dlp  # imported lazily: heavy module

        _hook_ytdl_instagram()

        opts: dict[str, Any] = {
            "quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": False,
            "socket_timeout": 10, "extractor_retries": 1, "logger": _QuietYtdlLogger(),
            "cachedir": False,
        }
        if self.cookiefile:
            opts["cookiefile"] = self.cookiefile
        if self.proxy:
            opts["proxy"] = self.proxy
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False) or {}

    @staticmethod
    def impersonation_available() -> bool:
        """True if curl_cffi is installed: yt-dlp can then pose as a real browser and use Instagram's
        main (GraphQL) method. Without it only the fallback web-page method runs — often blocked."""
        try:
            import yt_dlp
            from yt_dlp.networking.impersonate import ImpersonateTarget

            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                return bool(ydl._impersonate_target_available(ImpersonateTarget()))
        except Exception:  # noqa: BLE001
            return False

    async def _resolve_uncached(self, url: str) -> ResolvedPost:
        started = asyncio.get_running_loop().time()
        info: dict = {}
        for attempt in (1, 2, 3):
            try:
                info = await asyncio.wait_for(asyncio.to_thread(self._extract_sync, url), timeout=20)
                break
            except asyncio.TimeoutError as exc:
                raise MediaUnavailable(MSG_REEL_BLOCKED, "reel extractor timeout") from exc
            except Exception as exc:  # noqa: BLE001 — yt-dlp raises many types
                reason = re.sub(r"\s+", " ", str(exc))[:300]
                # Private posts won't change; a rate-limit / login wall won't lift in a few seconds either
                # (retrying only makes Instagram angrier) → fail fast, the user gets the screen-record tip.
                permanent = any(w in reason.lower() for w in ("private", "follow this account", "there is no video",
                                                              "not available", "rate-limit", "login page"))
                if attempt == 3 or permanent:
                    raise MediaUnavailable(MSG_REEL_BLOCKED, f"reel extractor failed: {reason}") from exc
                log.info("Reel extractor attempt %d failed (%s) → retrying", attempt, reason[:120])
                await asyncio.sleep(1.0 * attempt)
        post = pick_post_formats(info)
        log.info("Reel resolved in %.1fs: progressive=%s dash_video=%s dash_audio=%s thumb=%s "
                 "audio_label=%r (%s)",
                 asyncio.get_running_loop().time() - started, bool(post.video_url),
                 bool(post.dash_video), bool(post.dash_audio), bool(post.thumbnail),
                 f"{post.music_artist} — {post.music_title}"[:80] if post.music_title else "",
                 "original" if post.music_original else "licensed")
        return post

    async def resolve(self, url: str) -> ResolvedPost:
        if url in self.cache:
            return self.cache[url]
        task = self.inflight.get(url)
        if task is None:  # several taps / prefetch share one extraction
            task = asyncio.create_task(self._resolve_uncached(url))
            self.inflight[url] = task
            task.add_done_callback(lambda _t, u=url: self.inflight.pop(u, None))
        post = await asyncio.shield(task)
        self.cache[url] = post
        return post

    def cached(self, url: str) -> ResolvedPost | None:
        ig_url = instagram_post_url(url)
        return self.cache.get(ig_url) if ig_url else None

    def prefetch(self, url: str) -> None:
        """Starts resolving while the user is still choosing a button."""
        async def _run() -> None:
            try:
                await self.resolve(url)
            except Exception as exc:  # noqa: BLE001
                log.info("Reel prefetch failed: %s", exc)
        if url not in self.cache and url not in self.inflight:
            asyncio.create_task(_run())


OG_IMAGE_RE = re.compile(
    r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']'
    r'|<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
    re.IGNORECASE,
)


async def fetch_og_image(http: httpx.AsyncClient, page_url: str) -> str:
    """Cover image from the post page's Open Graph tags (what link previews use). '' if none."""
    try:
        async with http.stream("GET", page_url, headers={"User-Agent": DOWNLOAD_USER_AGENTS[-1]},
                               timeout=httpx.Timeout(8.0, connect=4.0), follow_redirects=False) as resp:
            if resp.status_code != 200:
                return ""
            body = b""
            async for chunk in resp.aiter_bytes(64 * 1024):
                body += chunk
                if len(body) > 512 * 1024 or b"og:image" in body and b"</head>" in body:
                    break
    except httpx.HTTPError:
        return ""
    m = OG_IMAGE_RE.search(body.decode("utf-8", "replace"))
    return html.unescape(m.group(1) or m.group(2)) if m else ""


def describe_url(url: str) -> str:
    """host + shortened path, without the query string (which carries CDN signatures)."""
    parts = urlparse(url)
    path = parts.path if len(parts.path) <= 60 else parts.path[:60] + "…"
    return f"{parts.hostname}{path}"


async def download_media(http: httpx.AsyncClient, url: str, dest: Path, max_bytes: int,
                         allow_private: bool = False) -> str:
    """Streams media to disk with a hard size cap. Returns the response content-type.
    Redirects are followed manually so every hop is checked against SSRF."""

    async def _attempt(user_agent: str) -> str | None:
        """Returns content-type on success, None if the server answered with an HTML page."""
        current = url
        for _hop in range(5):
            if not allow_private:
                await assert_public_url(current)
            async with http.stream(
                "GET",
                current,
                headers={"User-Agent": user_agent, "Accept": "video/*,image/*,audio/*,*/*;q=0.8"},
                timeout=httpx.Timeout(12.0, connect=4.0),
                follow_redirects=False,
            ) as resp:
                if resp.status_code in REDIRECT_CODES and resp.headers.get("location"):
                    current = urljoin(current, resp.headers["location"])
                    continue
                if resp.status_code in (401, 403, 404, 410):
                    raise MediaUnavailable(MSG_EXPIRED, f"media HTTP {resp.status_code} at {describe_url(current)}")
                if resp.status_code >= 400:
                    raise MediaUnavailable(MSG_DOWNLOAD_FAILED, f"media HTTP {resp.status_code} at {describe_url(current)}")
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype.startswith("text/") or ctype in ("application/json",):
                    head = b""
                    async for chunk in resp.aiter_bytes(2048):
                        head = chunk[:300]
                        break
                    snippet = re.sub(r"\s+", " ", head.decode("utf-8", "replace"))[:200]
                    log.warning("Media URL returned %s (UA=%s…) at %s; first bytes: %r",
                                ctype, user_agent[:20], describe_url(current), snippet)
                    return None
                declared = int(resp.headers.get("content-length") or 0)
                if declared and declared > max_bytes:
                    raise MediaUnavailable(MSG_TOO_BIG, f"content-length {declared}")
                size = 0
                with dest.open("wb") as fh:
                    async for chunk in resp.aiter_bytes(256 * 1024):
                        size += len(chunk)
                        if size > max_bytes:
                            raise MediaUnavailable(MSG_TOO_BIG, f"stream exceeded {max_bytes}")
                        fh.write(chunk)
                if size == 0:
                    raise MediaUnavailable(MSG_DOWNLOAD_FAILED, "empty body")
                return ctype
        raise MediaUnavailable(MSG_DOWNLOAD_FAILED, "too many redirects")

    async def _do() -> str:
        try:
            for user_agent in DOWNLOAD_USER_AGENTS:
                ctype = await _attempt(user_agent)
                if ctype is not None:
                    return ctype
            raise MediaUnavailable(MSG_NOT_MEDIA, f"html page instead of media at {describe_url(url)}")
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise MediaUnavailable(MSG_DOWNLOAD_FAILED, f"download error: {exc}") from exc

    try:
        return await asyncio.wait_for(_do(), timeout=25)
    except asyncio.TimeoutError as exc:
        raise MediaUnavailable(MSG_DOWNLOAD_FAILED, "download timeout") from exc


async def probe_media(path: Path, content_type: str) -> MediaInfo:
    rc, out, err = await run_cmd(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration,format_name:stream=codec_type,codec_name",
            "-of", "json", str(path),
        ],
        timeout=10,
    )
    if rc != 0:
        raise MediaUnavailable(MSG_UNSUPPORTED, f"ffprobe failed: {err[:200]!r}")
    data = json.loads(out or b"{}")
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    has_video = bool(video_streams)
    try:
        duration = float(fmt.get("duration"))
    except (TypeError, ValueError):
        duration = 0.0
    format_names = set((fmt.get("format_name") or "").split(","))

    is_image = content_type.startswith("image/") or bool(format_names & IMAGE_FORMATS) or (
        has_video and not has_audio and duration <= 0.1
        and any(s.get("codec_name") in IMAGE_CODECS for s in video_streams)
    )
    if is_image and has_video:
        kind = "image"
    elif has_video:
        kind = "video"
    elif has_audio:
        kind = "audio"
    else:
        raise MediaUnavailable(MSG_UNSUPPORTED, "no audio/video streams")
    return MediaInfo(path=path, kind=kind, duration=duration, has_audio=has_audio, has_video=has_video)


async def extract_audio_clip(info: MediaInfo, workdir: Path) -> bytes:
    """Exactly 8 s of mono 22.05 kHz 64 kbps MP3: from 0:03 (or 0:00 for short clips)."""
    if info.kind == "image":
        raise MediaUnavailable(MSG_PHOTO_NO_AUDIO, "image has no audio")
    if not info.has_audio:
        raise MediaUnavailable(MSG_NO_AUDIO, "no audio stream")
    start = 3.0 if info.duration >= 11.0 else 0.0
    out = workdir / "clip.mp3"
    rc, _, err = await run_cmd(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.2f}", "-i", str(info.path),
            "-t", "8", "-vn", "-ac", "1", "-ar", "22050", "-b:a", "64k",
            "-f", "mp3", str(out),
        ],
        timeout=15,
    )
    if rc != 0 or not out.exists() or out.stat().st_size == 0:
        raise MediaUnavailable(MSG_UNSUPPORTED, f"audio extract failed: {err[:200]!r}")
    return out.read_bytes()


async def extract_wav_clip(info: MediaInfo, workdir: Path, rate: float = 1.0, seconds: int = 12,
                           start: float | None = None, clean: bool = False) -> bytes:
    """12 s mono 16 kHz WAV for Shazam. rate != 1 plays it faster/slower WITH pitch change:
    that undoes "sped up" / "slowed" edits, so the original track can be matched."""
    if start is None:
        start = 1.0 if info.duration >= 14.0 else 0.0
    out = workdir / f"shz_{rate:.2f}_{uuid.uuid4().hex[:6]}.wav"
    filters = []
    if rate != 1.0:
        filters.append(f"aresample=44100,asetrate={44100 * rate:.0f},aresample=16000")
    if clean:  # phone-recorded / noisy covers: cut rumble & hiss, even out loudness
        filters.append("highpass=f=90,lowpass=f=7500,dynaudnorm=f=150:g=15")
    af = ["-af", ",".join(filters)] if filters else []
    rc, _, err = await run_cmd(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{start:.2f}", "-i", str(info.path), "-vn", *af,
         "-ac", "1", "-ar", "16000", "-t", str(seconds), "-f", "wav", str(out)],
        timeout=15,
    )
    if rc != 0 or not out.exists() or out.stat().st_size < 1000:
        raise MediaUnavailable(MSG_UNSUPPORTED, f"wav extract failed: {err[:200]!r}")
    return out.read_bytes()


async def _grab_frame(src: Path, at: float, out: Path, width: int = 640) -> bytes | None:
    """Grabs one frame. Fast path decodes only the nearest keyframe (~3x less CPU,
    matters on 0.1-0.5 vCPU free instances); falls back to an exact seek."""
    base = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    tail = ["-frames:v", "1", "-vf", f"scale='min({width},iw)':-2", "-q:v", "3", "-update", "1", str(out)]
    ss = ["-ss", f"{at:.2f}"] if at > 0 else []
    attempts = [
        base + ["-skip_frame", "nokey"] + ss + ["-noaccurate_seek", "-i", str(src)] + tail,
        base + ss + ["-i", str(src)] + tail,
    ]
    for args in attempts:
        out.unlink(missing_ok=True)
        rc, _, err = await run_cmd(args, timeout=15)
        if rc == 0 and out.exists() and out.stat().st_size > 0:
            return out.read_bytes()
        log.debug("frame grab at %.2fs attempt failed: %r", at, err[:200])
    log.warning("frame grab at %.2fs failed", at)
    return None


async def extract_frames(info: MediaInfo, workdir: Path, width: int = 640) -> list[bytes]:
    """3 JPEG frames at 20/50/80% of the duration (1 frame for photos). Item search uses wider
    frames so logos and printed model names stay readable."""
    if info.kind == "audio":
        raise MediaUnavailable(MSG_NO_VIDEO, "audio-only media")
    if info.kind == "image" or info.duration < 0.5:
        timestamps = [0.0]
    else:
        timestamps = [info.duration * p for p in (0.2, 0.5, 0.8)]
    results = await asyncio.gather(
        *(_grab_frame(info.path, t, workdir / f"frame_{i}.jpg", width) for i, t in enumerate(timestamps))
    )
    frames = [f for f in results if f]
    if not frames:
        raise MediaUnavailable(MSG_UNSUPPORTED, "no frames extracted")
    return frames


async def crop_item(frame: bytes, box: list, workdir: Path) -> bytes | None:
    """Cuts one item out of a frame (box = [ymin, xmin, ymax, xmax] on 0-1000) with a little margin."""
    try:
        y1, x1, y2, x2 = (max(0, min(1000, int(v))) for v in box[:4])
    except (TypeError, ValueError):
        return None
    if y2 - y1 < 40 or x2 - x1 < 40:
        return None
    pad_y, pad_x = (y2 - y1) * 0.08, (x2 - x1) * 0.08
    y1, x1 = max(0, y1 - pad_y), max(0, x1 - pad_x)
    y2, x2 = min(1000, y2 + pad_y), min(1000, x2 + pad_x)
    src, out = workdir / f"crop_src_{uuid.uuid4().hex[:6]}.jpg", workdir / f"crop_{uuid.uuid4().hex[:6]}.jpg"
    src.write_bytes(frame)
    vf = (f"crop=w=iw*{(x2 - x1) / 1000:.4f}:h=ih*{(y2 - y1) / 1000:.4f}:"
          f"x=iw*{x1 / 1000:.4f}:y=ih*{y1 / 1000:.4f}")
    rc, _, err = await run_cmd(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                                "-vf", vf, "-q:v", "3", str(out)], timeout=10)
    if rc != 0 or not out.exists():
        log.debug("crop failed: %r", err[:200])
        return None
    return out.read_bytes()


# =============================================================================
# Stripe (hosted Checkout + Customer Portal; no SDK needed)
# =============================================================================

CURRENCY_SIGNS = {"usd": "$", "eur": "€", "gbp": "£", "cad": "CA$", "aud": "A$", "brl": "R$"}


class StripeError(Exception):
    pass


class StripeClient:
    API = "https://api.stripe.com/v1"

    def __init__(self, http: httpx.AsyncClient, key: str):
        self._http = http
        self._key = key
        self.price_labels: dict[str, str] = {}  # Price ID → "$4.99"

    async def _call(self, method: str, path: str, form: list[tuple[str, str]] | None = None) -> dict:
        try:
            resp = await self._http.request(
                method, f"{self.API}/{path}", content=urlencode(form or []).encode() if form else None,
                headers={"Authorization": f"Bearer {self._key}", "Stripe-Version": "2024-06-20",
                         "Content-Type": "application/x-www-form-urlencoded"},
                timeout=httpx.Timeout(15.0, connect=5.0))
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise StripeError(f"Stripe {path}: {exc}") from exc
        if resp.status_code >= 400:
            raise StripeError(f"Stripe {path}: HTTP {resp.status_code} "
                              f"{_s((data.get('error') or {}).get('message'))[:200]}")
        return data

    async def load_prices(self, price_ids: list[str]) -> None:
        for pid in price_ids:
            try:
                p = await self._call("GET", f"prices/{pid}")
            except StripeError as exc:
                log.error("Stripe price %s: %s", pid, exc)
                continue
            amount, cur = (p.get("unit_amount") or 0) / 100, _s(p.get("currency")).lower()
            sign = CURRENCY_SIGNS.get(cur)
            num = f"{amount:.2f}".rstrip("0").rstrip(".") if amount == int(amount) else f"{amount:.2f}"
            self.price_labels[pid] = f"{sign}{num}" if sign else f"{num} {cur.upper()}"

    async def create_checkout(self, form: list[tuple[str, str]]) -> str:
        return _s((await self._call("POST", "checkout/sessions", form)).get("url"))

    async def portal(self, customer: str, return_url: str) -> str:
        return _s((await self._call("POST", "billing_portal/sessions",
                                    [("customer", customer), ("return_url", return_url)])).get("url"))

    async def subscription(self, sub_id: str) -> dict:
        return await self._call("GET", f"subscriptions/{sub_id}")


def subscription_period_end(sub: dict) -> datetime | None:
    """Works with old (top-level) and new (per-item) Stripe API shapes."""
    end = sub.get("current_period_end") or next(
        (it.get("current_period_end") for it in ((sub.get("items") or {}).get("data") or [])
         if it.get("current_period_end")), None)
    return datetime.fromtimestamp(int(end), timezone.utc) if end else None


def verify_stripe_signature(raw: bytes, header: str, secret: str, tolerance: int = 300) -> bool:
    try:
        parts = dict(kv.split("=", 1) for kv in header.split(",") if "=" in kv)
        ts = int(parts.get("t", "0"))
    except ValueError:
        return False
    if not ts or abs(time.time() - ts) > tolerance:
        return False
    expected = hmac.new(secret.encode(), f"{ts}.".encode() + raw, hashlib.sha256).hexdigest()
    sigs = [kv.split("=", 1)[1] for kv in header.split(",") if kv.startswith("v1=")]
    return any(hmac.compare_digest(expected, sig) for sig in sigs)


# =============================================================================
# AudD (music recognition)
# =============================================================================


class AudDClient:
    URL = "https://api.audd.io/"

    def __init__(self, http: httpx.AsyncClient, token: str, timeout: float):
        self._http = http
        self._token = token
        self._timeout = timeout

    async def recognize(self, mp3: bytes) -> dict | None:
        """Returns AudD `result` (None = not recognised). Raises ExternalServiceError."""
        for attempt in (1, 2):
            try:
                resp = await self._http.post(
                    self.URL,
                    data={"api_token": self._token, "return": "apple_music,spotify"},
                    files={"file": ("clip.mp3", mp3, "audio/mpeg")},
                    timeout=httpx.Timeout(self._timeout, connect=3.0),
                )
            except httpx.TimeoutException as exc:
                raise ExternalServiceError(f"AudD timeout: {exc}") from exc
            except httpx.TransportError as exc:
                if attempt == 1:
                    await asyncio.sleep(0.25)
                    continue
                raise ExternalServiceError(f"AudD transport error: {exc}") from exc

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == 1 and resp.status_code >= 500:
                    await asyncio.sleep(0.25)
                    continue
                raise ExternalServiceError(f"AudD HTTP {resp.status_code}")
            try:
                data = resp.json()
            except ValueError as exc:
                raise ExternalServiceError("AudD returned non-JSON") from exc
            if data.get("status") == "success":
                return data.get("result") or None
            err = data.get("error") or {}
            raise ExternalServiceError(f"AudD error {err.get('error_code')}: {err.get('error_message')}")
        raise ExternalServiceError("AudD unreachable")


def _norm(text: str) -> str:
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text.lower())  # drop "(feat. X)", "[Remix]"
    text = re.sub(r"\b(feat|ft|featuring|prod|remix|edit|version|sped up|slowed|reverb)\b.*", " ", text)
    return re.sub(r"[^\w]+", " ", text).strip()


_CYR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюяіїєґ",
                ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s",
                 "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e", "yu", "ya", "i", "yi", "ye", "g"],
                strict=True))


def _translit(text: str) -> str:
    return "".join(_CYR.get(ch, ch) for ch in text.lower())


def _similar(a: str, b: str) -> float:
    """0..1; Cyrillic and Latin spellings of the same name count as similar (Пошлая ≈ Poshlaya)."""
    from difflib import SequenceMatcher

    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    best = 0.0
    for x, y in ((a, b), (_translit(a), _translit(b))):
        x, y = x.replace("kh", "h").replace("j", "y"), y.replace("kh", "h").replace("j", "y")
        if x in y or y in x:
            return 1.0
        best = max(best, SequenceMatcher(None, x, y).ratio())
    return best


async def verify_song_itunes(http: httpx.AsyncClient, artist: str, title: str) -> dict | None:
    """Checks that the AI's answer is a real released track (free iTunes Search API, no key).
    Returns an AudD-shaped result with canonical names + Apple Music link, or None."""
    # US catalogue + the user's local one (Russian/Brazilian/… artists may only be listed there)
    local = {"ru": "ru", "uk": "ua", "es": "es", "pt": "br", "de": "de", "fr": "fr", "it": "it"}.get(CURRENT_LANG.get())
    countries = ["us"] + ([local] if local else [])

    async def search(country: str) -> list[dict]:
        try:
            resp = await http.get("https://itunes.apple.com/search",
                                  params={"term": f"{artist} {title}"[:200], "entity": "song", "limit": 15,
                                          "country": country},
                                  timeout=httpx.Timeout(5.0, connect=3.0))
            return resp.json().get("results") or [] if resp.status_code == 200 else []
        except (httpx.HTTPError, ValueError) as exc:
            log.info("iTunes verification (%s) unavailable: %s", country, exc)
            return []

    items = [it for batch in await asyncio.gather(*(search(c) for c in countries)) for it in batch]
    best, best_score = None, 0.0
    for it in items:
        t_score = _similar(title, _s(it.get("trackName")))
        a_score = _similar(artist, _s(it.get("artistName")))
        score = t_score * 0.6 + a_score * 0.4
        if t_score >= 0.75 and a_score >= 0.6 and score > best_score:
            best, best_score = it, score
    if not best:
        return None
    return {
        "artist": _s(best.get("artistName")), "title": _s(best.get("trackName")),
        "album": _s(best.get("collectionName")), "release_date": _s(best.get("releaseDate"))[:10],
        "apple_music": {"url": _s(best.get("trackViewUrl")).split("?")[0]},
        "_cover": _s(best.get("artworkUrl100")).replace("100x100", "600x600"),
    }


def shazam_to_result(track: dict) -> dict:
    """Shazam `track` → the same shape format_song() expects from AudD."""
    result: dict[str, Any] = {"title": _s(track.get("title")), "artist": _s(track.get("subtitle"))}
    for section in track.get("sections") or []:
        for meta in (section or {}).get("metadata") or []:
            label = _s((meta or {}).get("title")).lower()
            if label == "album":
                result["album"] = _s(meta.get("text"))
            elif label == "released":
                result["release_date"] = _s(meta.get("text"))
    if _s(track.get("url")).startswith("https://"):
        result["song_link"] = track["url"]
    cover = _s((track.get("images") or {}).get("coverart"))
    if cover.startswith("https://"):
        result["_cover"] = cover
    return result


class ShazamClient:
    """Shazam via the open-source shazamio library (unofficial API, free). Never raises:
    any problem simply counts as "not recognised", AudD stays the primary engine."""

    def __init__(self, timeout: float = 8.0):
        from shazamio import Shazam  # lazy: optional dependency

        self._shazam = Shazam(segment_duration_seconds=12)
        self._timeout = timeout

    async def recognize(self, wav: bytes) -> dict | None:
        try:
            out = await asyncio.wait_for(self._shazam.recognize(wav), timeout=self._timeout)
        except Exception as exc:  # noqa: BLE001
            log.warning("Shazam failed: %s", re.sub(r"\s+", " ", str(exc))[:200] or type(exc).__name__)
            return None
        track = (out or {}).get("track")
        if not track or not _s(track.get("title")):
            return None
        return shazam_to_result(track)


# =============================================================================
# Gemini Vision (google-genai) with model fallback
# =============================================================================

MOVIE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "title": {"type": "string"},
        "original_title": {"type": "string"},
        "year": {"type": "string"},
        "kind": {"type": "string", "enum": ["movie", "tv_series", "anime", "documentary", "music_video", "other"]},
        "season": {"type": "string"},
        "episode": {"type": "string"},
        "actors": {"type": "array", "items": {"type": "string"}},
        "scene": {"type": "string"},
        "approx_timestamp": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["found", "title", "year", "kind", "scene", "confidence"],
}

ITEMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short natural name, grammatically correct"},
                    "print": {"type": "string", "description": "Exact text / graphic printed on it, or empty"},
                    "category": {"type": "string"},
                    "brand": {"type": "string"},
                    "model": {"type": "string"},
                    "color": {"type": "string"},
                    "description": {"type": "string"},
                    "search_query": {"type": "string"},
                    "confidence": {"type": "number"},
                    "frame": {"type": "integer", "description": "0-based index of the image where it's best visible"},
                    "box_2d": {"type": "array", "items": {"type": "integer"},
                               "description": "[ymin, xmin, ymax, xmax] of the item in that image, 0-1000 scale"},
                },
                "required": ["name", "category", "brand", "print", "description", "search_query", "confidence",
                             "frame", "box_2d"],
            },
        },
    },
    "required": ["found", "items"],
}

PLACE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "name": {"type": "string", "description": "Specific place: landmark, venue, beach, street, viewpoint…"},
        "city": {"type": "string"},
        "country": {"type": "string"},
        "kind": {"type": "string", "description": "landmark / restaurant / hotel / beach / nature / street / …"},
        "description": {"type": "string", "description": "One sentence: what it is and why it's worth a visit"},
        "confidence": {"type": "number"},
    },
    "required": ["found", "name", "city", "country", "confidence"],
}

ALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"movie": MOVIE_SCHEMA, "items": ITEMS_SCHEMA, "place": PLACE_SCHEMA},
    "required": ["movie", "items", "place"],
}

_MOVIE_RULES = (
    "MOVIE / TV TASK: Identify the movie, TV series or anime this footage comes from. Use actors' faces, "
    "sets, costumes, cinematography, subtitles, logos and on-screen text. Social-media edits often add "
    "captions or watermarks — ignore those. If it's original creator content (vlog, skit, ad, UGC), "
    "set found=false. If you are not reasonably sure (confidence < 0.5), set found=false — never invent a title. "
    "Fill `scene` with a one-sentence description of what happens and where it sits in the story. "
    "Fill `approx_timestamp` with an approximate moment (e.g. '~01:12:00' or 'S02E05, ~00:34:00') only if you "
    "genuinely know it, otherwise an empty string. Use empty strings for unknown fields."
)
_ITEMS_RULES = (
    "FASHION / ITEMS TASK: List up to 5 of the most prominent clothing items, shoes, accessories, bags, "
    "eyewear, jewellery or notable products. For each: category (1-2 words, e.g. 'hoodie', 'sneakers'), "
    "brand (only if a logo/design makes it identifiable, otherwise 'Unknown'), model name if recognisable "
    "(else empty), main color (1 word), description (max 6 words, only what tells it apart, e.g. "
    "'cropped, with a back print'; empty if nothing notable), and a precise shopping search query "
    "(brand + model + color + item type, or a descriptive query if brand is unknown). "
    "PRINTS: read any print, slogan, logo text or graphic on clothes and accessories carefully, letter by letter, "
    "even if you don't recognise it; put the exact text in `print` (or a 2-4 word description of a graphic, "
    "e.g. 'skull with roses'), and add it to search_query in quotes. "
    "`name`: how a person would call the item in 2-6 words — colour + item, plus brand/model if known and the "
    "print if there is one (e.g. 'Black T-shirt with a “Nirvana” print', 'Nike Dunk Low sneakers'). "
    "It must be grammatically correct in the output language: adjectives agree in gender and number with the "
    "noun, natural word order. `frame` + `box_2d`: the image where the item is clearest and a tight box around "
    "the item there (it is cropped for a visual 'search by photo'). confidence is 0..1. found=false if nothing identifiable is visible. "
    "Prioritise distinctive things (gear, gadgets, instruments, sneakers, bags, watches, statement pieces) over "
    "generic basics like plain black pants. For instruments, synthesizers, electronics, cameras, cars, watches "
    "and sneakers try hard to name the exact make and model: read logos and model names printed on the device, "
    "and use distinctive design (layout and number of keys/knobs/pads/screens, colour scheme, shape). "
    "If you know the brand but can only narrow the model down to 2-3 candidates, put them in `model` "
    "(e.g. 'MODX6 or Montage 6') instead of leaving it empty."
)
_PLACE_RULES = (
    "PLACE TASK: Identify where this was filmed: a specific landmark, city, beach, restaurant, hotel, street or "
    "nature spot. Use architecture, skyline, landscape, signs, shop names, licence plates, language on signs, "
    "and the Reel caption / location tag if given. Be as specific as possible (e.g. 'Oia viewpoint, Santorini' "
    "rather than just 'Greece'). If it's a generic room, studio or an unrecognisable street, set found=false. "
    "Never invent a place; confidence is 0..1 (0.8+ only if you clearly recognise it)."
)
PROMPTS = {
    FIND_MOVIE: ("You receive up to 3 frames (20%, 50%, 80% of the clip) from an Instagram Reel. " + _MOVIE_RULES, MOVIE_SCHEMA),
    FIND_ITEM: ("You receive up to 3 frames (20%, 50%, 80% of the clip) from an Instagram Reel or a photo. " + _ITEMS_RULES, ITEMS_SCHEMA),
    FIND_PLACE: ("You receive up to 3 frames (20%, 50%, 80% of the clip) from an Instagram Reel or a photo. " + _PLACE_RULES, PLACE_SCHEMA),
    FIND_ALL: (
        "You receive up to 3 frames (20%, 50%, 80% of the clip) from an Instagram Reel. Solve ALL THREE tasks and "
        "return them under the keys `movie`, `items` and `place`.\n" + _MOVIE_RULES + "\n" + _ITEMS_RULES
        + "\n" + _PLACE_RULES,
        ALL_SCHEMA,
    ),
}

RETRYABLE_GEMINI_CODES = {404, 408, 429, 499, 500, 502, 503, 504}

SONG_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "artist": {"type": "string", "description": "Artist of the ORIGINAL released track"},
        "title": {"type": "string", "description": "Title of the ORIGINAL released track"},
        "version": {"type": "string", "description": "e.g. 'original', 'sped up', 'slowed', 'phonk remix', "
                                                     "'cover', 'mashup'; empty if unsure"},
        "confidence": {"type": "number", "description": "0..1 — how sure you are about artist+title"},
        "evidence": {"type": "string", "description": "short: melody / lyrics heard / caption / audio label"},
    },
    "required": ["found", "artist", "title", "confidence"],
}


def song_prompt(caption: str, audio_label: str, has_frame: bool = False) -> str:
    ctx = []
    if audio_label:
        ctx.append(f"Instagram audio label: {audio_label}")
    if caption:
        ctx.append(f"Reel caption: {caption}")
    context = "\n".join(ctx) or "(no caption or audio label available)"
    return (
        "You are a music identification expert. Identify the song playing in this audio clip from an "
        "Instagram Reel. Audio fingerprinting (Shazam) already failed, so it is probably a remix, "
        "sped-up/slowed edit, phonk/funk edit, cover or mashup — identify the ORIGINAL track it is based on "
        "(recognise the melody, hook, vocals, lyrics or sample). Instrumental covers (piano, synth, guitar, "
        "violin…) are common: name the song whose MELODY is being played. Songs can be from any country "
        "(Russian, Ukrainian, Latin, K-pop, Turkish, Arabic…).\n"
        + ("You also get a frame from the video: on-screen text, the performer or the instrument may help.\n"
           if has_frame else "")
        + 
        "Context (often names the track or credits it with 🎵, 'song:', '#songname'; may also be unrelated):\n"
        f"{context}\n\n"
        "Write artist and title exactly as on Spotify / Apple Music, in the original script "
        "(e.g. Cyrillic for Russian artists). "
        "Rules: never invent a song. If you can't name a real released track, return found=false. "
        "confidence must reflect real certainty (0.9 = you clearly recognise it)."
    )


class _ModelFailed(Exception):
    pass


class VisionClient:
    TOTAL_BUDGET_SEC = 25.0  # whole fallback chain must fit in this
    HEDGE_SEC = 5.0          # no answer after this → start the next model in parallel

    def __init__(self, api_key: str, models: list[str], timeout_sec: float, thinking_level: str,
                 backup: tuple[str, str, str] | None = None):
        self._timeout = max(timeout_sec, 12.0)
        self._models = list(dict.fromkeys(m.strip() for m in models if m and m.strip()))  # dedupe, keep order
        self._dead_models: set[str] = set()  # 404 "model not found" → don't try again
        self._thinking_level = thinking_level
        self._backup = backup if backup and all(backup) else None
        self._backup_http: httpx.AsyncClient | None = None
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(
                timeout=int(self._timeout * 1000),
                retry_options=genai_types.HttpRetryOptions(attempts=1),
            ),
        )

    @property
    def describe(self) -> str:
        chain = " → ".join(self._models)
        return chain + (f" → backup {self._backup[2]}" if self._backup else "")

    def _thinking_config(self, model: str) -> genai_types.ThinkingConfig | None:
        m = model.lower()
        if m.startswith("gemini-2.5-flash"):
            return genai_types.ThinkingConfig(thinking_budget=0)  # no thinking => lowest latency
        if m.startswith("gemini-3"):
            level = getattr(genai_types.ThinkingLevel, self._thinking_level, genai_types.ThinkingLevel.MINIMAL)
            return genai_types.ThinkingConfig(thinking_level=level)
        return None

    async def analyze(self, task: str, frames: list[bytes], context: str = "") -> dict:
        prompt, schema = PROMPTS[task]
        if context:
            prompt += "\n\nContext from the Reel (may help, may be unrelated):\n" + context[:800]
        lang = CURRENT_LANG.get()
        if lang != "en" and lang in LANGUAGES:
            prompt += (f"\n\nWrite all descriptive text values (name, scene, description, color, category, kind) in "
                       f"{LANGUAGES[lang][1]}, grammatically correct (gender, number and case agreement, natural word "
                       "order — e.g. Russian 'чёрная футболка', not 'черный футболка'). Keep proper names (titles, "
                       "brands, models, actors) and printed text unchanged, and keep search_query in English.")
        contents: list[Any] = [genai_types.Part.from_bytes(data=f, mime_type="image/jpeg") for f in frames]
        contents.append(prompt)
        try:
            return await self._generate(contents, schema)
        except ExternalServiceError as exc:
            if not self._backup:
                raise
            log.warning("%s → trying backup AI %s", exc, self._backup[2])
            return await self._backup_generate(frames, prompt, schema, str(exc))

    async def identify_song(self, wav: bytes, caption: str, audio_label: str,
                            frames: list[bytes] | None = None) -> dict:
        contents: list[Any] = [genai_types.Part.from_bytes(data=wav, mime_type="audio/wav")]
        contents += [genai_types.Part.from_bytes(data=f, mime_type="image/jpeg") for f in frames or []]
        contents.append(song_prompt(caption, audio_label, bool(frames)))
        return await self._generate(contents, SONG_SCHEMA, strong_first=True)

    async def _call_model(self, model: str, contents: list[Any], schema: dict, timeout: float,
                          hi_res: bool = False) -> dict:
        """One Gemini attempt. Returns parsed JSON or raises _ModelFailed(reason)."""
        started = asyncio.get_running_loop().time()
        tuned = True  # minimal thinking + medium image detail = much faster; plain call if rejected
        for attempt in (1, 2):
            extra: dict[str, Any] = {}
            if tuned:
                extra = {"thinking_config": self._thinking_config(model),
                         "media_resolution": genai_types.MediaResolution.MEDIA_RESOLUTION_HIGH if hi_res
                         else genai_types.MediaResolution.MEDIA_RESOLUTION_MEDIUM}
            config = genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=schema,
                temperature=0.2,
                max_output_tokens=2048,
                **extra,
            )
            try:
                resp = await asyncio.wait_for(
                    self._client.aio.models.generate_content(model=model, contents=contents, config=config),
                    timeout=timeout,
                )
            except asyncio.TimeoutError as exc:
                raise _ModelFailed(f"{model}: timeout {timeout:.0f}s") from exc
            except genai_errors.APIError as exc:
                msg = _s(exc.message)
                if exc.code == 400 and tuned and attempt == 1:
                    log.info("Gemini %s rejected speed settings (%s) → plain call", model, msg[:120])
                    tuned = False
                    continue
                if exc.code == 404:
                    self._dead_models.add(model)
                raise _ModelFailed(f"{model}: HTTP {exc.code} {msg[:160]}") from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                raise _ModelFailed(f"{model}: transport {exc}") from exc
            text = (resp.text or "").strip()
            if not text:
                raise _ModelFailed(f"{model}: empty/blocked response")
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise _ModelFailed(f"{model}: invalid JSON") from exc
            log.info("Gemini %s answered in %.1fs", model, asyncio.get_running_loop().time() - started)
            return data
        raise _ModelFailed(f"{model}: request rejected")

    def strong_chain(self) -> list[str]:
        """Full Flash models (best at recognising melodies) before Lite ones, newest first."""
        def ver(m: str) -> float:
            v = re.search(r"(\d+(?:\.\d+)?)", m)
            return float(v.group(1)) if v else 0.0
        return sorted(self._models, key=lambda m: ("lite" in m, "preview" in m, -ver(m)))

    async def _generate(self, contents: list[Any], schema: dict, hi_res: bool = False,
                        strong_first: bool = False) -> dict:
        """Hedged fallback chain: start the 1st model; if it is slow (> HEDGE_SEC) or fails, start the
        next one in parallel; the first good answer wins, the rest are cancelled."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.TOTAL_BUDGET_SEC
        queue = [m for m in (self.strong_chain() if strong_first else self._models) if m not in self._dead_models]
        errors: list[str] = []
        pending: dict[asyncio.Task, str] = {}

        def start_next() -> bool:
            left = deadline - loop.time()
            if not queue or left < 3:
                return False
            model = queue.pop(0)
            task = asyncio.create_task(self._call_model(model, contents, schema, min(self._timeout, left), hi_res))
            pending[task] = model
            return True

        start_next()
        try:
            while pending:
                wait_for = self.HEDGE_SEC if queue else max(deadline - loop.time(), 0.1)
                done, _ = await asyncio.wait(set(pending), timeout=wait_for,
                                             return_when=asyncio.FIRST_COMPLETED)
                if not done:  # current model(s) slow → hedge with the next one
                    if queue:
                        log.info("Gemini %s is slow → also asking %s", ", ".join(pending.values()), queue[0])
                        start_next()
                    elif loop.time() >= deadline:
                        break
                    continue
                for task in done:
                    model = pending.pop(task)
                    exc = task.exception()
                    if exc is None:
                        return task.result()
                    errors.append(str(exc))
                    log.warning("Gemini %s failed (%s)", model, exc)
                    if not pending:
                        start_next()
        finally:
            for task in pending:
                task.cancel()
        raise ExternalServiceError("All Gemini models failed: " + ("; ".join(errors[-4:]) or "no models"))

    async def discover_models(self) -> None:
        """Adds Flash models that really exist for this API key to the end of the chain
        (names change between Gemini generations; a wrong name in settings is skipped)."""
        try:
            pager = await asyncio.wait_for(self._client.aio.models.list(), timeout=8)
            names = []
            async for m in pager:
                name = _s(m.name).removeprefix("models/")
                actions = m.supported_actions or []
                if "generateContent" in actions and "flash" in name:
                    names.append(name)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not list Gemini models: %s", exc)
            return
        bad = ("tts", "image", "live", "audio", "embed", "robotics", "computer", "exp", "8b", "native")
        usable = [n for n in names if not any(b in n for b in bad)]
        if not usable:
            return
        self._dead_models |= {m for m in self._models if m not in names}

        def rank(n: str) -> tuple:
            ver = re.search(r"(\d+(?:\.\d+)?)", n)
            return ("preview" in n, -(float(ver.group(1)) if ver else 0), "lite" not in n, n)

        configured = [m for m in self._models if m in names]
        extra = [n for n in sorted(usable, key=rank) if n not in configured]
        self._models = (configured + extra)[: max(4, len(configured))]
        log.info("Gemini models available: %s", ", ".join(sorted(usable)))
        if self._dead_models:
            log.warning("Gemini models from settings that don't exist (skipped): %s",
                        ", ".join(sorted(self._dead_models)))

    async def _backup_generate(self, frames: list[bytes], prompt: str, schema: dict, why: str) -> dict:
        """Same task on an OpenAI-compatible vision model (OpenRouter, OpenAI, Groq, …)."""
        import base64

        url, key, model = self._backup  # type: ignore[misc]
        if self._backup_http is None:
            self._backup_http = httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=5.0))
        content: list[dict] = [{"type": "text", "text": prompt + "\n\nReply with ONLY a JSON object "
                                "that matches this JSON schema:\n" + json.dumps(schema)}]
        content += [{"type": "image_url",
                     "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(f).decode()}}
                    for f in frames]
        body = {"model": model, "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_object"}, "temperature": 0.2, "max_tokens": 2048}
        try:
            resp = await self._backup_http.post(f"{url}/chat/completions", json=body,
                                                headers={"Authorization": f"Bearer {key}"})
            data = resp.json()
            if resp.status_code >= 400:
                raise ExternalServiceError(f"backup AI HTTP {resp.status_code}: {str(data)[:200]}")
            text = _s(((data.get("choices") or [{}])[0].get("message") or {}).get("content"))
            text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
            result = json.loads(text)
        except (httpx.HTTPError, ValueError, AttributeError, IndexError) as exc:
            raise ExternalServiceError(f"{why}; backup AI failed: {exc}") from exc
        log.info("Backup AI %s answered", model)
        return result

    async def aclose(self) -> None:
        if self._backup_http is not None:
            await self._backup_http.aclose()
        closer = getattr(self._client.aio, "aclose", None)
        if closer:
            try:
                await closer()
            except Exception:  # noqa: BLE001
                pass


# =============================================================================
# Result formatting
# =============================================================================


def _s(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def url_button(title: str, url: str) -> dict:
    return {"type": "web_url", "title": title[:20], "url": url}


def make_card(title: str, subtitle: str, buttons: list[dict], image: str = "") -> dict:
    """One element of an Instagram generic template (tappable buttons — plain links in DMs often
    aren't clickable for people who don't follow the account)."""
    card: dict[str, Any] = {"title": title[:80] or "Result", "buttons": buttons[:3]}
    if subtitle:
        card["subtitle"] = subtitle[:80]
    if image.startswith("https://"):
        card["image_url"] = image
    return card


def cards_as_text(cards: list[dict]) -> str:
    """Fallback when the template can't be sent: same links as plain text."""
    lines: list[str] = []
    for c in cards:
        lines.append(c.get("title", ""))
        lines += [f"  {b['title']}: {b['url']}" for b in c.get("buttons", [])]
    return "\n".join(lines)


def song_cover(result: dict) -> str:
    cover = _s(result.get("_cover"))
    if cover:
        return cover
    art = ((result.get("apple_music") or {}).get("artwork") or {}).get("url")
    if art:
        return _s(art).replace("{w}", "600").replace("{h}", "600")
    images = (((result.get("spotify") or {}).get("album") or {}).get("images")) or []
    return _s(images[0].get("url")) if images else ""


def song_card(result: dict) -> dict:
    artist, title = _s(result.get("artist")), _s(result.get("title"))
    query = quote_plus(f"{artist} {title}")
    spotify = ((result.get("spotify") or {}).get("external_urls") or {}).get("spotify") or (
        f"https://open.spotify.com/search/{query}")
    apple = (result.get("apple_music") or {}).get("url") or f"https://music.apple.com/us/search?term={query}"
    album, release = _s(result.get("album")), _s(result.get("release_date"))
    subtitle = album + (f" ({release[:4]})" if album and release else "")
    return make_card(f"{artist} — {title}", subtitle, [
        url_button("▶️ Spotify", spotify),
        url_button("🍎 Apple Music", apple),
        url_button("📺 YouTube", f"https://www.youtube.com/results?search_query={query}"),
    ], song_cover(result))


def format_song(result: dict, links: bool = True) -> str:
    artist, title = _s(result.get("artist")), _s(result.get("title"))
    album, release = _s(result.get("album")), _s(result.get("release_date"))
    query = quote_plus(f"{artist} {title}")
    spotify = ((result.get("spotify") or {}).get("external_urls") or {}).get("spotify") or (
        f"https://open.spotify.com/search/{quote_plus(f'{artist} {title}')}"
    )
    apple = (result.get("apple_music") or {}).get("url") or f"https://music.apple.com/us/search?term={query}"
    lines = [_t("🎵 Song found!")]
    if result.get("_note"):
        lines.append(_s(result["_note"]))
    lines.append(f"{artist} — {title}")
    if album:
        lines.append(f"💿 {album}" + (f" ({release[:4]})" if release else ""))
    if links:
        lines += [f"▶️ Spotify: {spotify}", f"🍎 Apple Music: {apple}"]
        if result.get("song_link"):
            lines.append(f"🔗 All platforms: {result['song_link']}")
    return "\n".join(lines)


def movie_is_found(data: dict) -> bool:
    try:
        conf = float(data.get("confidence") or 0)
    except (TypeError, ValueError):
        conf = 0.0
    return bool(data.get("found")) and bool(_s(data.get("title"))) and conf >= 0.45


def format_movie(data: dict) -> str:
    title, year = _s(data.get("title")), _s(data.get("year"))
    original = _s(data.get("original_title"))
    kind = {"movie": "Movie", "tv_series": "TV series", "anime": "Anime", "documentary": "Documentary",
            "music_video": "Music video"}.get(_s(data.get("kind")), "")
    kind = _t(kind) if kind else ""
    season, episode = _s(data.get("season")), _s(data.get("episode"))
    head = f"{title} ({year})" if year else title
    if original and original.lower() != title.lower():
        head += f" / {original}"
    lines = [_t("🎬 It's from:"), head]
    meta = kind
    if season or episode:
        se = f"S{season}" if season else ""
        se += f"E{episode}" if episode else ""
        meta = f"{meta}, {se}" if meta else se
    if meta:
        lines.append(f"📺 {meta}")
    actors = [a for a in (data.get("actors") or []) if _s(a)][:4]
    if actors:
        lines.append(_t("🎭 Cast: {x}", x=", ".join(actors)))
    if _s(data.get("scene")):
        lines.append(_t("📍 Scene: {x}", x=_s(data.get("scene"))))
    if _s(data.get("approx_timestamp")):
        lines.append(_t("⏱ Moment: {x}", x=_s(data.get("approx_timestamp"))))
    try:
        conf = float(data.get("confidence") or 0)
    except (TypeError, ValueError):
        conf = 0.0
    lines.append(_t("🎯 Confidence: high") if conf >= 0.8 else _t("🎯 Confidence: medium"))
    return "\n".join(lines)


def movie_card(data: dict) -> dict:
    title, year = _s(data.get("title")), _s(data.get("year"))
    q = quote_plus(f"{title} {year}".strip())
    region = {"es": "es", "pt": "br", "de": "de", "fr": "fr", "it": "it", "uk": "ua"}.get(CURRENT_LANG.get(), "us")
    return make_card(f"🎬 {title}" + (f" ({year})" if year else ""), _t("Where to watch & details"), [
        url_button(_t("📺 Where to watch"), f"https://www.justwatch.com/{region}/search?q={quote_plus(title)}"),
        url_button("⭐ IMDb", f"https://www.imdb.com/find/?q={q}"),
        url_button(_t("🎞 Trailer"), f"https://www.youtube.com/results?search_query={q}+trailer"),
    ])


def filter_items(data: dict) -> list[dict]:
    items = []
    for it in data.get("items") or []:
        try:
            conf = float(it.get("confidence") or 0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf >= 0.3 and _s(it.get("search_query")):
            items.append(it)
    return items[:5]


def item_names(items: list[dict]) -> list[str]:
    """Short names: 'Nike Dunk Low (sneakers)', 'Black hoodie'. A detail is added only when two
    items would otherwise look the same."""
    names = []
    for it in items:
        if _s(it.get("name")):  # written by the AI in the user's language, grammatically correct
            names.append(_s(it.get("name"))[:1].upper() + _s(it.get("name"))[1:80])
            continue
        brand = _s(it.get("brand"))
        brand = "" if brand.lower() in ("unknown", "n/a", "none", "-") else brand
        model, category, color = _s(it.get("model")), _s(it.get("category")).lower(), _s(it.get("color")).lower()
        if brand and model:
            name = f"{brand} {model}" + (f" ({category})" if category and category not in model.lower() else "")
        elif brand:
            name = f"{brand} {category}".strip()
        else:
            name = " ".join(x for x in (color, category) if x) or _s(it.get("search_query"))
        names.append(name[:1].upper() + name[1:])
    base = list(names)
    for i, n in enumerate(base):
        if base.count(n) > 1 and _s(items[i].get("description")):
            names[i] = f"{n} — {_s(items[i].get('description'))}"
    return names


def format_items(items: list[dict]) -> str:
    lines = [_t("👕 Items spotted:")]
    lines += [f"{i}. {n}" for i, n in enumerate(item_names(items), 1)]
    return "\n".join(lines)


def place_is_found(data: dict) -> bool:
    try:
        conf = float(data.get("confidence") or 0)
    except (TypeError, ValueError):
        conf = 0.0
    return bool(data.get("found")) and bool(_s(data.get("name"))) and conf >= 0.5


def place_where(data: dict) -> str:
    return ", ".join(x for x in (_s(data.get("city")), _s(data.get("country")))
                     if x and x.lower() not in _s(data.get("name")).lower())


def format_place(data: dict) -> str:
    lines = [_t("📍 Place found!"), _s(data.get("name"))]
    if place_where(data):
        lines.append(f"🏙 {place_where(data)}")
    if _s(data.get("description")):
        lines.append(_s(data.get("description")))
    try:
        conf = float(data.get("confidence") or 0)
    except (TypeError, ValueError):
        conf = 0.0
    lines.append(_t("🎯 Confidence: high") if conf >= 0.8 else _t("🎯 Confidence: medium"))
    return "\n".join(lines)


def place_maps_url(data: dict) -> str:
    if isinstance(data.get("_lat"), float) and isinstance(data.get("_lng"), float):
        return f"https://www.google.com/maps/search/?api=1&query={data['_lat']},{data['_lng']}"
    q = quote_plus(", ".join(x for x in (_s(data.get("name")), place_where(data)) if x))
    return f"https://www.google.com/maps/search/?api=1&query={q}"


def place_card(data: dict) -> dict:
    q = quote_plus(", ".join(x for x in (_s(data.get("name")), place_where(data)) if x))
    return make_card(f"📍 {_s(data.get('name'))}", place_where(data), [
        url_button("🗺 Google Maps", place_maps_url(data)),
        url_button(_t("🔎 Photos & info"), f"https://www.google.com/search?q={q}"),
        url_button("🧳 Tripadvisor", f"https://www.tripadvisor.com/Search?q={q}"),
    ])


def shopping_url(item: dict) -> str:
    return f"https://www.google.com/search?tbm=shop&q={quote_plus(_s(item.get('search_query')))}"


def item_cards(items: list[dict], photos: list[str] | None = None) -> list[dict]:
    """photos[i] = public URL of the item's crop → card picture + 'Search by photo' (Google Lens finds
    the exact piece, prints included, far better than any text query)."""
    amazon = {"es": "amazon.es", "pt": "amazon.com.br", "de": "amazon.de", "fr": "amazon.fr",
              "it": "amazon.it"}.get(CURRENT_LANG.get(), "amazon.com")
    cards = []
    for i, (it, name) in enumerate(zip(items, item_names(items), strict=True), 1):
        q = _s(it.get("search_query"))
        photo = (photos or [""] * len(items))[i - 1]
        buttons = [url_button("🛍 Google Shopping", shopping_url(it)),
                   url_button("📦 Amazon", f"https://www.{amazon}/s?k={quote_plus(q)}")]
        if photo:
            buttons.insert(0, url_button(_t("🔍 Search by photo"),
                                         f"https://lens.google.com/uploadbyurl?url={quote_plus(photo)}"))
        cards.append(make_card(f"{i}. {name}", f"🔎 {q}", buttons, photo))
    return cards


# =============================================================================
# Search orchestration
# =============================================================================


@dataclass
class BranchResult:
    found: bool = False
    text: str = ""
    failed: bool = False   # upstream API failure (do not charge for it)
    skipped: bool = False  # branch could not run (e.g. photo has no audio)
    cards: list[dict] = field(default_factory=list)  # tappable link buttons
    finds: list[tuple[str, str, str, str]] = field(default_factory=list)  # (kind, title, subtitle, url) → lists
    summary: str = ""  # one short line, used for public comment replies


@dataclass
class SearchOutcome:
    branches: dict[str, BranchResult] = field(default_factory=dict)

    @property
    def any_found(self) -> bool:
        return any(b.found for b in self.branches.values())

    @property
    def any_failed(self) -> bool:
        return any(b.failed for b in self.branches.values())

    @property
    def partial(self) -> bool:
        return any(b.failed or b.skipped for b in self.branches.values())


# =============================================================================
# The bot
# =============================================================================


class Bot:
    def __init__(self, settings: Settings, http: httpx.AsyncClient, db: Database, ig: InstagramClient,
                 audd: AudDClient, vision: VisionClient, shazam: ShazamClient | None = None):
        self.s = settings
        self.shazam = shazam
        self.http = http
        self.db = db
        self.ig = ig
        self.audd = audd
        self.vision = vision
        self.dedup = Deduplicator(maxsize=1000)
        self.locks = UserLocks()
        self.jobs = asyncio.Semaphore(settings.max_parallel_jobs)
        self.seen_accounts: set[str] = set()
        self.admin_fails: dict[str, int] = {}
        self.stripe: StripeClient | None = (StripeClient(http, settings.stripe_secret_key)
                                            if settings.stripe_secret_key and settings.stripe_prices else None)
        self.langs: TTLCache = TTLCache(maxsize=20_000, ttl=3600)
        self.mention_times: deque[float] = deque()
        self.photos: TTLCache = TTLCache(maxsize=600, ttl=24 * 3600)  # item crops for "Search by photo"
        self.mention_cache: TTLCache = TTLCache(maxsize=2000, ttl=6 * 3600)
        self.redeem_fails: dict[str, int] = {}
        self.reels = ReelExtractor(settings.ig_cookies, settings.download_proxy, settings.tmp_root)

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def localized_replies(quick_replies: list[dict] | None) -> list[dict] | None:
        if not quick_replies or CURRENT_LANG.get() == "en":
            return quick_replies
        return [{**q, "title": _t(q["title"])} for q in quick_replies]

    async def safe_send(self, igsid: str, text: str, quick_replies: list[dict] | None = None) -> bool:
        try:
            await self.ig.send_text(igsid, _t(text), self.localized_replies(quick_replies))
            return True
        except GraphAPIError as exc:
            log.error("send_text to %s failed: %s", igsid, exc)
            return False

    async def typing(self, igsid: str) -> None:
        try:
            await self.ig.send_action(igsid, "typing_on")
        except GraphAPIError as exc:
            log.debug("typing_on failed: %s", exc)

    @staticmethod
    def is_subscribed(user: asyncpg.Record) -> bool:
        sub = user["sub_until"]
        return bool(sub and sub > datetime.now(timezone.utc))

    def balance_line(self, user: asyncpg.Record, credits: int | None = None) -> str:
        if self.is_subscribed(user):
            return _t("⭐ Unlimited until {date}", date=f"{user['sub_until']:%b %d, %Y}")
        n = user["credits"] if credits is None else credits
        return _t("💳 Credits left: {n}", n=n)

    def checkout_link(self, base: str, igsid: str) -> str:
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}{urlencode({'checkout[custom][igsid]': igsid})}"

    def pay_url(self, token: str, plan: str) -> str:
        return f"{self.s.public_url}/pay/{token}/{plan}"

    async def send_plans(self, igsid: str, out_of_credits: bool = False) -> None:
        """Plan cards with tappable 'Buy' buttons. Each link carries the user's secret pay token,
        so whatever is bought is credited to exactly this Instagram account."""
        assert self.stripe
        user = await self.db.get_or_create_user(igsid)
        token = await self.db.pay_token(igsid)
        cards: list[dict] = []
        if self.is_subscribed(user):
            cards.append(make_card(_t("⭐ Your Unlimited plan"),
                                   _t("Active until {date} • change or cancel", date=f"{user['sub_until']:%b %d, %Y}"),
                                   [url_button(_t("⚙️ Manage"), self.pay_url(token, "manage"))]))
        for plan in ("s", "l", "m", "y"):
            pid = self.s.stripe_prices.get(plan)
            if not pid or (plan in ("m", "y") and self.is_subscribed(user)):
                continue
            price = self.stripe.price_labels.get(pid, "")
            if plan in ("s", "l"):
                title, sub = _t("⚡ {n} searches", n=self.s.pack_credits[plan]), _t("One-time purchase • credits never expire")
                btn = _t("Buy · {price}", price=price) if price else _t("Buy · {price}", price="").rstrip(" ·")
            else:
                title = _t("♾ Unlimited · 1 month") if plan == "m" else _t("♾ Unlimited · 1 year")
                sub = _t("Search as much as you like • cancel anytime") if plan == "m" else _t("Best value • cancel anytime")
                btn = _t("Subscribe · {price}", price=price) if price else _t("Subscribe · {price}", price="").rstrip(" ·")
            cards.append(make_card(title, sub, [url_button(btn, self.pay_url(token, plan))], self.s.paywall_image_url))
        intro = _t("💳 Choose a plan 👇 You'll pay on a secure Stripe page — the purchase is linked to your "
                   "Instagram account automatically.")
        if out_of_credits:
            intro = _t("🔋 You're out of credits.") + "\n" + intro
        await self.safe_send(igsid, intro)
        try:
            await self.ig.send_generic_template(igsid, cards[:10])
        except GraphAPIError as exc:
            log.warning("Plan cards failed (%s) → links as text", exc)
            await self.safe_send(igsid, cards_as_text(cards))

    async def send_manage(self, igsid: str) -> None:
        user = await self.db.get_or_create_user(igsid)
        if self.stripe and user["stripe_customer"]:
            token = await self.db.pay_token(igsid)
            title = _t("⭐ Your Unlimited plan") if self.is_subscribed(user) else "Stripe"
            sub = (_t("Active until {date} • change or cancel", date=f"{user['sub_until']:%b %d, %Y}")
                   if self.is_subscribed(user) else "")
            await self.ig.send_generic_template(igsid, [make_card(title, sub, [
                url_button(_t("⚙️ Manage"), self.pay_url(token, "manage"))])])
        elif self.stripe:
            await self.send_plans(igsid)
        else:
            await self.safe_send(igsid, self.balance_line(user))

    async def checkout_url(self, user: asyncpg.Record, plan: str) -> str:
        """Creates a Stripe Checkout Session for this user and plan → its URL."""
        assert self.stripe
        igsid, token = user["igsid"], user["pay_token"]
        CURRENT_LANG.set(user["lang"] or "en")
        if plan == "manage":
            if not user["stripe_customer"]:
                return ""
            return await self.stripe.portal(user["stripe_customer"], f"{self.s.public_url}/paid?t={token}&ok=manage")
        price = self.s.stripe_prices.get(plan)
        if not price:
            return ""
        subscription = plan in ("m", "y")
        if subscription and self.is_subscribed(user) and user["stripe_customer"]:
            return await self.checkout_url(user, "manage")  # already subscribed → manage instead of a 2nd sub
        credits = str(self.s.pack_credits.get(plan, 0))
        form = [
            ("mode", "subscription" if subscription else "payment"),
            ("line_items[0][price]", price), ("line_items[0][quantity]", "1"),
            ("success_url", f"{self.s.public_url}/paid?t={token}&ok=1"),
            ("cancel_url", f"{self.s.public_url}/paid?t={token}&ok=0"),
            ("client_reference_id", igsid),
            ("metadata[igsid]", igsid), ("metadata[plan]", plan), ("metadata[credits]", credits),
            ("allow_promotion_codes", "true"),
            ("locale", {"pt": "pt-BR", "uk": "auto"}.get(user["lang"] or "", user["lang"] or "auto")),
        ]
        if user["stripe_customer"]:
            form.append(("customer", user["stripe_customer"]))
        elif not subscription:
            form.append(("customer_creation", "always"))
        if subscription:
            form += [("subscription_data[metadata][igsid]", igsid), ("subscription_data[metadata][plan]", plan)]
        else:
            form += [("payment_intent_data[metadata][igsid]", igsid),
                     ("payment_intent_data[metadata][plan]", plan),
                     ("payment_intent_data[metadata][credits]", credits)]
        if self.s.stripe_automatic_tax:
            form.append(("automatic_tax[enabled]", "true"))
        return await self.stripe.create_checkout(form)

    async def handle_stripe_event(self, event: dict) -> str:
        """Applies one Stripe webhook event. Idempotent (payment_events table). Raises → Stripe retries."""
        assert self.stripe
        etype, obj = _s(event.get("type")), (event.get("data") or {}).get("object") or {}
        meta = obj.get("metadata") or {}

        if etype in ("checkout.session.completed", "checkout.session.async_payment_succeeded"):
            if _s(obj.get("payment_status")) not in ("paid", "no_payment_required"):
                return "not paid yet"
            igsid = _s(meta.get("igsid")) or _s(obj.get("client_reference_id"))
            if not igsid:
                return "no igsid"
            if _s(obj.get("customer")):
                await self.db.set_stripe_customer(igsid, _s(obj.get("customer")))
            key = f"stripe:{obj.get('id')}"
            if _s(obj.get("mode")) == "payment":
                credits = int(_s(meta.get("credits")) or 0)
                if credits <= 0:
                    return "no credits in metadata"
                balance = await self.db.apply_credit_pack(key, igsid, etype, credits)
                if balance is None:
                    return "duplicate"
                log.info("Stripe: +%d credits for %s (balance %s)", credits, igsid, balance)
                asyncio.create_task(_run_safely(self.notify_payment, igsid, MSG_PAYMENT_CREDITS,
                                                {"n": credits, "balance": balance}))
                return "credits added"
            sub = await self.stripe.subscription(_s(obj.get("subscription")))
            until = subscription_period_end(sub)
            if not until:
                raise StripeError(f"subscription {obj.get('subscription')} has no period end")
            result = await self.db.set_subscription_until(key, igsid, etype, until)
            if result is None:
                return "duplicate"
            log.info("Stripe: Unlimited for %s until %s", igsid, result.isoformat())
            asyncio.create_task(_run_safely(self.notify_payment, igsid, MSG_PAYMENT_SUB,
                                            {"date": f"{result:%b %d, %Y}"}))
            return "subscription started"

        if etype == "invoice.paid":  # renewals (and the first invoice — harmless, same period end)
            sub_id = _s(obj.get("subscription")) or _s(
                (((obj.get("parent") or {}).get("subscription_details")) or {}).get("subscription"))
            if not sub_id:
                return "not a subscription invoice"
            sub = await self.stripe.subscription(sub_id)
            igsid, until = _s((sub.get("metadata") or {}).get("igsid")), subscription_period_end(sub)
            if not igsid or not until:
                return "no igsid / period"
            result = await self.db.set_subscription_until(f"stripe:{obj.get('id')}", igsid, etype, until)
            if result:
                log.info("Stripe: Unlimited for %s extended to %s", igsid, result.isoformat())
            return "renewed" if result else "duplicate"

        if etype == "customer.subscription.deleted":
            igsid = _s(meta.get("igsid"))
            if igsid:
                ended = obj.get("ended_at") or obj.get("canceled_at") or time.time()
                await self.db.end_subscription(igsid, datetime.fromtimestamp(int(ended), timezone.utc))
                log.info("Stripe: subscription of %s ended", igsid)
            return "subscription ended"

        if etype == "charge.refunded" and obj.get("refunded"):  # full refund of a credit pack
            igsid, credits = _s(meta.get("igsid")), int(_s(meta.get("credits")) or 0)
            if igsid and credits:
                left = await self.db.remove_credits(f"stripe:refund:{obj.get('id')}", igsid, etype, credits)
                log.info("Stripe: refund → -%d credits for %s (now %s)", credits, igsid, left)
            return "refund processed"

        return "ignored"

    async def send_paywall(self, igsid: str, user: asyncpg.Record, cost: int) -> None:
        if self.stripe and self.s.public_url:
            await self.send_plans(igsid, out_of_credits=True)
            await self.send_invite(igsid)
            return
        if not self.s.lemon_checkout_url:
            await self.safe_send(igsid, MSG_PAYMENTS_SOON)
            await self.send_invite(igsid)
            return
        credits_url = self.checkout_link(self.s.lemon_checkout_url, igsid)
        buttons = [{"type": "web_url", "url": credits_url, "title": f"Get {self.s.credits_per_pack} credits"}]
        if self.s.lemon_subscription_checkout_url:
            buttons.append({
                "type": "web_url",
                "url": self.checkout_link(self.s.lemon_subscription_checkout_url, igsid),
                "title": "Go Unlimited ⭐",
            })
        element: dict[str, Any] = {
            "title": "You're out of credits 🔋",
            # Generic Template subtitle is limited to 80 characters.
            "subtitle": (f"Needs {cost} credit{'s' if cost > 1 else ''}, you have {user['credits']}. "
                         "Top up to keep finding songs & movies.")[:80],
            "buttons": buttons,
        }
        if self.s.paywall_image_url:
            element["image_url"] = self.s.paywall_image_url
        try:
            await self.ig.send_generic_template(igsid, [element])
        except GraphAPIError as exc:
            log.warning("Generic template failed (%s); falling back to text", exc)
            await self.safe_send(
                igsid,
                f"You're out of credits 🔋 This search needs {cost}, you have {user['credits']}.\n"
                f"Get {self.s.credits_per_pack} credits here: {credits_url}",
            )

    async def deliver_pending_notice(self, igsid: str) -> None:
        try:
            notice = await self.db.pop_pending_notice(igsid)
        except Exception as exc:  # noqa: BLE001
            log.error("pop_pending_notice failed: %s", exc)
            return
        if notice and not await self.safe_send(igsid, notice):
            await self.db.set_pending_notice(igsid, notice)

    # ---------------------------------------------------------------- webhook routing

    async def handle_messaging_event(self, ev: dict, own_id: str) -> None:
        sender = _s((ev.get("sender") or {}).get("id"))
        if not sender or sender == own_id:
            return
        msg = ev.get("message")
        postback = ev.get("postback")
        if msg and (msg.get("is_echo") or msg.get("is_deleted") or msg.get("is_unsupported")):
            return
        lang = await self.user_lang(sender)
        CURRENT_LANG.set(lang or "en")

        if msg:
            await self.deliver_pending_notice(sender)

            qr_payload = _s((msg.get("quick_reply") or {}).get("payload"))
            if qr_payload.startswith("LANG_"):
                await self.choose_language(sender, qr_payload[5:])
                return
            if await self.handle_menu_payload(sender, qr_payload):
                return
            if qr_payload in ACTIONS:
                await self.handle_action(sender, qr_payload)
                return

            attachments = msg.get("attachments") or []
            if attachments:
                media = self.pick_media(attachments)
                if media is None:
                    await self.safe_send(sender, MSG_NO_MEDIA)
                elif media == ("", "restricted"):
                    await self.safe_send(sender, MSG_MEDIA_RESTRICTED)
                else:
                    await self.handle_media(sender, *media, mid=_s(msg.get("mid")))
                return

            await self.handle_text(sender, _s(msg.get("text")))
            return

        if postback:
            await self.deliver_pending_notice(sender)
            payload = _s(postback.get("payload"))
            if await self.handle_menu_payload(sender, payload):
                return
            if payload in ACTIONS:
                await self.handle_action(sender, payload)
            else:
                await self.handle_text(sender, payload)

    async def handle_menu_payload(self, igsid: str, payload: str) -> bool:
        if payload == MY_LISTS:
            await self.show_lists(igsid)
        elif payload.startswith("LIST_"):
            await self.show_list(igsid, payload[5:])
        elif payload == INVITE:
            await self.send_invite(igsid)
        else:
            return False
        return True

    async def user_lang(self, igsid: str) -> str | None:
        """User's chosen language ('' cached = not chosen yet)."""
        if igsid not in self.langs:
            try:
                self.langs[igsid] = await self.db.get_lang(igsid) or ""
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not read language for %s: %s", igsid, exc)
                return None
        return self.langs[igsid] or None

    def needs_language(self, igsid: str) -> bool:
        return self.s.ask_language and not self.langs.get(igsid)

    async def send_language_picker(self, igsid: str) -> None:
        await self.safe_send(igsid, MSG_CHOOSE_LANG, lang_picker())

    async def choose_language(self, igsid: str, code: str) -> None:
        if code not in LANGUAGES:
            await self.send_language_picker(igsid)
            return
        await self.db.set_lang(igsid, code)
        self.langs[igsid] = code
        CURRENT_LANG.set(code)
        log.info("Language for %s set to %s", igsid, code)
        user = await self.db.get_or_create_user(igsid)
        if user["last_media_url"]:  # they sent a Reel before choosing → continue with it
            await self.safe_send(igsid, MSG_LANG_SET)
            await self.safe_send(igsid, MSG_MENU, QUICK_REPLIES)
        else:
            await self.safe_send(igsid, _t(MSG_LANG_SET) + "\n\n" + welcome_text(self.s.bot_name, self.balance_line(user)))

    @staticmethod
    def pick_media(attachments: list[dict]) -> tuple[str, str] | None:
        """Returns (url, type_hint), ("", "restricted") or None if nothing usable."""
        type_hint = {
            "ig_reel": "video", "reel": "video", "video": "video",
            "image": "image", "animated_image": "video", "audio": "audio",
            "ig_post": "unknown", "share": "unknown", "story_mention": "unknown",
        }
        saw_media_type = False
        for att in attachments:
            a_type = _s(att.get("type")).lower()
            if a_type not in type_hint:
                continue
            saw_media_type = True
            url = _s((att.get("payload") or {}).get("url"))
            if url.startswith("http"):
                log.info("Attachment type=%s keys=%s", a_type, sorted((att.get("payload") or {}).keys()))
                return url, type_hint[a_type]
        return ("", "restricted") if saw_media_type else None

    async def handle_media(self, igsid: str, url: str, media_type: str, mid: str = "") -> None:
        log.info("Media received: type=%s url=%s", media_type, describe_url(url))
        ig_url = instagram_post_url(url)
        if ig_url:
            self.reels.prefetch(ig_url)
        if self.needs_language(igsid):  # first contact: language first, the Reel is kept
            await self.db.get_or_create_user(igsid)
            results = await asyncio.gather(
                self.db.set_last_media(igsid, url, media_type, mid),
                self.ig.send_text(igsid, MSG_CHOOSE_LANG, lang_picker()),
                return_exceptions=True,
            )
        else:
            results = await asyncio.gather(
                self.db.set_last_media(igsid, url, media_type, mid),
                self.ig.send_text(igsid, _t(MSG_MENU), self.localized_replies(QUICK_REPLIES)),
                return_exceptions=True,
            )
        for r in results:
            if isinstance(r, Exception):
                log.error("handle_media error for %s: %s", igsid, r)
        if isinstance(results[0], Exception):
            await self.safe_send(igsid, "Oops, I couldn't save that one 😕 Please send it again in a moment.")

    async def handle_text(self, igsid: str, text: str) -> None:
        clean = text.strip()
        lowered = clean.lower().strip(" !.?")

        url_match = URL_RE.search(clean)
        if url_match:
            url = url_match.group(0).rstrip(").,!?")
            host = (urlparse(url).hostname or "").lower()
            if instagram_post_url(url):
                await self.handle_media(igsid, url, "video")
                return
            if any(d in host for d in UNSUPPORTED_LINK_DOMAINS):
                await self.safe_send(igsid, MSG_LINK_UNSUPPORTED)
                return
            await self.handle_media(igsid, url, "unknown")
            return

        if lowered in LANGUAGE_COMMANDS:
            await self.send_language_picker(igsid)
            return
        if lowered in LIST_COMMANDS:
            await self.show_lists(igsid)
            return
        if lowered in INVITE_COMMANDS:
            await self.send_invite(igsid)
            return

        if lowered in TEXT_COMMANDS:
            await self.handle_action(igsid, TEXT_COMMANDS[lowered])
            return

        if lowered in DELETE_COMMANDS:
            await self.db.delete_user(igsid)
            self.langs.pop(igsid, None)
            log.info("User data deleted on request")
            await self.safe_send(igsid, MSG_DATA_DELETED)
            return

        user = await self.db.get_or_create_user(igsid)
        if lowered.startswith("admin ") and await self.handle_admin(igsid, clean):
            return
        if lowered in BALANCE_COMMANDS:
            await self.safe_send(igsid, self.balance_line(user))
            return
        if lowered in BUY_COMMANDS:
            if self.stripe and self.s.public_url:
                await self.send_plans(igsid)
            else:
                await self.send_paywall(igsid, user, 1)
            return
        if lowered in MANAGE_COMMANDS:
            await self.send_manage(igsid)
            return
        if await self.try_redeem(igsid, clean):
            return
        if self.needs_language(igsid):
            await self.send_language_picker(igsid)
            return
        await self.safe_send(igsid, welcome_text(self.s.bot_name, self.balance_line(user)))

    # ---------------------------------------------------------------- invites

    def bot_link(self) -> str:
        h = self.s.instagram_handle
        return f"https://ig.me/m/{h}" if h else "this account in Instagram DM"

    async def send_invite(self, igsid: str) -> None:
        if self.s.referral_credits <= 0:
            return
        code = await self.db.ref_code(igsid)
        n = self.s.referral_credits
        await self.safe_send(igsid, _t("🎁 Invite friends and get +{n} free searches for each friend who joins — "
                                       "your friend gets +{n} too!\nJust forward them the next message 👇", n=n))
        await self.safe_send(igsid, _t("🔥 This bot finds the song, movie, outfit & place from any Instagram Reel "
                                       "— right in DM!\n\n1️⃣ Open {link}\n2️⃣ Send it this code: {code}\n\n"
                                       "We both get +{n} free searches 🎁", link=self.bot_link(), code=code, n=n))

    async def redeem_referral(self, igsid: str, code: str) -> bool:
        if self.s.referral_credits <= 0:
            return False
        status, referrer, mine, theirs = await self.db.redeem_referral(
            igsid, code, self.s.referral_credits, self.s.referral_daily_cap)
        if status == "unknown":
            self.redeem_fails[igsid] = self.redeem_fails.get(igsid, 0) + 1
            return False
        log.info("Referral %s by %s → %s (referrer %s)", code, igsid, status, referrer)
        n = self.s.referral_credits
        msg = {
            "ok": _t("🎁 Invite code accepted: +{credits} credits!\n💳 Credits left: {balance}\n\nSend me a Reel 👇",
                     credits=n, balance=mine),
            "self": _t("That's your own invite code 😄 Share it with friends!"),
            "used": _t("You've already used an invite code 🙂"),
            "old": _t("Invite codes are only for new users 🙂"),
            "cap": _t("This invite code has reached today's limit — try again tomorrow 🙂"),
        }[status]
        await self.safe_send(igsid, msg)
        if status == "ok":
            lang = CURRENT_LANG.get()
            await self.notify_payment(referrer, "🎉 A friend joined with your invite code! +{credits} credits\n"
                                                "💳 Credits left: {balance}", {"credits": n, "balance": theirs})
            CURRENT_LANG.set(lang)
        return True

    # ---------------------------------------------------------------- my lists

    LIST_LABELS = {"song": "🎵 Songs ({n})", "movie": "🎬 Movies ({n})", "item": "👕 Items ({n})",
                   "place": "📍 Places ({n})"}
    LIST_HEADERS = {"song": "🎵 Your songs ({n}):", "movie": "🎬 Your movies & shows ({n}):",
                    "item": "👕 Your items ({n}):", "place": "📍 Your places ({n}):"}

    async def show_lists(self, igsid: str) -> None:
        counts = await self.db.count_finds(igsid)
        replies = [{"content_type": "text", "title": _t(self.LIST_LABELS[k], n=counts[k]), "payload": f"LIST_{k}"}
                   for k in LIST_KINDS if counts.get(k)]
        if not replies:
            await self.safe_send(igsid, "📚 Your lists are empty for now — every song, movie, item and place I find "
                                        "for you is saved here automatically.")
            return
        await self.ig.send_text(igsid, _t("📚 Your lists — what should I show?"), replies)

    async def show_list(self, igsid: str, kind: str) -> None:
        if kind not in LIST_KINDS:
            await self.show_lists(igsid)
            return
        rows = await self.db.list_finds(igsid, kind, 30)
        counts = await self.db.count_finds(igsid)
        lines = [_t(self.LIST_HEADERS[kind], n=counts.get(kind, 0))]
        for i, r in enumerate(rows, 1):
            lines.append(f"{i}. {r['title']}" + (f" — {r['subtitle']}" if r["subtitle"] and kind != "song" else ""))
        cards = [make_card(r["title"], r["subtitle"], [url_button(_t("🔗 Open"), r["url"])])
                 for r in rows[:10] if r["url"].startswith("https://")]
        others = [{"content_type": "text", "title": _t(self.LIST_LABELS[k], n=counts[k]), "payload": f"LIST_{k}"}
                  for k in LIST_KINDS if k != kind and counts.get(k)]
        await self.safe_send(igsid, "\n".join(lines), None if cards else others or None)
        if cards:
            try:
                await self.ig.send_generic_template(igsid, cards)
            except GraphAPIError as exc:
                log.warning("List cards failed: %s", exc)
            if others:
                await self.ig.send_text(igsid, _t("📚 Your lists — what should I show?"), others)

    async def handle_admin(self, igsid: str, text: str) -> bool:
        """Secret DM commands (wrong code = ordinary text; 5 wrong tries lock the sender out):
          admin <code> 500                     → +500 credits to yourself (max 10000)
          admin <code> gift 50                 → one-time code worth 50 credits, to send to a friend
          admin <code> promo SUMMER 10 100 7   → code SUMMER: 10 credits, 100 people, valid 7 days
          admin <code> codes                   → list active codes
          admin <code> off SUMMER              → disable a code"""
        code = self.s.admin_code
        if len(code) < 12 or self.admin_fails.get(igsid, 0) >= 5:
            return False
        parts = text.split()
        if len(parts) < 2 or not hmac.compare_digest(parts[1].encode(), code.encode()):
            self.admin_fails[igsid] = self.admin_fails.get(igsid, 0) + 1
            log.warning("Wrong admin code from %s (%d/5)", igsid, self.admin_fails[igsid])
            return False
        args = parts[2:]
        sub = args[0].lower() if args else ""

        def num(i: int, default: int, hi: int) -> int:
            return max(1, min(int(args[i]), hi)) if len(args) > i and args[i].isdigit() else default

        if sub == "gift":
            credits = num(1, 10, 10_000)
            gift = "GIFT-" + secrets.token_hex(4).upper()
            await self.db.create_promo(gift, credits, 1, 30)
            log.info("Admin: gift code %s for %d credits", gift, credits)
            await self.safe_send(igsid, f"🎁 Gift code for {credits} credits (1 person, 30 days):")
            await self.safe_send(igsid, gift)  # separate message = easy to copy & forward
            await self.safe_send(igsid, "Your friend just sends this code to me in DM.")
        elif sub == "promo":
            name = args[1].upper() if len(args) > 1 else ""
            if not PROMO_RE.fullmatch(name):
                await self.safe_send(igsid, "Promo name: 3–24 letters/digits, e.g. SUMMER")
                return True
            credits, uses = num(2, 5, 10_000), num(3, 100, 1_000_000)
            days = num(4, 0, 3650) if len(args) > 4 else None
            ok = await self.db.create_promo(name, credits, uses, days)
            await self.safe_send(igsid, (
                f"✅ Promo {name}: {credits} credits, up to {uses} people"
                + (f", {days} days" if days else ", no expiry")
                + "\nPeople just send the code to me in DM.") if ok else f"⚠️ Code {name} already exists")
        elif sub == "codes":
            rows = await self.db.list_promos()
            lines = [f"{r['code']}: {r['credits']} cr, used {r['used']}/{r['max_uses']}"
                     + (f", until {r['expires_at']:%b %d}" if r["expires_at"] else "") for r in rows]
            await self.safe_send(igsid, "🎟 Active codes:\n" + ("\n".join(lines) or "none"))
        elif sub == "off" and len(args) > 1:
            ok = await self.db.disable_promo(args[1].upper())
            await self.safe_send(igsid, f"⛔ {args[1].upper()} disabled" if ok else "Code not found")
        else:
            amount = num(0, 100, 10_000)
            balance = await self.db.add_credits(igsid, amount)
            log.info("Admin: +%d credits to %s (now %d)", amount, igsid, balance)
            await self.safe_send(igsid, f"🛠 Admin: +{amount} credits added.\n💳 Credits left: {balance}")
        return True

    async def try_redeem(self, igsid: str, text: str) -> bool:
        """A message that is just a promo/gift code → credits. Unknown code → False (normal reply)."""
        token = text.strip().upper()
        if token.startswith("REDEEM "):
            token = token[7:].strip()
        if not PROMO_RE.fullmatch(token) or self.redeem_fails.get(igsid, 0) >= 20:
            return False
        if token.startswith("REF-"):
            return await self.redeem_referral(igsid, token)
        status, credits, balance = await self.db.redeem_promo(igsid, token)
        if status == "unknown":
            if token.startswith("GIFT-"):  # only guessing of private gift codes is rate-limited
                self.redeem_fails[igsid] = self.redeem_fails.get(igsid, 0) + 1
            return False
        log.info("Promo %s → %s for %s", token, status, igsid)
        msg = {
            "ok": _t("🎁 Code applied: +{credits} credits!\n💳 Credits left: {balance}\n\nSend me a Reel to use them 👇",
                     credits=credits, balance=balance),
            "used": _t("You've already used this code 🙂"),
            "expired": _t("This code has expired or run out 😕"),
        }[status]
        await self.safe_send(igsid, msg)
        return True

    async def handle_mention(self, value: dict, own_id: str) -> None:
        """Someone wrote '@ourbot what song?' under any public Reel → we answer in the comments.
        Free for the commenter (it's advertising), limited to MENTIONS_PER_HOUR searches."""
        if not self.s.mentions_enabled:
            return
        comment_id, media_id = _s(value.get("comment_id")), _s(value.get("media_id"))
        if not media_id:
            return
        try:
            if comment_id:
                data = await self.ig.get_mentioned_comment(comment_id)
                text, media = _s(data.get("text")), data.get("media") or {}
                username = _s(data.get("username"))
            else:
                media = await self.ig.get_mentioned_media(media_id)
                text, username = _s(media.get("caption")), _s(media.get("username"))
        except (GraphAPIError, httpx.HTTPError) as exc:
            log.warning("Mention %s: could not read it: %s", comment_id or media_id, exc)
            return
        if username and username.lower() == self.s.instagram_handle.lower():
            return
        permalink = _s(media.get("permalink"))
        if not instagram_post_url(permalink):
            log.info("Mention %s: no usable permalink (%r)", comment_id or media_id, permalink)
            return
        action, lang = mention_intent(text)
        CURRENT_LANG.set(lang)
        key = (media_id, action, lang)
        reply = self.mention_cache.get(key)
        if reply is None:
            now = asyncio.get_running_loop().time()
            while self.mention_times and now - self.mention_times[0] > 3600:
                self.mention_times.popleft()
            if len(self.mention_times) >= self.s.mentions_per_hour:
                log.warning("Mention %s skipped: hourly limit (%d) reached", comment_id, self.s.mentions_per_hour)
                return
            self.mention_times.append(now)
            log.info("Mention %s by @%s: %s on %s", comment_id or media_id, username, action, describe_url(permalink))
            try:
                outcome = await self.run_search(action, permalink)
            except (MediaUnavailable, ExternalServiceError) as exc:
                log.info("Mention %s: search failed: %s", comment_id or media_id, exc)
                return
            summary = "\n".join(b.summary for b in outcome.branches.values() if b.found and b.summary)
            reply = (summary + "\n\n" + _t("👋 DM me any Reel — I find songs, movies, outfits & places!"))[:900] \
                if summary else ""
            self.mention_cache[key] = reply
        if not reply:
            log.info("Mention %s: nothing found → no reply", comment_id or media_id)
            return
        try:
            await self.ig.reply_to_mention(media_id, comment_id, reply)
            log.info("Mention %s answered", comment_id or media_id)
        except (GraphAPIError, httpx.HTTPError) as exc:
            log.warning("Mention %s: reply failed: %s", comment_id or media_id, exc)

    async def handle_comment(self, value: dict, own_id: str) -> None:
        text = _s(value.get("text"))
        from_id = _s((value.get("from") or {}).get("id"))
        comment_id = _s(value.get("id"))
        if not comment_id or not FIND_KEYWORD_RE.search(text) or from_id == own_id:
            return
        try:
            await self.ig.send_private_reply(comment_id, private_reply_text(self.s.free_credits))
            log.info("Private reply sent for comment %s", comment_id)
        except GraphAPIError as exc:
            log.warning("Private reply for comment %s failed: %s", comment_id, exc)

    # ---------------------------------------------------------------- core action

    async def handle_action(self, igsid: str, action: str) -> None:
        async with self.locks.hold(igsid):
            typing_task = asyncio.create_task(self.typing(igsid))
            try:
                user = await self.db.get_or_create_user(igsid)
                cost = ACTION_COST[action]
                subscribed = self.is_subscribed(user)

                if not user["last_media_url"]:
                    await self.safe_send(igsid, MSG_NO_MEDIA)
                    return
                if not subscribed and user["credits"] < cost:
                    await self.send_paywall(igsid, user, cost)
                    return
                if subscribed and self.s.sub_daily_limit > 0 and \
                        await self.db.sub_quota_used(igsid) + cost > self.s.sub_daily_limit:
                    await self.safe_send(igsid, _t("You've reached today's fair-use limit for Unlimited ({n} searches) "
                                                   "— it resets tomorrow 🙂", n=self.s.sub_daily_limit))
                    return

                try:
                    outcome = await self.run_search(action, user["last_media_url"], user["last_media_mid"] or "")
                except MediaUnavailable as exc:
                    log.info("Media unavailable for %s: %s", igsid, exc)
                    await self.safe_send(igsid, exc.user_message, followup_replies(action) if exc.user_message in
                                         (MSG_NO_AUDIO, MSG_PHOTO_NO_AUDIO, MSG_NO_VIDEO) else None)
                    return
                except ExternalServiceError as exc:
                    log.warning("External service failure for %s: %s", igsid, exc)
                    await self.safe_send(igsid, MSG_OVERLOAD, QUICK_REPLIES)
                    return

                if not outcome.any_found:
                    if outcome.any_failed:
                        await self.safe_send(igsid, MSG_OVERLOAD, QUICK_REPLIES)
                    else:
                        await self.safe_send(igsid, MSG_NOT_FOUND[action], followup_replies(action))
                    return

                # Charge ONLY after a successful, non-empty answer.
                # FIND_ALL where one half failed / couldn't run is charged as a single search.
                charge = cost if not outcome.partial else 1
                footer: str
                new_balance: int | None = None
                if subscribed:
                    footer = self.balance_line(user)
                    await self.db.add_sub_quota(igsid, charge)
                else:
                    new_balance = await self.db.charge(igsid, charge)
                    if new_balance is None:  # should not happen thanks to the per-user lock
                        log.error("Charge race for %s (cost %s)", igsid, charge)
                        new_balance = max(user["credits"] - charge, 0)
                    footer = self.balance_line(user, new_balance)

                try:
                    await self.db.save_finds(igsid, [f for b in outcome.branches.values() for f in b.finds])
                except Exception as exc:  # noqa: BLE001 — lists are a bonus, never break the answer
                    log.warning("Could not save finds for %s: %s", igsid, exc)

                blocks = [b.text for b in outcome.branches.values() if b.text]
                blocks.append(footer)
                cards = [c for b in outcome.branches.values() for c in b.cards][:10]
                more = followup_replies(action)
                if self.s.referral_credits <= 0:
                    more = [q for q in more if q["payload"] != INVITE]
                prompt = "📚 Saved to My lists" if action == FIND_ALL else "Find something else? 👇"
                if not cards:
                    blocks.append(_t(prompt))
                    await self.safe_send(igsid, "\n\n".join(blocks), more)
                else:
                    await self.safe_send(igsid, "\n\n".join(blocks))
                    try:
                        await self.ig.send_generic_template(igsid, cards)
                    except GraphAPIError as exc:
                        log.warning("Link buttons failed (%s) → sending links as text", exc)
                        await self.safe_send(igsid, cards_as_text(cards))
                    await self.safe_send(igsid, prompt, more)
                if new_balance == 0:  # just used the last free search → offer the invite bonus
                    await self.safe_send(igsid, "🔋 That was your last free search.")
                    await self.send_invite(igsid)
            except Exception:  # noqa: BLE001 — never leave the user hanging
                log.exception("Unexpected error while handling %s for %s", action, igsid)
                await self.safe_send(igsid, MSG_OVERLOAD)
            finally:
                if not typing_task.done():
                    typing_task.cancel()

    async def run_search(self, action: str, media_url: str, mid: str = "") -> SearchOutcome:
        async with self.jobs:
            workdir = Path(self.s.tmp_root) / uuid.uuid4().hex  # /tmp/{uuid}
            workdir.mkdir(parents=True, exist_ok=False)
            try:
                src = workdir / "source"
                ctype = await self.download_with_fallback(
                    media_url, mid, src, cover_ok=action in (FIND_MOVIE, FIND_ITEM, FIND_PLACE))
                info = await probe_media(src, ctype)
                log.info("Media: kind=%s duration=%.1fs audio=%s ctype=%s",
                         info.kind, info.duration, info.has_audio, ctype)

                tasks: dict[str, Awaitable[BranchResult]] = {}
                if action in (FIND_MUSIC, FIND_ALL):
                    tasks["music"] = self._music_branch(info, workdir, strict=action == FIND_MUSIC,
                                                        hint=self.reels.cached(media_url))
                if action in (FIND_MOVIE, FIND_ITEM, FIND_PLACE, FIND_ALL):
                    tasks["vision"] = self._vision_branch(action, info, workdir, strict=action != FIND_ALL,
                                                          hint=self.reels.cached(media_url))

                # Music (ffmpeg → AudD) and vision (ffmpeg → Gemini) run in parallel.
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                outcome = SearchOutcome()
                for name, res in zip(tasks.keys(), results, strict=True):
                    if isinstance(res, BaseException):
                        raise res  # strict-mode MediaUnavailable / ExternalServiceError
                    outcome.branches[name] = res

                if action == FIND_ALL and outcome.any_found:
                    music, vision = outcome.branches["music"], outcome.branches["vision"]
                    if not music.found and not music.text:
                        music.text = _t("🎵 Song: search temporarily unavailable" if music.failed
                                        else "🎵 Song: not recognised")
                    if not vision.found and not vision.text:
                        vision.text = _t("🎬 Movie, 👕 items & 📍 place: temporarily unavailable" if vision.failed
                                         else "🎬 Movie, 👕 items & 📍 place: nothing identifiable")
                return outcome
            finally:
                shutil.rmtree(workdir, ignore_errors=True)

    async def download_with_fallback(self, media_url: str, mid: str, dest: Path,
                                     cover_ok: bool = False) -> str:
        """Instagram post links go through the Reel downloader; anything else is fetched directly
        (and if that returns an HTML page, the Graph API is asked for other URLs)."""
        max_bytes = self.s.max_download_mb * 1024 * 1024
        ig_url = instagram_post_url(media_url)
        if ig_url:
            return await self.download_instagram_post(ig_url, dest, max_bytes, cover_ok)
        try:
            return await download_media(self.http, media_url, dest, max_bytes,
                                        allow_private=self.s.allow_private_urls)
        except MediaUnavailable as first_exc:
            if first_exc.user_message not in (MSG_NOT_MEDIA, MSG_EXPIRED) or not mid:
                raise
            try:
                alternatives = [u for u in await self.ig.get_message_media_urls(mid) if u != media_url]
            except Exception as exc:  # noqa: BLE001
                log.warning("Graph media lookup failed: %s", exc)
                raise first_exc from exc
            log.info("Trying %d alternative media URL(s): %s", len(alternatives),
                     [describe_url(u) for u in alternatives])
            for alt in alternatives:
                try:
                    return await download_media(self.http, alt, dest, max_bytes,
                                                allow_private=self.s.allow_private_urls)
                except MediaUnavailable as exc:
                    log.info("Alternative %s failed: %s", describe_url(alt), exc)
            raise first_exc

    async def download_instagram_post(self, ig_url: str, dest: Path, max_bytes: int, cover_ok: bool) -> str:
        allow = self.s.allow_private_urls
        post: ResolvedPost | None = None
        try:
            post = await self.reels.resolve(ig_url)
        except MediaUnavailable as exc:
            log.warning("Reel downloader: %s (%s)", exc, describe_url(ig_url))

        if post and post.video_url:
            try:
                return await download_media(self.http, post.video_url, dest, max_bytes, allow_private=allow)
            except MediaUnavailable as exc:
                if exc.user_message == MSG_TOO_BIG:
                    raise
                log.warning("Reel video download failed: %s", exc)
                self.reels.cache.pop(ig_url, None)
        elif post and (post.dash_video or post.dash_audio):
            try:
                return await self._download_dash(post, dest, max_bytes)
            except MediaUnavailable as exc:
                if exc.user_message == MSG_TOO_BIG:
                    raise
                log.warning("Reel DASH download failed: %s", exc)
                self.reels.cache.pop(ig_url, None)

        if cover_ok:  # movie / outfit search can work from the cover frame alone
            cover = (post.thumbnail if post else "") or await fetch_og_image(self.http, ig_url)
            if cover:
                try:
                    ctype = await download_media(self.http, cover, dest, max_bytes, allow_private=allow)
                    log.info("Using the Reel cover image instead of the video")
                    return ctype
                except MediaUnavailable as exc:
                    log.warning("Reel cover download failed: %s", exc)
        raise MediaUnavailable(MSG_REEL_BLOCKED, f"could not fetch {describe_url(ig_url)}")

    async def _download_dash(self, post: ResolvedPost, dest: Path, max_bytes: int) -> str:
        """Separate video + audio tracks → one mp4 (stream copy, no re-encode)."""
        allow = self.s.allow_private_urls
        parts: list[tuple[str, Path]] = []
        if post.dash_video:
            parts.append((post.dash_video, dest.with_name("dash_v")))
        if post.dash_audio:
            parts.append((post.dash_audio, dest.with_name("dash_a")))
        await asyncio.gather(*(download_media(self.http, u, p, max_bytes, allow_private=allow)
                               for u, p in parts))
        if len(parts) == 1:
            parts[0][1].replace(dest)
            return "video/mp4" if post.dash_video else "audio/mp4"
        out = dest.with_name("merged.mp4")
        rc, _out, err = await run_cmd(
            ["ffmpeg", "-v", "error", "-y", "-i", str(parts[0][1]), "-i", str(parts[1][1]),
             "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-f", "mp4", str(out)], timeout=15)
        if rc != 0:
            raise MediaUnavailable(MSG_DOWNLOAD_FAILED, f"dash merge failed: {err[:200]!r}")
        out.replace(dest)
        return "video/mp4"

    # ---------------------------------------------------------------- inbox polling

    def mark_seen_event(self, sender: str, timestamp_ms: int) -> None:
        """Second dedup key (sender + second) in case webhook mids differ from Graph message ids."""
        if sender and timestamp_ms:
            self.dedup.seen(f"t:{sender}:{timestamp_ms // 1000}")

    async def poll_inbox_once(self, not_before: datetime) -> None:
        own_id = self.s.ig_account_id
        now = datetime.now(timezone.utc)
        for message in await self.ig.list_recent_messages():
            created = parse_graph_time(_s(message.get("created_time")))
            if not created or created < not_before or now - created > timedelta(minutes=15):
                continue
            if now - created < timedelta(seconds=4):
                continue  # give the webhook a moment to arrive first
            sender = _s((message.get("from") or {}).get("id"))
            if not sender or sender == own_id:
                continue
            mid = _s(message.get("id"))
            second = int(created.timestamp())
            keys = [f"t:{sender}:{second + d}" for d in (-1, 0, 1)]
            if f"m:{mid}" in self.dedup._cache or any(k in self.dedup._cache for k in keys):
                continue
            self.dedup.seen(f"m:{mid}")
            self.dedup.seen(keys[1])
            ev = graph_message_to_event(message, own_id)
            if ev is None:
                continue
            log.info("Inbox poll picked up a message the webhook did not deliver (mid=%s…)", mid[:12])
            asyncio.create_task(_run_safely(self.handle_messaging_event, ev, own_id))

    async def inbox_poll_loop(self) -> None:
        if self.s.inbox_poll_sec <= 0 or not self.s.ig_account_id.isdigit():
            return
        not_before = datetime.now(timezone.utc) - timedelta(minutes=2)
        backoff = 0
        log.info("Inbox polling every %ss (catches DMs stuck in Message requests)", self.s.inbox_poll_sec)
        while True:
            try:
                await self.poll_inbox_once(not_before)
                backoff = 0
            except GraphAPIError as exc:
                if exc.code in (4, 17, 32, 613, 80002):
                    # Meta's per-account rate limit: back off hard, so replies to users keep working
                    backoff = min(max(backoff * 2, 600), 3600)
                    log.warning("Inbox poll paused for %d min: Meta rate limit for this Instagram account "
                                "(code %s). Replies are not affected by the pause.", backoff // 60, exc.code)
                else:
                    backoff = min(backoff * 2 or 60, 900)
                    log.warning("Inbox poll failed (%s); next try in %ss", exc, self.s.inbox_poll_sec + backoff)
            except httpx.HTTPError as exc:
                backoff = min(backoff * 2 or 60, 900)
                log.warning("Inbox poll failed (%s); next try in %ss", exc, self.s.inbox_poll_sec + backoff)
            except Exception:  # noqa: BLE001
                log.exception("Inbox poll crashed")
            await asyncio.sleep(self.s.inbox_poll_sec + backoff)

    async def _shazam_once(self, info: MediaInfo, workdir: Path, rate: float = 1.0,
                           start: float | None = None, clean: bool = False) -> dict | None:
        if not self.shazam:
            return None
        try:
            wav = await extract_wav_clip(info, workdir, rate, start=start, clean=clean)
        except MediaUnavailable as exc:
            log.info("Shazam clip skipped: %s", exc)
            return None
        return await self.shazam.recognize(wav)

    async def _ai_song(self, info: MediaInfo, workdir: Path, caption: str, label: str) -> dict | None:
        """Gemini listens to the clip (+ caption / audio label) → guess → verified in iTunes catalog."""
        try:
            wav, frame = await asyncio.gather(
                extract_wav_clip(info, workdir, seconds=25),  # longer clip = melody is easier to recognise
                _grab_frame(info.path, info.duration * 0.5, workdir / "song_frame.jpg")
                if info.kind == "video" else asyncio.sleep(0, result=None))
            data = await self.vision.identify_song(wav, caption, label, [frame] if frame else None)
        except (MediaUnavailable, ExternalServiceError) as exc:
            log.warning("AI song search failed: %s", exc)
            return None
        artist, title = _s(data.get("artist")), _s(data.get("title"))
        try:
            conf = float(data.get("confidence") or 0)
        except (TypeError, ValueError):
            conf = 0.0
        log.info("AI song guess: %r — %r version=%r conf=%.2f evidence=%r", artist, title,
                 _s(data.get("version")), conf, _s(data.get("evidence"))[:120])
        if not data.get("found") or not title or conf < 0.5:
            return None
        verified = await verify_song_itunes(self.http, artist, title)
        if not verified:
            log.info("AI guess not found in the Apple Music catalog → ignored")
            return None
        version = _s(data.get("version")).lower()
        if version and version not in ("original", "unknown", "n/a", "none"):
            verified["_note"] = _t("🎛 This Reel uses a {v} of:", v=version[:40])
        elif conf < 0.8:
            verified["_note"] = _t("🤔 Most likely:")
        return verified

    async def recognize_song(self, info: MediaInfo, workdir: Path,
                             hint: ResolvedPost | None = None) -> dict | None:
        """0) Instagram's licensed-track label → 1) AudD + Shazam race → 2) Shazam on sped-up /
        slowed versions + Gemini listening (verified in the iTunes catalog).
        Raises ExternalServiceError only if AudD failed AND nothing was found."""
        mp3 = await extract_audio_clip(info, workdir)  # MediaUnavailable for photos / silent clips
        loop = asyncio.get_running_loop()
        started = loop.time()

        if hint and hint.music_title and not hint.music_original:
            log.info("Song from the Instagram audio label: %s — %s", hint.music_artist, hint.music_title)
            return {"artist": hint.music_artist, "title": hint.music_title}

        label = ""
        if hint and hint.music_title:
            label = f"{hint.music_artist}: {hint.music_title} (user-uploaded original audio)"
        ai_task = (asyncio.create_task(self._ai_song(info, workdir, hint.caption if hint else "", label))
                   if self.s.ai_song_enabled else None)
        try:
            audd_task = asyncio.create_task(self.audd.recognize(mp3))
            names = {audd_task: "AudD", asyncio.create_task(self._shazam_once(info, workdir)): "Shazam"}
            for task in names:  # mark errors as retrieved even if the other engine wins first
                task.add_done_callback(lambda t: t.cancelled() or t.exception())
            pending: set[asyncio.Task] = set(names)
            audd_error: BaseException | None = None
            try:  # whichever fingerprint engine finds the song first wins
                while pending:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        if task.exception() is not None:
                            if task is audd_task:
                                audd_error = task.exception()
                            else:
                                log.warning("Shazam branch error: %s", task.exception())
                            continue
                        if task.result():
                            log.info("Song found by %s in %.1fs", names[task], loop.time() - started)
                            return task.result()
            finally:
                for task in pending:
                    task.cancel()

            if self.shazam:  # 0.8 / 0.88 undo "sped up" edits, 1.2 undoes "slowed + reverb";
                # plus a cleaned-up clip and another part of the video (bad audio, late chorus)
                mid = max(info.duration / 2 - 6, 0) if info.duration >= 20 else None
                attempts = [(0.8, None, False), (0.88, None, False), (1.2, None, False), (1.0, None, True)]
                if mid is not None:
                    attempts.append((1.0, mid, True))
                variants = await asyncio.gather(*(self._shazam_once(info, workdir, r, st, cl) for r, st, cl in attempts),
                                                return_exceptions=True)
                for (rate, _st, _cl), res in zip(attempts, variants, strict=True):
                    if res and not isinstance(res, BaseException):
                        log.info("Song found by Shazam (speed x%.2f, start=%s, clean=%s) in %.1fs", 1 / rate, _st,
                                 _cl, loop.time() - started)
                        if rate != 1.0:
                            res["_note"] = _t("⚡ This Reel uses a sped-up version of:" if rate < 1
                                              else "🐢 This Reel uses a slowed-down version of:")
                        return res
            if ai_task:
                res = await ai_task
                if res:
                    log.info("Song found by AI (verified) in %.1fs", loop.time() - started)
                    return res
            log.info("Song not recognised (AudD %s, Shazam %s, AI %s) in %.1fs",
                     "error" if audd_error else "miss", "miss" if self.shazam else "off",
                     "miss" if ai_task else "off", loop.time() - started)
            if audd_error:
                raise audd_error
            return None
        finally:
            if ai_task and not ai_task.done():
                ai_task.cancel()

    async def _music_branch(self, info: MediaInfo, workdir: Path, strict: bool,
                            hint: ResolvedPost | None = None) -> BranchResult:
        try:
            result = await self.recognize_song(info, workdir, hint)
        except MediaUnavailable as exc:
            if strict:
                raise
            no_sound = exc.user_message in (MSG_NO_AUDIO, MSG_PHOTO_NO_AUDIO)
            return BranchResult(skipped=True, text=_t("🎵 Song: no sound in this media") if no_sound else "")
        except ExternalServiceError as exc:
            if strict:
                raise
            log.warning("Music branch failed: %s", exc)
            return BranchResult(failed=True)
        if not result:
            return BranchResult(found=False)
        card = song_card(result)
        name = f"{_s(result.get('artist'))} — {_s(result.get('title'))}"
        return BranchResult(found=True, text=format_song(result, links=False), cards=[card],
                            finds=[("song", name, _s(result.get("album")), card["buttons"][0]["url"])],
                            summary=f"🎵 {name}")

    async def item_photos(self, items: list[dict], frames: list[bytes], workdir: Path) -> list[str]:
        """Crops each item and serves it for 24 h at an unguessable URL (for Google Lens)."""
        if not self.s.public_url:
            return [""] * len(items)

        async def one(it: dict) -> str:
            try:
                idx = int(it.get("frame") or 0)
            except (TypeError, ValueError):
                idx = 0
            frame = frames[idx] if 0 <= idx < len(frames) else frames[0]
            crop = await crop_item(frame, it.get("box_2d") or [], workdir)
            if not crop:
                return ""
            token = secrets.token_urlsafe(12)
            self.photos[token] = crop
            return f"{self.s.public_url}/img/{token}.jpg"

        return list(await asyncio.gather(*(one(it) for it in items)))

    async def _vision_branch(self, action: str, info: MediaInfo, workdir: Path, strict: bool,
                             hint: ResolvedPost | None = None) -> BranchResult:
        context = ""
        if hint:
            parts = []
            if hint.location:
                parts.append(f"Instagram location tag: {hint.location}")
            if hint.caption:
                parts.append(f"Caption: {hint.caption}")
            context = "\n".join(parts)
        missing: list[str] = []
        try:
            frames = await extract_frames(info, workdir, 960 if action in (FIND_ITEM, FIND_ALL) else 640)
            if action == FIND_ALL:
                # Three focused requests in parallel instead of one big one: each task gets the
                # model's full attention (better answers), and it takes no longer.
                got = await asyncio.gather(*(self.vision.analyze(t, frames, context)
                                             for t in (FIND_MOVIE, FIND_ITEM, FIND_PLACE)), return_exceptions=True)
                errors = [g for g in got if isinstance(g, BaseException)]
                for g in errors:
                    if not isinstance(g, ExternalServiceError):
                        raise g
                if len(errors) == 3:
                    raise errors[0]
                data = {"movie": got[0], "items": got[1], "place": got[2]}
                data = {k: ({} if isinstance(v, BaseException) else v) for k, v in data.items()}
            else:
                data = await self.vision.analyze(action, frames, context)
        except MediaUnavailable:
            if strict:
                raise
            return BranchResult(skipped=True)
        except ExternalServiceError as exc:
            if strict:
                raise
            log.warning("Vision branch failed: %s", exc)
            return BranchResult(failed=True)

        if action == FIND_ALL:
            movie, items, place = data.get("movie") or {}, filter_items(data.get("items") or {}), data.get("place") or {}
            if not movie_is_found(movie):
                missing.append(_t("🎬 Movie/show: not identified"))
            if not items:
                missing.append(_t("👕 Items: nothing identifiable"))
            if not place_is_found(place):
                missing.append(_t("📍 Place: not recognised"))
        else:
            movie = data if action == FIND_MOVIE else {}
            items = filter_items(data) if action == FIND_ITEM else []
            place = data if action == FIND_PLACE else {}
        if place and hint and hint.location_lat is not None and hint.location_lng is not None:
            place["_lat"], place["_lng"] = hint.location_lat, hint.location_lng

        res = BranchResult()
        texts: list[str] = []
        summaries: list[str] = []
        if movie_is_found(movie):
            title, year = _s(movie.get("title")), _s(movie.get("year"))
            card = movie_card(movie)
            texts.append(format_movie(movie))
            res.cards.append(card)
            head = title + (f" ({year})" if year else "")
            res.finds.append(("movie", head, _s(movie.get("kind")), card["buttons"][1]["url"]))
            summaries.append(f"🎬 {head}")
        if items:
            photos = await self.item_photos(items, frames, workdir)
            cards = item_cards(items, photos)
            texts.append(format_items(items))
            res.cards += cards
            for card, it in zip(cards, items, strict=True):
                name = card["title"].split(". ", 1)[-1]
                res.finds.append(("item", name, _s(it.get("search_query")), shopping_url(it)))
            summaries.append("👕 " + "; ".join(c["title"].split(". ", 1)[-1] for c in cards[:3]))
        if place_is_found(place):
            card = place_card(place)
            texts.append(format_place(place))
            res.cards.append(card)
            res.finds.append(("place", _s(place.get("name")), place_where(place), card["buttons"][0]["url"]))
            summaries.append(f"📍 {_s(place.get('name'))}" + (f", {place_where(place)}" if place_where(place) else ""))
        res.found = bool(texts)
        if res.found and missing:
            texts.append("\n".join(missing))
        res.text = "\n\n".join(texts)
        res.summary = "\n".join(summaries)
        return res

    # ---------------------------------------------------------------- payments

    async def notify_payment(self, igsid: str, template: str, values: dict | None = None) -> None:
        try:
            CURRENT_LANG.set(await self.db.get_lang(igsid) or "en")
        except Exception:  # noqa: BLE001
            pass
        text = _t(template, **(values or {}))
        try:
            await self.ig.send_text(igsid, text)
        except GraphAPIError as exc:
            if exc.outside_messaging_window:
                log.info("User %s is outside the 24h window; notice queued", igsid)
            else:
                log.error("Payment DM to %s failed: %s; notice queued", igsid, exc)
            try:
                await self.db.set_pending_notice(igsid, text)
            except Exception as db_exc:  # noqa: BLE001
                log.error("Could not queue pending notice for %s: %s", igsid, db_exc)

    # ---------------------------------------------------------------- token refresh

    async def bootstrap_token(self) -> None:
        """Keeps a self-refreshing copy of the Instagram token in the DB."""
        seed = hashlib.sha256(self.s.ig_access_token.encode()).hexdigest()
        stored_seed = await self.db.kv_get("ig_token_seed")
        stored_token = await self.db.kv_get("ig_access_token")
        if stored_seed and stored_seed["value"] == seed and stored_token:
            self.ig.token = stored_token["value"]
        else:
            await self.db.kv_set("ig_token_seed", seed)
            await self.db.kv_set("ig_access_token", self.s.ig_access_token)
            self.ig.token = self.s.ig_access_token

    async def ensure_webhook_subscription(self) -> None:
        """Unified login (graph.facebook.com + system-user token): the dashboard toggle is not
        always enough, so the account is subscribed to the app's webhooks on every start."""
        if "graph.facebook.com" not in self.s.graph_api_base:
            return
        if not self.s.ig_account_id.isdigit():
            log.error("IG_ACCOUNT_ID must be the numeric Instagram account ID when GRAPH_API_BASE "
                      "is graph.facebook.com (Unified login); got %r", self.s.ig_account_id)
            return
        for fields in ("messages,messaging_postbacks,comments,mentions", "messages,messaging_postbacks,comments",
                       "messages,comments"):
            try:
                await self.ig.subscribe_app(fields)
                log.info("Instagram account %s subscribed to webhooks: %s", self.s.ig_account_id, fields)
                return
            except (GraphAPIError, httpx.HTTPError) as exc:
                log.warning("Webhook subscription with fields %s failed: %s", fields, exc)
        log.error("Could not subscribe the Instagram account to webhooks — check token permissions "
                  "(instagram_basic, instagram_manage_messages, instagram_manage_comments)")

    async def token_refresh_loop(self) -> None:
        if "graph.instagram.com" not in self.s.graph_api_base:
            return  # Page tokens (Facebook Login) don't use this refresh flow
        while True:
            try:
                rec = await self.db.kv_get("ig_access_token")
                age = datetime.now(timezone.utc) - rec["updated_at"] if rec else timedelta(days=999)
                if age > timedelta(days=5):
                    new_token = await self.ig.refresh_long_lived_token()
                    await self.db.kv_set("ig_access_token", new_token)
                    self.ig.token = new_token
                    log.info("Instagram access token refreshed")
            except Exception as exc:  # noqa: BLE001
                log.warning("Token refresh skipped: %s", exc)
            await asyncio.sleep(12 * 3600)


# =============================================================================
# Privacy policy page (served at /privacy — use this URL in the Meta dashboard)
# =============================================================================

PRIVACY_EFFECTIVE_DATE = "November 27, 2026"

PRIVACY_TEMPLATE = string.Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$bot_name — Privacy Policy</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
         max-width: 760px; margin: 40px auto; padding: 0 20px; line-height: 1.6; color: #1a1a1a; }
  h1 { font-size: 28px; margin-bottom: 4px; }
  h2 { font-size: 20px; margin-top: 32px; }
  .muted { color: #666; font-size: 14px; }
  li { margin-bottom: 6px; }
  code { background: #f2f2f2; padding: 1px 5px; border-radius: 4px; }
</style>
</head>
<body>
<h1>$bot_name — Privacy Policy</h1>
<p class="muted">Effective date: $effective_date</p>

<p>$bot_name ("we", "us") is an Instagram Direct Message service operated by $operator via the
Instagram account $handle_text. You forward us a Reel, video, photo or link, choose what to find,
and we identify the song, the movie or TV show, or the clothes and items in it. This policy explains
what data we process, why, and how you can have it deleted.</p>

<h2>1. Data we collect</h2>
<ul>
  <li><b>Your Instagram-scoped ID</b> — a number Instagram assigns to you for our account only.
      We do not receive or store your name, email address or phone number.</li>
  <li><b>The link to the last media you sent us</b> and its type (video, photo or audio).</li>
  <li><b>Your search balance</b>, subscription end date (if any) and the date you first messaged us.</li>
  <li><b>Payment records</b> — your Stripe customer ID, which package you bought and when, if paid plans are enabled.
      We never receive or store card details; payments are handled by our payment provider.</li>
  <li><b>Comments containing the word "FIND"</b> under our posts — used only to send you a private reply.
      We do not store the comment text.</li>
  <li><b>Your lists</b> — titles and links of the songs, movies, items and places we found for you
      (up to 100 per category), so you can see them again with "My lists".</li>
  <li><b>Your chosen language</b> and, if you use invites, your invite code. The fact that you joined with
      someone's invite is stored only as a one-way hash.</li>
  <li><b>Comments where you @mention us</b> under public posts — we read the comment and that post's video
      to reply publicly in the comments. We do not store the comment text.</li>
  <li><b>Technical server logs</b> (for example, request times and errors), kept for a limited time
      for security and troubleshooting.</li>
</ul>

<h2>2. How we process your media</h2>
<p>When you start a search, the media is downloaded to our server temporarily. We cut an 8-second audio
clip and/or 3 still frames from it, send them for recognition, and <b>delete all files right after
processing</b>, usually within seconds. We do not keep your videos or photos.</p>

<h2>3. Who we share data with</h2>
<p>We share data only to provide the service. We do <b>not</b> sell your data or use it for advertising.</p>
<ul>
  <li><b>Meta Platforms (Instagram Messaging API)</b> — to receive and send messages.</li>
  <li><b>Google (Gemini API)</b> — still frames (and the Reel's caption and location tag), to recognize movies,
      shows, items and places; a short audio clip when other song search fails.</li>
  <li><b>Shazam</b> — a short audio clip, to recognize the song.</li>
  <li><b>Apple (iTunes Search)</b> — the song title, to check that a recognized song really exists.</li>
  <li><b>AudD</b> — an 8-second audio clip, to recognize the song.</li>
  <li><b>Railway</b> — server hosting (United States).</li>
  <li><b>Supabase</b> — database hosting (United States).</li>
  <li><b>Stripe</b> — to process purchases (card details go only to Stripe, never to us), under its own privacy policy.</li>
</ul>

<h2>4. Why we process data</h2>
<p>To perform the searches you request, keep track of free and paid searches, confirm payments,
reply to your messages and comments, and keep the service secure.</p>

<h2>5. How long we keep data</h2>
<ul>
  <li>Your ID, balance and dates are kept while you use the service or until you ask us to delete them.</li>
  <li>Payment records may be kept longer where accounting or tax law requires it.</li>
  <li>Media files are not stored (see section 2). Small cut-outs of items we found are kept up to 24 hours at an
      unguessable link, so the "Search by photo" button can work.</li>
</ul>

<h2 id="delete">6. How to delete your data</h2>
<p>You can delete your data at any time, in either of these ways:</p>
<ul>
  <li>Send <code>delete my data</code> to $handle_text in Instagram Direct. Deletion happens immediately
      and you get a confirmation.</li>
  <li>Or email <a href="mailto:$email">$email</a> from any address, including your Instagram username.
      We delete the data within 30 days and confirm by reply.</li>
</ul>
<p>Deletion removes your ID, balance, subscription and media link. Remaining searches and subscription
time are lost. To prevent abuse of free searches, we keep only a one-way (irreversible) hash of your ID,
which cannot be used to identify you.</p>

<h2>7. Your rights</h2>
<p>You can ask what data we hold about you, ask us to correct or delete it, and stop using the service at
any time. If you are in the European Union or the UK, you have rights under the GDPR, including access,
rectification, erasure, restriction and objection, and the right to complain to your data protection
authority. If you are a California resident, you have rights under the CCPA, including the right to know
and to delete; we do not sell or share personal information for advertising. Contact us at
<a href="mailto:$email">$email</a> to exercise any of these rights.</p>

<h2>8. Children</h2>
<p>The service is not intended for anyone under 13 (under 16 in the EU). We do not knowingly collect
data from children. If you believe a child has used the service, contact us and we will delete the data.</p>

<h2>9. Security</h2>
<p>All connections are encrypted (HTTPS/TLS), database access is restricted, and secret keys are not
stored in our code.</p>

<h2>10. Cookies</h2>
<p>The Instagram service does not use cookies.</p>

<h2>11. Changes to this policy</h2>
<p>We will post any changes on this page and update the effective date above.</p>

<h2>12. Contact</h2>
<p>$operator — <a href="mailto:$email">$email</a></p>
</body>
</html>
""")


def render_privacy_page(s: Settings) -> str:
    handle = f"@{s.instagram_handle}" if s.instagram_handle else "our Instagram account"
    return PRIVACY_TEMPLATE.substitute(
        bot_name=html.escape(s.bot_name),
        effective_date=PRIVACY_EFFECTIVE_DATE,
        operator=html.escape(s.operator_name or s.bot_name),
        handle_text=html.escape(handle),
        email=html.escape(s.contact_email or "the email listed on our Instagram profile"),
    )


# =============================================================================
# FastAPI application
# =============================================================================

bot: Bot | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot
    settings = Settings.from_env()
    for binary in ("ffmpeg", "ffprobe"):
        if not shutil.which(binary):
            raise RuntimeError(f"{binary} is not installed")

    http = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=3.0, pool=3.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20, keepalive_expiry=30),
    )
    db = Database(settings.database_url, settings.free_credits)
    await db.connect()
    ig = InstagramClient(http, settings.graph_api_base, settings.ig_account_id, settings.ig_access_token)
    audd = AudDClient(http, settings.audd_api_token, settings.audd_timeout_sec)
    vision = VisionClient(
        settings.gemini_api_key,
        [settings.gemini_primary_model, settings.gemini_fallback_model,
         *settings.gemini_extra_models.split(",")],
        settings.gemini_timeout_sec,
        settings.gemini_thinking_level,
        backup=(settings.backup_ai_url, settings.backup_ai_key, settings.backup_ai_model),
    )
    shazam: ShazamClient | None = None
    if settings.shazam_enabled:
        try:
            shazam = ShazamClient()
        except Exception as exc:  # noqa: BLE001 — Shazam is optional
            log.warning("Shazam disabled: %s", exc)
    bot = Bot(settings, http, db, ig, audd, vision, shazam)
    await bot.bootstrap_token()
    await bot.ensure_webhook_subscription()
    refresh_task = asyncio.create_task(bot.token_refresh_loop())
    poll_task = asyncio.create_task(bot.inbox_poll_loop())
    await vision.discover_models()
    if bot.stripe:
        await bot.stripe.load_prices(list(settings.stripe_prices.values()))
        if not settings.public_url:
            log.error("Stripe is configured but PUBLIC_URL is unknown — set PUBLIC_URL=https://your-domain")
        if not settings.stripe_webhook_secret:
            log.error("STRIPE_WEBHOOK_SECRET is missing — payments will NOT be credited")
        log.info("Stripe payments ON: %s", ", ".join(f"{k}={bot.stripe.price_labels.get(v, '?')}"
                                                      for k, v in settings.stripe_prices.items()))
    if ReelExtractor.impersonation_available():
        log.info("Reel downloader: browser impersonation ON")
    else:
        log.warning("Reel downloader: browser impersonation OFF (curl_cffi missing) → more Reels fail")
    log.info("Bot started [build %s] (AI chain: %s)", BUILD, vision.describe)
    try:
        yield
    finally:
        refresh_task.cancel()
        poll_task.cancel()
        await vision.aclose()
        await http.aclose()
        await db.close()
        bot = None


app = FastAPI(title="Reels Finder Bot", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


async def _run_safely(coro_fn: Callable[..., Awaitable[None]], *args: Any) -> None:
    try:
        await coro_fn(*args)
    except Exception:  # noqa: BLE001
        log.exception("Background task %s failed", getattr(coro_fn, "__name__", coro_fn))


def _verify_meta_signature(raw: bytes, header: str | None, secrets: list[str]) -> bool:
    if not header or not header.startswith("sha256="):
        return False
    received = header.split("=", 1)[1]
    return any(
        hmac.compare_digest(hmac.new(s.encode(), raw, hashlib.sha256).hexdigest(), received) for s in secrets
    )


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "service": "reels-finder-bot"}


@app.get("/privacy", response_class=HTMLResponse)
async def privacy() -> HTMLResponse:
    settings = bot.s if bot else Settings.from_env()
    return HTMLResponse(render_privacy_page(settings))


@app.get("/health")
async def health() -> JSONResponse:
    ok = bool(bot and await bot.db.ping())
    return JSONResponse({"status": "ok" if ok else "degraded", "db": ok}, status_code=200 if ok else 503)


@app.get("/webhook")
async def verify_webhook(request: Request) -> Response:
    params = request.query_params
    if (
        bot
        and params.get("hub.mode") == "subscribe"
        and hmac.compare_digest(params.get("hub.verify_token", ""), bot.s.ig_verify_token)
    ):
        return PlainTextResponse(params.get("hub.challenge", ""))
    return PlainTextResponse("Verification failed", status_code=403)


@app.post("/webhook")
async def receive_webhook(request: Request, background: BackgroundTasks) -> Response:
    raw = await request.body()
    assert bot is not None
    if not _verify_meta_signature(raw, request.headers.get("x-hub-signature-256"), bot.s.ig_app_secrets):
        log.warning("Rejected webhook with invalid signature")
        return PlainTextResponse("invalid signature", status_code=401)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return PlainTextResponse("EVENT_RECEIVED")

    if payload.get("object") not in ("instagram", "page"):
        return PlainTextResponse("EVENT_RECEIVED")

    for entry in payload.get("entry") or []:
        own_id = _s(entry.get("id"))
        if own_id and own_id != bot.s.ig_account_id and own_id not in bot.seen_accounts:
            bot.seen_accounts.add(own_id)
            log.warning("Webhook is for Instagram account id=%s (IG_ACCOUNT_ID=%s). "
                        "With Unified login set IG_ACCOUNT_ID to this id.", own_id, bot.s.ig_account_id)

        for ev in entry.get("messaging") or []:
            mid = _s((ev.get("message") or {}).get("mid") or (ev.get("postback") or {}).get("mid"))
            if not mid:
                continue  # read receipts, reactions, typing, etc.
            sender_id = _s((ev.get("sender") or {}).get("id"))
            ts = int(ev.get("timestamp") or 0)
            if bot.dedup.seen(f"m:{mid}"):
                log.info("Duplicate webhook ignored: %s", mid)
                continue
            bot.mark_seen_event(sender_id, ts)
            background.add_task(_run_safely, bot.handle_messaging_event, ev, own_id)

        for change in entry.get("changes") or []:
            if change.get("field") == "mentions":
                value = change.get("value") or {}
                key = _s(value.get("comment_id")) or _s(value.get("media_id"))
                if key and not bot.dedup.seen(f"mn:{key}"):
                    background.add_task(_run_safely, bot.handle_mention, value, own_id)
                continue
            if change.get("field") != "comments":
                continue
            value = change.get("value") or {}
            cid = _s(value.get("id"))
            if not cid or bot.dedup.seen(f"c:{cid}"):
                continue
            background.add_task(_run_safely, bot.handle_comment, value, own_id)

    return PlainTextResponse("EVENT_RECEIVED")


_PAGE = string.Template("""<!DOCTYPE html><html lang="$lang"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>$title</title>
<style>body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#0f0f12;color:#f2f2f2;display:flex;
min-height:100vh;align-items:center;justify-content:center;margin:0;padding:24px;text-align:center}
main{max-width:420px}h1{font-size:28px;margin:0 0 12px}p{color:#bdbdc7;line-height:1.5;margin:0 0 24px}
a{display:inline-block;background:linear-gradient(90deg,#f58529,#dd2a7b,#8134af);color:#fff;text-decoration:none;
padding:14px 26px;border-radius:14px;font-weight:600}</style></head>
<body><main><h1>$title</h1><p>$text</p>$button</main></body></html>""")


def _page(lang: str, title: str, text: str, link: str = "", status: int = 200) -> HTMLResponse:
    token = CURRENT_LANG.set(lang if lang in LANGUAGES else "en")
    try:
        button = f'<a href="{html.escape(link)}">{html.escape(_t("Back to Instagram"))}</a>' if link else ""
        body = _PAGE.substitute(lang=html.escape(lang), title=html.escape(_t(title)), text=html.escape(_t(text)),
                                button=button)
    finally:
        CURRENT_LANG.reset(token)
    return HTMLResponse(body, status_code=status, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})


@app.get("/pay/{token}/{plan}")
async def pay(token: str, plan: str) -> Response:
    """Personal payment link from the bot → fresh Stripe Checkout (or Customer Portal) session."""
    assert bot is not None
    user = await bot.db.user_by_pay_token(token) if bot.stripe else None
    if not user or plan not in ("s", "l", "m", "y", "manage"):
        return _page("en", "🤔", "This payment link is invalid. Open a fresh one from the bot.", status=404)
    try:
        url = await bot.checkout_url(user, plan)
    except StripeError as exc:
        log.error("Checkout for %s (%s) failed: %s", user["igsid"], plan, exc)
        url = ""
    if not url:
        return _page(user["lang"] or "en", "😕", "Payments are temporarily unavailable. Please try again in a "
                                                 "few minutes.", bot.bot_link(), status=503)
    return RedirectResponse(url, status_code=303, headers={"Cache-Control": "no-store"})


@app.get("/img/{name}")
async def item_photo(name: str) -> Response:
    assert bot is not None
    data = bot.photos.get(name.removesuffix(".jpg"))
    if not data:
        return Response(status_code=404)
    return Response(data, media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400", "X-Robots-Tag": "noindex"})


@app.get("/paid")
async def paid(t: str = "", ok: str = "1") -> HTMLResponse:
    assert bot is not None
    user = await bot.db.user_by_pay_token(t)
    lang = (user["lang"] if user else None) or "en"
    link = bot.bot_link() if bot.s.instagram_handle else ""
    if ok == "0":
        return _page(lang, "👌", "Payment cancelled — nothing was charged.", link)
    return _page(lang, "✅ Payment received!", "Go back to Instagram — your purchase is already active.", link)


@app.post("/stripe-webhook")
async def stripe_webhook(request: Request) -> Response:
    raw = await request.body()
    assert bot is not None
    if not bot.stripe or not bot.s.stripe_webhook_secret:
        return JSONResponse({"error": "stripe is not configured"}, status_code=503)
    if not verify_stripe_signature(raw, request.headers.get("stripe-signature", ""), bot.s.stripe_webhook_secret):
        log.warning("Rejected Stripe webhook: bad signature")
        return JSONResponse({"error": "invalid signature"}, status_code=400)
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    try:
        result = await bot.handle_stripe_event(event)
    except Exception:  # noqa: BLE001 — non-2xx makes Stripe retry later
        log.exception("Stripe event %s (%s) failed", event.get("id"), event.get("type"))
        return JSONResponse({"error": "processing failed"}, status_code=500)
    return JSONResponse({"status": result})


@app.post("/payment-webhook")
async def payment_webhook(request: Request, background: BackgroundTasks) -> Response:
    raw = await request.body()
    assert bot is not None
    if not bot.s.lemon_signing_secret:
        return JSONResponse({"error": "payments are not configured"}, status_code=503)
    signature = request.headers.get("x-signature", "")
    expected = hmac.new(bot.s.lemon_signing_secret.encode(), raw, hashlib.sha256).hexdigest()
    if not signature or not hmac.compare_digest(expected, signature):
        log.warning("Rejected Lemon Squeezy webhook: bad signature")
        return JSONResponse({"error": "invalid signature"}, status_code=401)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    meta = payload.get("meta") or {}
    event_name = _s(meta.get("event_name"))
    igsid = _s((meta.get("custom_data") or {}).get("igsid"))
    data = payload.get("data") or {}
    attrs = data.get("attributes") or {}
    event_key = f"{event_name}:{data.get('type')}:{data.get('id')}"

    if not igsid:
        log.warning("Lemon event %s without igsid custom data — ignored", event_key)
        return JSONResponse({"status": "ignored", "reason": "no igsid"})

    try:
        if event_name == "order_created":
            if _s(attrs.get("status")) != "paid":
                return JSONResponse({"status": "ignored", "reason": "order not paid"})
            variant_id = _s((attrs.get("first_order_item") or {}).get("variant_id"))
            if variant_id and variant_id in bot.s.lemon_subscription_variant_ids:
                # First subscription payment is handled by subscription_payment_success.
                return JSONResponse({"status": "ignored", "reason": "subscription order"})
            if bot.s.lemon_credit_pack_variant_ids and variant_id not in bot.s.lemon_credit_pack_variant_ids:
                log.warning("Order %s for unknown variant %s ignored", data.get("id"), variant_id)
                return JSONResponse({"status": "ignored", "reason": "unknown variant"})

            balance = await bot.db.apply_credit_pack(event_key, igsid, event_name, bot.s.credits_per_pack)
            if balance is None:
                return JSONResponse({"status": "duplicate"})
            log.info("Credit pack applied: igsid=%s balance=%s", igsid, balance)
            background.add_task(
                _run_safely, bot.notify_payment, igsid,
                MSG_PAYMENT_CREDITS, {"n": bot.s.credits_per_pack, "balance": balance},
            )
            return JSONResponse({"status": "ok", "credits": balance})

        if event_name == "subscription_payment_success":
            if _s(attrs.get("status")) not in ("paid", ""):
                return JSONResponse({"status": "ignored", "reason": "invoice not paid"})
            until = await bot.db.apply_subscription(event_key, igsid, event_name, bot.s.subscription_days)
            if until is None:
                return JSONResponse({"status": "duplicate"})
            log.info("Subscription extended: igsid=%s until=%s", igsid, until.isoformat())
            if _s(attrs.get("billing_reason")) in ("initial", ""):
                background.add_task(
                    _run_safely, bot.notify_payment, igsid, MSG_PAYMENT_SUB, {"date": f"{until:%b %d, %Y}"}
                )
            return JSONResponse({"status": "ok", "sub_until": until.isoformat()})

    except Exception:  # noqa: BLE001 — 500 makes Lemon Squeezy retry
        log.exception("Payment webhook processing failed for %s", event_key)
        return JSONResponse({"error": "processing failed"}, status_code=500)

    return JSONResponse({"status": "ignored", "event": event_name})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), workers=1)
